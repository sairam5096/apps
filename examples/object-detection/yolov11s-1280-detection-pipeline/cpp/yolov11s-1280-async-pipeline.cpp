/**
 * yolov11s-1280-async-pipeline.cpp
 *
 * RTSP detection pipeline using the SDK's async push/pull inference API.
 *
 *   decode : neatdecoder SiMa HW H.264 via GStreamer appsink (~2ms vs ~11ms FFmpeg SW)
 *   infer  : async runner.push() / runner.pull() in separate threads
 *   encode : neatencoder SiMa HW H.264 with non-blocking dispatch thread
 *
 * Push and pull run in separate threads — safe per SDK contract:
 *   "push() from one thread and pull() from another is safe."
 *
 * With queue_depth=3 (RunPreset::Realtime, OverflowPolicy::Block), Thread A
 * can pre-fill up to 3 frames into the MLA queue while Thread B is draining
 * it. This keeps the MLA continuously fed even when decode jitters.
 *
 * 5-thread pipeline:
 *
 *   Thread A0 [neatdecoder HW H.264 via GStreamer     ] → decode_q(8)
 *   Thread A1 [preproc + runner.push()               ] → push_rec_q(4)
 *   Thread B [runner.pull() → tensors               ] → fast_post_q(4)
 *   Thread C [post-process + draw                   ] → fast_encode_q(16)
 *   Thread D [dispatch, non-blocking                ] → hw_write_q(1)
 *   Thread E [neatencoder write ~20ms               ]
 *
 * Timing stages reported:
 *   read     – cap.read()
 *   preproc  – resize/format conversion
 *   push     – runner.push() call time
 *   pull     – runner.pull() call time (≈ MLA latency)
 *   post     – decode_yolov11_postsurgery + NMS + draw
 *   encode   – hw_writer.write()
 *
 * Usage:
 *   yolov11s-1280-async-pipeline <model.tar.gz> <labels.txt>
 *     --rtsp <URL> [--insight <host>] [--frames N]
 *     [--classes <csv>] [--video-port N] [--meta-port N]
 */

#include "neat.h"
#include "support/object_detection/obj_detection_utils.h"
#include <nodes/io/MetadataSender.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>

#ifdef __ARM_NEON
#  include <arm_neon.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

// Build an NV12 simaai::neat::Tensor wrapping a CPU-owned copy of nv12_data.
// w×h must be even. data must point to w*h*3/2 bytes (Y plane then UV plane).
// Build NV12 tensor from a pre-allocated pool storage (no allocation, no memcpy).
// Caller must have already written NV12 data into storage->data.
// Uses shared_ptr (not move) so the pool slot stays alive for reuse.
static simaai::neat::Tensor make_nv12_tensor(
    std::shared_ptr<simaai::neat::TensorBuffer> storage, int w, int h) {
  simaai::neat::Tensor t;
  t.storage          = storage;  // shared: pool still holds its ref
  t.dtype            = simaai::neat::TensorDType::UInt8;
  t.layout           = simaai::neat::TensorLayout::HW;
  t.shape            = {h, w};
  t.strides_bytes    = {w, 1};
  t.byte_offset      = 0;
  t.device           = {simaai::neat::DeviceType::CPU, 0};
  t.read_only        = true;

  simaai::neat::ImageSpec img;
  img.format         = simaai::neat::ImageSpec::PixelFormat::NV12;
  t.semantic.image   = img;

  simaai::neat::Plane y_plane;
  y_plane.role         = simaai::neat::PlaneRole::Y;
  y_plane.shape        = {h, w};
  y_plane.strides_bytes = {w, 1};
  y_plane.byte_offset  = 0;

  simaai::neat::Plane uv_plane;
  uv_plane.role         = simaai::neat::PlaneRole::UV;
  uv_plane.shape        = {h / 2, w};
  uv_plane.strides_bytes = {w, 1};
  uv_plane.byte_offset  = static_cast<int64_t>(w) * h;

  t.planes = {std::move(y_plane), std::move(uv_plane)};
  return t;
}
using Ms    = std::chrono::duration<double, std::milli>;

namespace {

// ── Constants ────────────────────────────────────────────────────────────────
constexpr int   kInputW    = 1280;
constexpr int   kInputH    = 720;
constexpr float kLbGain    = 1.0f;
constexpr float kLbPadTop  = 280.0f;
constexpr float kLbPadLeft = 0.0f;
constexpr float kMinScore  = 0.60f;
constexpr float kNmsIou    = 0.50f;
constexpr int   kMaxDet    = 24;
constexpr int   kTimeoutMs = 5000;

// ── Timing ───────────────────────────────────────────────────────────────────
struct TimingAcc {
  double sum{0}, mn{1e18}, mx{-1.0}; int n{0};
  void add(double v) { sum += v; mn = std::min(mn,v); mx = std::max(mx,v); ++n; }
  double mean() const { return n > 0 ? sum/n : 0.0; }
};

// ── Bounded blocking queue ────────────────────────────────────────────────────
template<typename T>
class PipelineQueue {
public:
  explicit PipelineQueue(size_t cap) : cap_(cap) {}
  // Blocking push — also tracks how long the caller blocked (backpressure_ms out).
  void push(T item, double* backpressure_ms = nullptr) {
    auto t0 = Clock::now();
    std::unique_lock<std::mutex> lk(mu_);
    cv_push_.wait(lk, [&]{ return q_.size() < cap_ || closed_; });
    if (backpressure_ms)
      *backpressure_ms = Ms(Clock::now() - t0).count();
    if (closed_) return;
    q_.push(std::move(item));
    size_t sz = q_.size();
    if (sz > hwm_) hwm_ = sz;
    cv_pop_.notify_one();
  }
  // Non-blocking push: returns false immediately if queue is full or closed.
  bool try_push(T item) {
    std::lock_guard<std::mutex> lk(mu_);
    if (q_.size() >= cap_ || closed_) return false;
    q_.push(std::move(item));
    size_t sz = q_.size();
    if (sz > hwm_) hwm_ = sz;
    cv_pop_.notify_one();
    return true;
  }
  bool pop(T& out) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_pop_.wait(lk, [&]{ return !q_.empty() || closed_; });
    if (q_.empty()) return false;
    out = std::move(q_.front()); q_.pop();
    cv_push_.notify_one();
    return true;
  }
  void close() {
    std::lock_guard<std::mutex> lk(mu_);
    closed_ = true;
    cv_pop_.notify_all(); cv_push_.notify_all();
  }
  size_t capacity()   const { return cap_; }
  size_t high_water() const { return hwm_.load(); }
  bool   is_closed()  const {
    std::lock_guard<std::mutex> lk(mu_); return closed_;
  }
private:
  std::queue<T> q_; mutable std::mutex mu_;
  std::condition_variable cv_push_, cv_pop_;
  size_t cap_; bool closed_{false};
  std::atomic<size_t> hwm_{0};
};

// ── Inter-stage data structs ──────────────────────────────────────────────────

// Thread A0 → Thread A1: raw decoded frame + decode timing.
// Holds GstSample* (ref-counted) to avoid an A0 memcpy.  Thread A1 maps the
// buffer, copies into the pre-allocated storage pool, then unrefs.
struct DecodeRecord {
  GstSample* sample{nullptr};   // A1 must call gst_sample_unref()
  double     read_ms{0};
};

// Thread A1 → Thread B: pushed frame record (display + timing context).
// Must stay FIFO-aligned with runner's internal push queue so that Thread B's
// pull() result matches the display frame popped from push_rec_q.
struct PushRecord {
  cv::Mat           display;      // rendered frame for boxes overlay
  cv::Mat           model_in;     // 1280×720 BGR copy kept for the push call
  double            read_ms{0}, preproc_ms{0};
  Clock::time_point push_time{};  // wall clock when runner.push() completed
  int               idx{0};
};

// Thread B → Thread C: pull result bundled with display + timing.
struct PostIn {
  cv::Mat                  display;
  simaai::neat::TensorList tensors;
  double                   read_ms{0}, preproc_ms{0}, push_ms{0}, pull_ms{0};
  int                      idx{0};
};

// ── Tensor decode utilities (same as hw-pipeline) ─────────────────────────────
struct TensorHWC {
  int h{0}, w{0}, c{0};
  std::vector<uint8_t> raw;
  const float* floats() const noexcept { return reinterpret_cast<const float*>(raw.data()); }
};

#ifdef __ARM_NEON
#include <arm_neon.h>

// Read src once, write to two destinations — saves one full read pass vs two
// separate memcpy calls.  On bandwidth-bound Cortex-A55, this cuts memory
// traffic by ~25% (3 passes vs 4).  Processes 64 bytes/iteration (4x uint8x16).
static void neon_tee_copy(void* __restrict__ dst1, void* __restrict__ dst2,
                          const void* __restrict__ src, size_t n) {
  auto*       d1 = static_cast<uint8_t*>(dst1);
  auto*       d2 = static_cast<uint8_t*>(dst2);
  const auto* s  = static_cast<const uint8_t*>(src);
  for (; n >= 64; n -= 64, s += 64, d1 += 64, d2 += 64) {
    uint8x16_t v0 = vld1q_u8(s),    v1 = vld1q_u8(s+16),
               v2 = vld1q_u8(s+32), v3 = vld1q_u8(s+48);
    vst1q_u8(d1,    v0); vst1q_u8(d1+16, v1);
    vst1q_u8(d1+32, v2); vst1q_u8(d1+48, v3);
    vst1q_u8(d2,    v0); vst1q_u8(d2+16, v1);
    vst1q_u8(d2+32, v2); vst1q_u8(d2+48, v3);
  }
  if (n) { std::memcpy(d1, s, n); std::memcpy(d2, s, n); }
}

static void argmax80_neon(const float* __restrict__ p, float& best_val, int& best_idx) {
  float32x4_t v0 = vld1q_f32(p+0),  v1 = vld1q_f32(p+4),
              v2 = vld1q_f32(p+8),  v3 = vld1q_f32(p+12);
  for (int i = 16; i < 80; i += 16) {
    v0 = vmaxq_f32(v0, vld1q_f32(p+i+0));  v1 = vmaxq_f32(v1, vld1q_f32(p+i+4));
    v2 = vmaxq_f32(v2, vld1q_f32(p+i+8));  v3 = vmaxq_f32(v3, vld1q_f32(p+i+12));
  }
  best_val = vmaxvq_f32(vmaxq_f32(vmaxq_f32(v0,v1), vmaxq_f32(v2,v3)));
  best_idx = -1;
  for (int i = 0; i < 80; ++i) if (p[i] == best_val) { best_idx = i; break; }
}
#endif

static TensorHWC tensor_to_hwc_f32(const simaai::neat::Tensor& t) {
  if (t.dtype != simaai::neat::TensorDType::Float32)
    throw std::runtime_error("expected Float32 tensor");
  auto shape = t.shape;
  while (shape.size() > 3 && shape.front() == 1) shape.erase(shape.begin());
  TensorHWC out;
  if (shape.size() == 3) { out.h = shape[0]; out.w = shape[1]; out.c = shape[2]; }
  else throw std::runtime_error("unexpected tensor rank: " + std::to_string(t.shape.size()));
  out.raw = t.copy_dense_bytes_tight();
  const size_t expected = (size_t)out.h * out.w * out.c * sizeof(float);
  if (out.raw.size() < expected) throw std::runtime_error("tensor too small");
  return out;
}

static float iou_xyxy(const objdet::Box& a, const objdet::Box& b) {
  float xx1 = std::max(a.x1,b.x1), yy1 = std::max(a.y1,b.y1);
  float xx2 = std::min(a.x2,b.x2), yy2 = std::min(a.y2,b.y2);
  float inter = std::max(0.f,xx2-xx1) * std::max(0.f,yy2-yy1);
  float da = std::max(0.f,a.x2-a.x1)*std::max(0.f,a.y2-a.y1);
  float db = std::max(0.f,b.x2-b.x1)*std::max(0.f,b.y2-b.y1);
  float den = da + db - inter;
  return den > 0.f ? inter/den : 0.f;
}

static std::vector<objdet::Box> decode_yolov11_postsurgery(
    const simaai::neat::TensorList& tensors, float min_score) {
  if (tensors.size() < 6)
    throw std::runtime_error("expected 6 tensors, got " + std::to_string(tensors.size()));
  const float inv_gain = 1.0f / kLbGain;
  std::vector<objdet::Box> candidates;
  candidates.reserve(4000);
  for (size_t level = 0; level < 3; ++level) {
    const TensorHWC bbox = tensor_to_hwc_f32(tensors[level]);
    const TensorHWC cls  = tensor_to_hwc_f32(tensors[level + 3]);
    const float* cls_ptr = cls.floats(), *bbox_ptr = bbox.floats();
    for (int y = 0; y < cls.h; ++y) {
      for (int x = 0; x < cls.w; ++x, cls_ptr += cls.c, bbox_ptr += 4) {
        int best_class = -1; float best_score = 0.f;
#ifdef __ARM_NEON
        if (cls.c == 80) { argmax80_neon(cls_ptr, best_score, best_class); } else
#endif
        { for (int c = 0; c < cls.c; ++c) if (cls_ptr[c] > best_score) { best_score = cls_ptr[c]; best_class = c; } }
        if (best_score < min_score || best_class < 0) [[likely]] continue;
        float cx = bbox_ptr[0], cy = bbox_ptr[1], bw = bbox_ptr[2], bh = bbox_ptr[3];
        float x1 = std::clamp((cx-bw*.5f-kLbPadLeft)*inv_gain, 0.f, (float)kInputW);
        float y1 = std::clamp((cy-bh*.5f-kLbPadTop )*inv_gain, 0.f, (float)kInputH);
        float x2 = std::clamp((cx+bw*.5f-kLbPadLeft)*inv_gain, 0.f, (float)kInputW);
        float y2 = std::clamp((cy+bh*.5f-kLbPadTop )*inv_gain, 0.f, (float)kInputH);
        if (x2 > x1 && y2 > y1) [[unlikely]]
          candidates.emplace_back(x1, y1, x2, y2, best_score, best_class);
      }
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const objdet::Box& a, const objdet::Box& b){ return a.score > b.score; });
  std::vector<objdet::Box> keep;
  keep.reserve(kMaxDet);
  for (const auto& box : candidates) {
    bool sup = false;
    for (const auto& k : keep)
      if (k.class_id == box.class_id && iou_xyxy(k, box) > kNmsIou) { sup = true; break; }
    if (!sup) { keep.push_back(box); if ((int)keep.size() >= kMaxDet) [[unlikely]] break; }
  }
  return keep;
}

static std::vector<objdet::Box> scale_boxes(const std::vector<objdet::Box>& boxes,
                                            int orig_w, int orig_h) {
  float sx = (float)orig_w / kInputW, sy = (float)orig_h / kInputH;
  std::vector<objdet::Box> out; out.reserve(boxes.size());
  for (const auto& b : boxes)
    out.emplace_back(b.x1*sx, b.y1*sy, b.x2*sx, b.y2*sy, b.score, b.class_id);
  return out;
}

static const std::array<cv::Scalar,8> kColors = {{
  {0,255,0},{255,0,0},{0,0,255},{255,255,0},{255,0,255},{0,255,255},{128,255,0},{255,128,0}
}};

static void draw_boxes(cv::Mat& frame, const std::vector<objdet::Box>& boxes,
                       const std::vector<std::string>& labels) {
  // NV12 mat is CV_8UC1 with rows=H*3/2; draw only on the Y (luma) plane.
  const bool is_nv12 = (frame.channels() == 1);
  cv::Mat canvas = is_nv12 ? frame.rowRange(0, frame.rows * 2 / 3) : frame;
  for (const auto& b : boxes) {
    int x1 = std::clamp((int)std::round(b.x1), 0, canvas.cols-1);
    int y1 = std::clamp((int)std::round(b.y1), 0, canvas.rows-1);
    int x2 = std::clamp((int)std::round(b.x2), 0, canvas.cols-1);
    int y2 = std::clamp((int)std::round(b.y2), 0, canvas.rows-1);
    if (x2 <= x1 || y2 <= y1) continue;
    // NV12: bright luma (220) for boxes; BGR: per-class color
    const cv::Scalar col = is_nv12
        ? cv::Scalar(220)
        : kColors[(size_t)std::abs(b.class_id) % kColors.size()];
    const std::string lbl = (b.class_id >= 0 && (size_t)b.class_id < labels.size()
                              ? labels[b.class_id] : std::to_string(b.class_id))
                          + " " + cv::format("%.2f", b.score);
    cv::rectangle(canvas, {x1,y1}, {x2,y2}, col, 2);
    int bl = 0;
    cv::Size tsz = cv::getTextSize(lbl, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &bl);
    int yt = std::max(0, y1 - tsz.height - 4);
    cv::rectangle(canvas, {x1,yt}, {x1+tsz.width, std::max(0,y1)},
                  is_nv12 ? cv::Scalar(40) : col, cv::FILLED);
    cv::putText(canvas, lbl, {x1, std::max(0,y1-2)},
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                is_nv12 ? cv::Scalar(220) : cv::Scalar(0,0,0), 1, cv::LINE_AA);
  }
}

static std::vector<std::string> load_labels(const fs::path& p) {
  std::ifstream in(p);
  if (!in.good()) throw std::runtime_error("labels not found: " + p.string());
  std::vector<std::string> v;
  std::string line;
  while (std::getline(in, line)) if (!line.empty()) v.push_back(line);
  if (v.empty()) throw std::runtime_error("labels file empty: " + p.string());
  return v;
}

// ── Frame save (--save) ───────────────────────────────────────────────────────
// Saves NV12 mat as JPEG every 100 frames when --save is active.
// Called outside timing blocks so save I/O does not affect latency stats.
static void save_frame(bool use_save, const cv::Mat& nv12, const char* stage, int idx) {
  if (!use_save || idx == 0 || idx % 100 != 0) return;
  cv::Mat bgr;
  cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
  char path[128];
  std::snprintf(path, sizeof(path), "/tmp/save_%s_%06d.jpg", stage, idx);
  cv::imwrite(path, bgr, {cv::IMWRITE_JPEG_QUALITY, 90});
  std::cout << "[save] " << path << "\n";
}

// ── GStreamer NV12 encode pipeline ────────────────────────────────────────────
// appsrc receives NV12 frames directly → neatencoder HW H.264 (no videoconvert).
// enc-ip-mode=async + ip-rate-ctrl=false: non-blocking HW submit, ~5ms vs 22ms sync.
static std::string build_hw_encode_gst(const std::string& host, int port,
                                       int w, int h, int fps) {
  std::ostringstream s;
  s << "appsrc name=enc_src"
    << " caps=video/x-raw,format=NV12,width=" << w << ",height=" << h
    << ",framerate=" << fps << "/1"
    << " is-live=true do-timestamp=false stream-type=0 format=time"
    << " ! neatencoder enc-type=h264 enc-fmt=NV12"
    << " enc-width=" << w << " enc-height=" << h
    << " enc-bitrate=3000 enc-frame-rate=" << fps
    << " enc-level=4.2 enc-profile=main"
    << " ! h264parse"
    << " ! rtph264pay config-interval=1 pt=96"
    << " ! queue max-size-time=200000000 max-size-buffers=0 max-size-bytes=0"
    << " ! udpsink host=" << host << " port=" << port << " sync=false async=false";
  return s.str();
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  ::setenv("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY", "1", 0);
  // Ensure SiMa HW plugins (neatdecoder, neatencoder, FPSSINK) are on the
  // GStreamer plugin search path so gst_parse_launch finds them.  The SDK's
  // gst_init_once() normally sets this, but we set it upfront so it is
  // already present when the SDK calls gst_init() during model.build().
  // Do NOT call gst_init() ourselves — letting the SDK own the first call
  // ensures neatdecoder's IPC channel initialises correctly.
  ::setenv("GST_PLUGIN_PATH_1_0",
           "/usr/lib/aarch64-linux-gnu/neat/gst-plugins",
           0);  // 0 = don't override if already set by caller

  // ── Argument parsing ──────────────────────────────────────────────────────
  std::string rtsp_url, insight_host, classes_arg;
  int  rtsp_frames = 99999, video_port = 9000, meta_port = 9100;
  bool use_zerocopy = false;
  bool use_save     = false;
  std::vector<std::string> pos_args;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if      (a == "--rtsp"       && i+1 < argc) rtsp_url     = argv[++i];
    else if (a == "--frames"     && i+1 < argc) rtsp_frames  = std::stoi(argv[++i]);
    else if (a == "--insight"    && i+1 < argc) insight_host = argv[++i];
    else if (a == "--video-port" && i+1 < argc) video_port   = std::stoi(argv[++i]);
    else if (a == "--meta-port"  && i+1 < argc) meta_port    = std::stoi(argv[++i]);
    else if (a == "--classes"    && i+1 < argc) classes_arg  = argv[++i];
    else if (a == "--zerocopy")                 use_zerocopy = true;
    else if (a == "--save")                     use_save     = true;
    else pos_args.push_back(a);
  }
  if (pos_args.size() < 2 || rtsp_url.empty()) {
    std::cerr << "Usage: " << argv[0]
              << " <model> <labels> --rtsp <URL>\n"
              << "  [--insight <host>] [--frames N] [--classes <csv>]\n"
              << "  [--video-port N] [--meta-port N] [--zerocopy] [--save]\n";
    return 1;
  }
  const std::string model_path  = pos_args[0];
  const fs::path    labels_path = pos_args[1];

  try {
    auto labels = load_labels(labels_path);

    // ── Class whitelist ───────────────────────────────────────────────────────
    std::unordered_set<int> allowed_classes;
    if (!classes_arg.empty()) {
      std::istringstream ss(classes_arg);
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        for (size_t i = 0; i < labels.size(); ++i)
          if (labels[i] == tok) { allowed_classes.insert((int)i); break; }
      }
      std::cout << "[filter] class whitelist (" << allowed_classes.size()
                << "): " << classes_arg << "\n";
    }

    // ── Model setup ───────────────────────────────────────────────────────────
    simaai::neat::Model::Options model_opt;
    model_opt.preprocess.kind   = simaai::neat::InputKind::Image;
    model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
    model_opt.preprocess.color_convert.input_format =
        simaai::neat::PreprocessColorFormat::NV12;
    model_opt.preprocess.preset         = simaai::neat::NormalizePreset::COCO_YOLO;
    model_opt.preprocess.input_max_width  = kInputW;
    model_opt.preprocess.input_max_height = kInputH;
    model_opt.score_threshold   = kMinScore;
    model_opt.nms_iou_threshold = kNmsIou;
    model_opt.top_k             = kMaxDet;

    simaai::neat::Model model(model_path, model_opt);

    // ── Async Runner: push/pull in separate threads ───────────────────────────
    //
    // RunPreset::Realtime: small queues, KeepLatest overflow, low-latency defaults.
    // queue_depth=3: up to 3 frames in-flight in the MLA pipeline simultaneously.
    // OverflowPolicy::Block: Thread A blocks when all 3 slots are full, preventing
    //   push_rec_q / runner queue alignment drift (safer than KeepLatest for sync).
    //
    // SDK contract: "push() from one thread and pull() from another is safe."
    // Multiple concurrent pushers are NOT safe — Thread A is the sole pusher.
    //
    simaai::neat::RunOptions run_opt;
    run_opt.preset           = simaai::neat::RunPreset::Realtime;
    run_opt.queue_depth      = 5;
    run_opt.overflow_policy  = simaai::neat::OverflowPolicy::Block;
    if (use_zerocopy) {
      // ZeroCopy: skip DMA copy of output tensors; SDK only does cache-visibility
      // maintenance on the MLA-backed GStreamer buffer before returning it to us.
      // prepare_output_cpu_visible=true makes tensors readable from CPU without
      // calling Tensor::map() — required for stages::Tensors() to work correctly.
      run_opt.output_memory                        = simaai::neat::OutputMemory::ZeroCopy;
      run_opt.advanced.prepare_output_cpu_visible  = true;
    } else {
      // Owned: SDK allocates a CPU buffer and DMA-copies each output frame into it.
      // Safer lifetime; avg_copy_us in MeasureReport captures this overhead.
      run_opt.output_memory = simaai::neat::OutputMemory::Owned;
    }

    auto runner = [&]() {
      for (int attempt = 1; ; ++attempt) {
        try { return model.build(run_opt); }
        catch (const std::exception& e) {
          if (attempt >= 3) throw;
          std::cerr << "[BUILD] attempt " << attempt << "/3: " << e.what()
                    << " — retrying in 3s\n";
          std::this_thread::sleep_for(std::chrono::seconds(3));
        }
      }
    }();
    std::cout << "[BUILD] " << model_path << " ready — "
              << kInputW << "x" << kInputH
              << " NV12 | async push/pull | queue_depth=" << run_opt.queue_depth
              << " | output=" << (use_zerocopy ? "ZeroCopy" : "Owned") << "\n";

    // ── SDK MeasureScope (plugin latency breakdown) ───────────────────────────
    // Wraps the entire push/pull workload. Reports avg_push_us / avg_pull_wait_us /
    // avg_copy_us / avg_alloc_us after threads join, revealing where the 8-10ms
    // overhead beyond MLA hw time (~6ms) actually goes.
    simaai::neat::MeasureOptions mopt;
    mopt.warmup_ms             = 2000;  // ~130 frames warm-up at 65fps
    mopt.include_plugin_latency = true; // per-node (EV74+MLA) breakdown
    auto mscope = runner.start_measurement(mopt);

    // ── neatdecoder HW H.264 decode via GStreamer appsink (direct API) ──────────
    // Opened AFTER model.build() so the SDK has called gst_init() first, which
    // ensures neatdecoder's VCU IPC channel initialises correctly under the SDK's
    // GStreamer environment.  We do NOT call gst_init() ourselves; the SDK owns
    // the first call.
    //
    // rtspsrc creates its GstRTSPWatch (RTSP PLAY response handler) from an
    // internal GstTask thread whose thread-default context is the global (NULL)
    // context.  A running g_main_loop on that context is required to dispatch the
    // callback.  Keep the loop alive for the decoder's entire lifetime — it also
    // serves RTSP keepalive responses.
    GMainLoop* rtsp_gloop    = g_main_loop_new(nullptr, FALSE);
    std::thread rtsp_gloop_th([rtsp_gloop]() { g_main_loop_run(rtsp_gloop); });

    // neatdecoder outputs video/x-raw(memory:SimaAI),format=NV12.
    // gate-mode=strict gate-timeout-ms=0: wait for IDR frame before pushing
    // any data to the VCU; without this the VCU receives non-IDR frames first
    // (relaxed mode opens after 1 s timeout even without IDR) and fails with
    // "No IPC response" because the HW codec cannot decode without a reference.
    // config-interval=-1 on h264parse inlines SPS/PPS before every IDR so the
    // VCU always has codec params when the gate opens.
    // videoconvert converts SimaAI→SystemMemory so gst_buffer_map works.
    const std::string gst_dec_pipe =
        "rtspsrc location=" + rtsp_url + " latency=200 protocols=tcp "
        "! rtph264depay ! h264parse config-interval=-1 "
        "! neatdecoder sima-allocator-type=2 dec-fmt=NV12 "
          "gate-mode=strict gate-timeout-ms=0 "
        "! videoconvert "
        "! video/x-raw(memory:SystemMemory),format=NV12 "
        "! appsink name=dec_sink sync=false drop=true max-buffers=4 emit-signals=false";
    std::cout << "[DEC] opening RTSP: " << rtsp_url << "\n";

    GError*     gst_err      = nullptr;
    GstElement* dec_pipeline = gst_parse_launch(gst_dec_pipe.c_str(), &gst_err);

    if (!dec_pipeline || gst_err) {
      std::cerr << "[DEC] failed to create pipeline: "
                << (gst_err ? gst_err->message : "unknown error") << "\n";
      if (gst_err) g_error_free(gst_err);
      g_main_loop_quit(rtsp_gloop);
      rtsp_gloop_th.join();
      g_main_loop_unref(rtsp_gloop);
      return 5;
    }
    GstElement* dec_appsink = gst_bin_get_by_name(GST_BIN(dec_pipeline), "dec_sink");
    GstBus*     dec_bus     = gst_element_get_bus(dec_pipeline);

    gst_element_set_state(dec_pipeline, GST_STATE_PLAYING);

    GstState             dec_state = GST_STATE_NULL;
    GstStateChangeReturn dec_sc    =
        gst_element_get_state(dec_pipeline, &dec_state, nullptr, 30 * GST_SECOND);
    const bool dec_ok = (dec_state == GST_STATE_PLAYING);

    if (!dec_ok) {
      for (GstMessageType mt : {GST_MESSAGE_ERROR, GST_MESSAGE_WARNING}) {
        GstMessage* m = gst_bus_timed_pop_filtered(dec_bus, 0, mt);
        if (!m) continue;
        GError* e = nullptr; gchar* dbg = nullptr;
        if (mt == GST_MESSAGE_ERROR)
          gst_message_parse_error(m, &e, &dbg);
        else
          gst_message_parse_warning(m, &e, &dbg);
        std::cerr << "[DEC] " << (mt == GST_MESSAGE_ERROR ? "error" : "warn")
                  << ": " << (e ? e->message : "?")
                  << (dbg ? std::string(" | ") + dbg : "") << "\n";
        if (e) g_error_free(e);
        g_free(dbg);
        gst_message_unref(m);
      }
      std::cerr << "[DEC] pipeline failed: state="
                << gst_element_state_get_name(dec_state) << "\n";
      gst_object_unref(dec_bus);
      gst_element_set_state(dec_pipeline, GST_STATE_NULL);
      gst_object_unref(dec_appsink);
      gst_object_unref(dec_pipeline);
      g_main_loop_quit(rtsp_gloop);
      rtsp_gloop_th.join();
      g_main_loop_unref(rtsp_gloop);
      return 5;
    }
    gst_object_unref(dec_bus);
    const int encode_fps = 30;
    std::cout << "[DEC] opened " << kInputW << "x" << kInputH
              << " NV12 via neatdecoder (encode fps=" << encode_fps << ")\n";

    // ── HW encode: neatencoder via GStreamer appsrc (NV12 direct push) ──────────
    // Use GStreamer API directly so NV12 frames are pushed without BGR conversion.
    GstElement* enc_pipeline = nullptr;
    GstElement* enc_appsrc   = nullptr;
    if (!insight_host.empty()) {
      const std::string encode_gst = build_hw_encode_gst(
          insight_host, video_port, kInputW, kInputH, encode_fps);
      std::cout << "[HW-ENC] pipeline:\n  " << encode_gst << "\n";
      GError* enc_err  = nullptr;
      enc_pipeline = gst_parse_launch(encode_gst.c_str(), &enc_err);
      if (!enc_pipeline || enc_err) {
        std::cerr << "[HW-ENC] warn: failed to create encoder: "
                  << (enc_err ? enc_err->message : "unknown") << " — video disabled\n";
        if (enc_err) g_error_free(enc_err);
        enc_pipeline = nullptr;
      } else {
        enc_appsrc = gst_bin_get_by_name(GST_BIN(enc_pipeline), "enc_src");
        if (!enc_appsrc)
          std::cerr << "[HW-ENC] ERROR: appsrc element 'enc_src' not found in pipeline!\n";
        else
          std::cout << "[HW-ENC] appsrc element found\n";
        GstStateChangeReturn sc = gst_element_set_state(enc_pipeline, GST_STATE_PLAYING);
        GstState cur_state = GST_STATE_NULL, pend_state = GST_STATE_NULL;
        gst_element_get_state(enc_pipeline, &cur_state, &pend_state,
                              5 * GST_SECOND);  // wait up to 5s
        std::cout << "[HW-ENC] state after wait: " << gst_element_state_get_name(cur_state)
                  << " (pending=" << gst_element_state_get_name(pend_state)
                  << " sc=" << sc << ")\n";
        // Read any error messages off the bus
        {
          GstBus* enc_bus = gst_pipeline_get_bus(GST_PIPELINE(enc_pipeline));
          GstMessage* bmsg;
          while ((bmsg = gst_bus_pop(enc_bus)) != nullptr) {
            if (GST_MESSAGE_TYPE(bmsg) == GST_MESSAGE_ERROR) {
              GError* berr = nullptr; gchar* bdbg = nullptr;
              gst_message_parse_error(bmsg, &berr, &bdbg);
              std::cerr << "[HW-ENC] bus ERROR: " << berr->message
                        << " | debug: " << (bdbg ? bdbg : "none") << "\n";
              g_free(bdbg); g_error_free(berr);
            } else if (GST_MESSAGE_TYPE(bmsg) == GST_MESSAGE_WARNING) {
              GError* berr = nullptr; gchar* bdbg = nullptr;
              gst_message_parse_warning(bmsg, &berr, &bdbg);
              std::cerr << "[HW-ENC] bus WARN: " << berr->message
                        << " | debug: " << (bdbg ? bdbg : "none") << "\n";
              g_free(bdbg); g_error_free(berr);
            }
            gst_message_unref(bmsg);
          }
          gst_object_unref(enc_bus);
        }
        std::cout << "[insight] video  -> udp://" << insight_host
                  << ":" << video_port << "\n";
      }
    }

    // ── MetadataSender ────────────────────────────────────────────────────────
    std::unique_ptr<simaai::neat::MetadataSender> meta_sender;
    if (!insight_host.empty()) {
      simaai::neat::MetadataSenderOptions meta_opt;
      meta_opt.host               = insight_host;
      meta_opt.channel            = 0;
      meta_opt.metadata_port_base = meta_port;
      std::string meta_err;
      meta_sender = std::make_unique<simaai::neat::MetadataSender>(meta_opt, &meta_err);
      if (meta_sender->ok())
        std::cout << "[insight] metadata -> udp://" << insight_host
                  << ":" << meta_sender->metadata_port() << "\n";
      else {
        std::cerr << "[insight] warn: MetadataSender: " << meta_err << "\n";
        meta_sender.reset();
      }
    }

    // ── 6-thread async pipeline ───────────────────────────────────────────────
    //
    //  Thread A0 [FFmpeg cap.read()                     ] → decode_q(8)
    //  Thread A1 [preproc + runner.push()               ] → push_rec_q(4)
    //  Thread B  [runner.pull() → tensors               ] → fast_post_q(4)
    //  Thread C  [post-process + draw                   ] → fast_encode_q(16)
    //  Thread D  [dispatch, non-blocking                ] → hw_write_q(1)
    //  Thread E  [neatencoder write ~20ms               ]
    //
    // Splitting decode (A0) from push (A1) eliminates the ~5ms pop_wait that
    // Thread B suffered when Thread A combined decode+push in one cycle: with
    // decode_q(8) always pre-filled, Thread A1 pops instantly and runner.push()
    // is ready immediately, keeping Thread B's pull() continuously fed.
    //
    // Thread A1 and Thread B share the runner: A1 calls push(), B calls pull().
    // push_rec_q FIFO-alignment is maintained: A1 enqueues AFTER runner.push()
    // with OverflowPolicy::Block ensuring no silent drops inside the runner.
    //
    PipelineQueue<DecodeRecord> decode_q(8);      // A0 → A1: raw frames
    PipelineQueue<PushRecord>   push_rec_q(4);    // A1 → B: display + timing
    PipelineQueue<PostIn>       fast_post_q(4);   // B → C: tensors + display
    using EncFrame = std::pair<cv::Mat, int>;     // {frame, original frame idx}
    PipelineQueue<EncFrame>     fast_encode_q(16);// C → D
    PipelineQueue<EncFrame>     hw_write_q(1);    // D → E: single slot, drops if E busy

    std::atomic<int>  frames_done{0};
    std::atomic<int>  frames_dropped{0};
    Clock::time_point t_first_out{};
    std::atomic<bool> first_out_seen{false};

    // stage_acc: [0]=read     [1]=preproc  [2]=push    [3]=pull   [4]=post
    //            [5]=encode   [6]=q_wait   [7]=inflight [8]=tensors
    //            [9]=dec_wait (Thread A1 blocked on decode_q — should be ~0)
    std::array<TimingAcc, 10> stage_acc;

    // ── Thread A0: neatdecoder HW decode ─────────────────────────────────────
    // Passes GstSample* (ref-counted) through decode_q without any memcpy.
    // Thread A1 maps the buffer, copies into the storage pool, then unrefs.
    const size_t kNV12Bytes = static_cast<size_t>(kInputW) * kInputH * 3 / 2;

    // Pre-allocate NV12 storage pool (queue_depth + 2 slots) so Thread A1 never
    // calls make_cpu_owned_storage() in the hot path (saves ~1.25 ms/frame).
    // The SDK copies our buffer into EV74 memory during push() (async DMA, ~683 µs).
    // pull() completes only after the DMA is done, so when push(frame[N]) is
    // allowed to proceed (Block policy, queue_depth=5), pull(frame[N-5]) has
    // returned and its slot is safe to reuse.  7 slots = queue_depth + 2 gives
    // adequate headroom.
    constexpr int kPoolSize = 7;
    std::array<std::shared_ptr<simaai::neat::TensorBuffer>, kPoolSize> nv12_pool;
    for (auto& s : nv12_pool) s = simaai::neat::make_cpu_owned_storage(kNV12Bytes);
    int pool_idx = 0;

    std::thread decode_th([&]() {
      for (;;) {
        if (decode_q.is_closed()) break;
        auto t0 = Clock::now();
        GstSample* sample = gst_app_sink_try_pull_sample(
            GST_APP_SINK(dec_appsink), 2 * GST_SECOND);
        if (!sample) {
          if (gst_app_sink_is_eos(GST_APP_SINK(dec_appsink)))
            std::cerr << "[DEC] stream ended (EOS)\n";
          else
            std::cerr << "[DEC] pull timeout\n";
          break;
        }
        // No memcpy here — just pass the ref to Thread A1.
        double read_ms = Ms(Clock::now() - t0).count();
        decode_q.push({sample, read_ms});
      }
      decode_q.close();
    });

    // ── Thread A1: preproc + runner.push() ───────────────────────────────────
    // Pops from decode_q (near-instant, always pre-filled), does format
    // conversion and resize, then pushes to runner and push_rec_q.
    // Sole thread calling runner.push() — SDK contract satisfied.
    std::thread push_th([&]() {
      for (int fi = 0; fi < rtsp_frames; ++fi) {
        // Pop decode frame — should return instantly since Thread A0 pre-fills.
        auto   t_pop = Clock::now();
        DecodeRecord dr;
        if (!decode_q.pop(dr)) { std::cerr << "[A1] decode_q closed\n"; break; }
        double dec_wait_ms = Ms(Clock::now() - t_pop).count();
        if (fi > 0) stage_acc[0].add(dr.read_ms);
        if (fi > 0) stage_acc[9].add(dec_wait_ms);

        auto t1 = Clock::now();
        // Map the GstBuffer once; copy into pool slot (no intermediate cv::Mat).
        // Also copy into display (Thread C draws boxes on Y plane of display).
        // Two copies from the same source: sdk_storage + display, saving one
        // copy vs the old A0(→cv::Mat) + A1(cv::Mat→sdk + cv::Mat.clone()) path.
        auto& sdk_storage = nv12_pool[pool_idx++ % kPoolSize];
        cv::Mat display;
        {
          GstBuffer* buf = gst_sample_get_buffer(dr.sample);
          GstMapInfo map;
          gst_buffer_map(buf, &map, GST_MAP_READ);
          const size_t copy_bytes = std::min(map.size, kNV12Bytes);
          display.create(kInputH * 3 / 2, kInputW, CV_8UC1);
          // Read the GstBuffer ONCE, write into both sdk_storage and display.
          // NEON tee-copy cuts memory traffic by ~25% vs two separate memcpy calls
          // (3 passes: 1 read + 2 writes  vs  4 passes: 2 reads + 2 writes).
#ifdef __ARM_NEON
          neon_tee_copy(sdk_storage->data, display.data, map.data, copy_bytes);
#else
          std::memcpy(sdk_storage->data, map.data, copy_bytes);
          std::memcpy(display.data,      map.data, copy_bytes);
#endif
          gst_buffer_unmap(buf, &map);
        }
        gst_sample_unref(dr.sample);       // A1 owns the ref, release now
        save_frame(use_save, display, "dec", fi);
        simaai::neat::Tensor nv12_t = make_nv12_tensor(sdk_storage, kInputW, kInputH);
        double  preproc_ms = Ms(Clock::now() - t1).count();
        if (fi > 0) stage_acc[1].add(preproc_ms);

        // push() non-blocking while runner queue has space (queue_depth=5),
        // blocks only when all slots are full (Thread B must pull first).
        auto t2 = Clock::now();
        if (!runner.push(simaai::neat::TensorList{nv12_t})) {
          std::cerr << "[PUSH] runner.push() rejected frame " << fi << "\n";
          break;
        }
        double            push_ms   = Ms(Clock::now() - t2).count();
        Clock::time_point push_time = Clock::now();
        if (fi > 0) stage_acc[2].add(push_ms);

        // Enqueue AFTER push — maintains FIFO alignment with runner's queue.
        double q_wait_ms = 0.0;
        push_rec_q.push({std::move(display), cv::Mat{},
                         dr.read_ms, preproc_ms, push_time, fi},
                        &q_wait_ms);
        if (fi > 0) stage_acc[6].add(q_wait_ms);
      }
      decode_q.close();    // signal Thread A0 to stop
      runner.close_input();
      push_rec_q.close();
    });

    // ── Thread B: runner.pull() → fast_post_q ─────────────────────────────────
    // Sole thread calling runner.pull(). Pops the matching display frame from
    // push_rec_q before pulling so that display + tensors are always paired.
    // SDK guarantee: pull() from one thread while push() runs in another is safe.
    std::thread pull_th([&]() {
      PushRecord pr;
      while (push_rec_q.pop(pr)) {
        auto t0 = Clock::now();
        simaai::neat::Sample sample = runner.pull(kTimeoutMs);
        double pull_ms = Ms(Clock::now() - t0).count();

        if (sample.empty()) {
          std::cerr << "[PULL] runner.pull() returned empty sample (timeout/closed)\n";
          break;
        }
        // inflight = push_time → now: time this frame spent inside the runner.
        // Ideally ≈ pull_ms (MLA+EV74 latency ≈ 14.5ms).
        // If inflight >> pull_ms, frames are queueing up inside the runner
        // (Thread A is outrunning Thread B).
        double inflight_ms = Ms(Clock::now() - pr.push_time).count();
        if (pr.idx > 0) stage_acc[3].add(pull_ms);
        if (pr.idx > 0) stage_acc[7].add(inflight_ms);

        // Extract TensorList from Sample using the SDK helper.
        // model.build().pull() may return a Bundle (kind==Bundle, fields contains
        // the per-output samples) rather than a plain TensorSet. stages::Tensors()
        // recursively collects tensors from all sample kinds (TensorSet, Tensor,
        // Bundle), giving the same TensorList that runner.run() returns internally.
        auto t_tensors = Clock::now();
        simaai::neat::TensorList tensors = simaai::neat::stages::Tensors(sample);
        double tensors_ms = Ms(Clock::now() - t_tensors).count();
        if (pr.idx > 0) stage_acc[8].add(tensors_ms);

        fast_post_q.push({std::move(pr.display), std::move(tensors),
                          pr.read_ms, pr.preproc_ms,
                          stage_acc[2].n > 0 ? stage_acc[2].mean() : 0.0,
                          pull_ms, pr.idx});
      }
      fast_post_q.close();
    });

    // ── Thread C: post-process + draw ─────────────────────────────────────────
    std::thread post_th([&]() {
      PostIn pi;
      while (fast_post_q.pop(pi)) {
        if (!first_out_seen.exchange(true)) t_first_out = Clock::now();

        auto t0 = Clock::now();
        std::vector<objdet::Box> boxes;
        if (!pi.tensors.empty()) {
          try {
            auto boxes_720 = decode_yolov11_postsurgery(pi.tensors, kMinScore);
            // NV12 mat has rows=H*3/2; pass the Y-plane height (rows*2/3) for scaling.
            auto all = scale_boxes(boxes_720,
                                   pi.display.cols,
                                   pi.display.rows * 2 / 3);
            if (allowed_classes.empty()) {
              boxes = std::move(all);
            } else {
              for (auto& b : all)
                if (allowed_classes.count(b.class_id)) boxes.push_back(b);
            }
          } catch (const std::exception& e) {
            std::cerr << "post failed frame " << pi.idx << ": " << e.what() << "\n";
          }
        }
        double post_ms = Ms(Clock::now() - t0).count();
        if (pi.idx > 0) stage_acc[4].add(post_ms);

        // pi.display is NV12 (CV_8UC1, rows=H*3/2); draw directly on Y plane —
        // no BGR conversion. Boxes appear as bright luma overlays on the viewer.
        draw_boxes(pi.display, boxes, labels);
        save_frame(use_save, pi.display, "box", pi.idx);

        if (enc_appsrc)
          fast_encode_q.push({std::move(pi.display), pi.idx});

        int done = ++frames_done;
        std::cout << "[" << done << "/" << rtsp_frames << "]"
                  << cv::format(" frame_%05d | read=%.1f push=%.1f pull=%.1f post=%.1f",
                                pi.idx, pi.read_ms, pi.push_ms, pi.pull_ms, post_ms);
        for (const auto& b : boxes)
          std::cout << " [" << (b.class_id >= 0 && (size_t)b.class_id < labels.size()
                                 ? labels[b.class_id] : "?")
                    << cv::format(" %.2f]", b.score);
        std::cout << "\n";
      }
      fast_encode_q.close();
    });

    // ── Thread D: non-blocking encode dispatcher ──────────────────────────────
    // Pops rendered frames and try_pushes to hw_write_q(1). Returns immediately
    // whether or not the push succeeds — never calls write() directly.
    std::thread dispatch_th([&]() {
      EncFrame ef;
      while (fast_encode_q.pop(ef)) {
        if (enc_appsrc) {
          if (!hw_write_q.try_push(std::move(ef)))
            ++frames_dropped;
        }
      }
      hw_write_q.close();
    });

    // ── Thread E: neatencoder write ────────────────────────────────────────────
    // Pushes NV12 GstBuffers directly to appsrc — neatencoder encodes to H.264.
    // Monotonic PTS avoids 122ms jitter that occurs when appsrc timestamps buffers
    // with wall-clock time (variable due to Thread D try_push drops and pipeline
    // backpressure). Without explicit PTS the WebRTC jitter buffer discards most
    // frames, producing a black screen despite 695 kbps arriving at the browser.
    std::thread encode_th([&]() {
      EncFrame ef;
      const GstClockTime enc_dur =
          gst_util_uint64_scale_int(GST_SECOND, 1, encode_fps);
      GstClockTime enc_pts = enc_dur;  // fallback if no clock yet
      if (enc_appsrc) {
        GstClock* clk = gst_element_get_clock(GST_ELEMENT(enc_appsrc));
        if (clk) {
          GstClockTime base = gst_element_get_base_time(GST_ELEMENT(enc_appsrc));
          GstClockTime now  = gst_clock_get_time(clk);
          if (now > base) enc_pts = (now - base) + enc_dur;
          gst_object_unref(clk);
        }
      }
      while (hw_write_q.pop(ef)) {
        cv::Mat& frame = ef.first;
        const int frame_idx = ef.second;
        save_frame(use_save, frame, "enc", frame_idx);  // outside timing, same idx as dec/box
        auto t0 = Clock::now();
        if (enc_appsrc && !frame.empty()) {
          const gsize nbytes = static_cast<gsize>(frame.total());
          GstBuffer* buf = gst_buffer_new_allocate(nullptr, nbytes, nullptr);
          GST_BUFFER_PTS(buf)      = enc_pts;
          GST_BUFFER_DTS(buf)      = enc_pts;
          GST_BUFFER_DURATION(buf) = enc_dur;
          enc_pts += enc_dur;
          // neatencoder's GstDataAdapter requires GstVideoMeta on every buffer
          // to know the NV12 plane offsets/strides.  Without it the encoder
          // silently drops every frame ("video meta: missing GstVideoMeta").
          gsize nv12_offsets[2] = {0, static_cast<gsize>(kInputW) * kInputH};
          gint  nv12_strides[2] = {kInputW, kInputW};
          gst_buffer_add_video_meta_full(buf, GST_VIDEO_FRAME_FLAG_NONE,
                                        GST_VIDEO_FORMAT_NV12,
                                        kInputW, kInputH,
                                        2, nv12_offsets, nv12_strides);
          GstMapInfo  map;
          gst_buffer_map(buf, &map, GST_MAP_WRITE);
          std::memcpy(map.data, frame.data, nbytes);
          gst_buffer_unmap(buf, &map);
          GstFlowReturn flow = gst_app_src_push_buffer(GST_APP_SRC(enc_appsrc), buf);
          if (flow != GST_FLOW_OK && frame_idx < 5)
            std::cerr << "[HW-ENC] push_buffer flow=" << flow << "\n";
        }
        stage_acc[5].add(Ms(Clock::now() - t0).count());
      }
      if (enc_appsrc)
        gst_app_src_end_of_stream(GST_APP_SRC(enc_appsrc));
    });

    push_th.join();
    decode_th.join();
    pull_th.join();
    post_th.join();
    dispatch_th.join();
    encode_th.join();

    // ── SDK internal breakdown ────────────────────────────────────────────────
    auto report = mscope.stop();
    std::cout << "\n── SDK MeasureReport ─────────────────────────────────────\n"
              << report.to_text() << "\n";
    // Highlight the input-boundary timings that account for pull() overhead:
    //   avg_push_us      — IPC overhead sending each frame to the MLA runtime
    //   avg_pull_wait_us — time Thread B blocks waiting for MLA output to be ready
    //   avg_copy_us      — DMA copy of output tensors (Owned mode only; 0 for ZeroCopy)
    //   avg_alloc_us     — output buffer allocation per frame
    std::cout << cv::format(
        "── SDK input stats ───────────────────────────────────────\n"
        "  output_memory  : %s\n"
        "  avg_push_us    : %.1f us\n"
        "  avg_pull_wait  : %.1f us\n"
        "  avg_copy_us    : %.1f us  (DMA copy, 0=ZeroCopy)\n"
        "  avg_alloc_us   : %.1f us\n"
        "  avg_map_us     : %.1f us\n",
        use_zerocopy ? "ZeroCopy+cpu_visible" : "Owned",
        report.input.avg_push_us,
        report.input.avg_pull_wait_us,
        report.input.avg_copy_us,
        report.input.avg_alloc_us,
        report.input.avg_map_us);

    const int processed = frames_done.load();
    const int dropped   = frames_dropped.load();
    if (processed > 1 && first_out_seen.load()) {
      double elapsed    = Ms(Clock::now() - t_first_out).count();
      double fps        = (processed - 1) * 1000.0 / elapsed;
      int    encoded    = stage_acc[5].n;
      double encode_fps = encoded > 0 ? encoded * 1000.0 / elapsed : 0.0;
      std::cout << cv::format("\n[pipeline] throughput: %.1f FPS  (%d frames, %.0f ms)\n",
                              fps, processed, elapsed);
      std::cout << cv::format("[pipeline] MLA ceil:   %.1f FPS  (1000/%.1fms pull)\n",
                              1000.0 / stage_acc[3].mean(), stage_acc[3].mean());
      std::cout << cv::format("[pipeline] viewer FPS: %.1f FPS  (%d encoded, %d dropped)\n",
                              encode_fps, encoded, dropped);

      const char* names[] = {"read", "preproc", "push", "pull", "post", "encode",
                             "q_wait", "inflight", "tensors", "dec_wait"};
      std::cout << "\n── Stage timings ─────────────────────────────────────────\n";
      std::cout << cv::format("%-10s  %8s  %8s  %8s  %8s\n",
                              "stage", "mean ms", "min ms", "max ms", "count");
      for (int i = 0; i < 10; ++i)
        if (stage_acc[i].n > 0)
          std::cout << cv::format("%-10s  %8.2f  %8.2f  %8.2f  %8d\n",
                                  names[i], stage_acc[i].mean(),
                                  stage_acc[i].mn, stage_acc[i].mx, stage_acc[i].n);

      // Queue high-water marks — detect backpressure between threads.
      std::cout << "\n── Queue high-water marks ────────────────────────────────\n";
      std::cout << cv::format("  decode_q    hwm=%zu / cap=%zu  (A0→A1: raw frames)\n",
                              decode_q.high_water(), decode_q.capacity());
      std::cout << cv::format("  push_rec_q  hwm=%zu / cap=%zu  (A1→B: runner record)\n",
                              push_rec_q.high_water(), push_rec_q.capacity());
      std::cout << cv::format("  fast_post_q hwm=%zu / cap=%zu  (B→C: tensors+display)\n",
                              fast_post_q.high_water(), fast_post_q.capacity());
      std::cout << cv::format("  fast_encode hwm=%zu / cap=%zu  (C→D: rendered frames)\n",
                              fast_encode_q.high_water(), fast_encode_q.capacity());
      std::cout << cv::format("  hw_write_q  hwm=%zu / cap=%zu  (D→E: encode slot)\n",
                              hw_write_q.high_water(), hw_write_q.capacity());
    }

    g_main_loop_quit(rtsp_gloop);
    rtsp_gloop_th.join();
    g_main_loop_unref(rtsp_gloop);
    gst_element_set_state(dec_pipeline, GST_STATE_NULL);
    gst_object_unref(dec_appsink);
    gst_object_unref(dec_pipeline);
    if (enc_pipeline) {
      gst_element_set_state(enc_pipeline, GST_STATE_NULL);
      if (enc_appsrc) gst_object_unref(enc_appsrc);
      gst_object_unref(enc_pipeline);
    }
    runner.close();
    std::cout << "Done: " << processed << " frames\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 4;
  }
}
