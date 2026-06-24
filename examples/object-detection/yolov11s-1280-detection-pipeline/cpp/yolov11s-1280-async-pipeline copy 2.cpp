/**
 * yolov11s-1280-async-pipeline.cpp
 *
 * RTSP detection pipeline using the SDK's async push/pull inference API.
 *
 *   decode : FFmpeg low-latency TCP  (rtspsrc/GStreamer hangs on this device)
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
 *   Thread A [FFmpeg-decode + preproc + runner.push()] → push_rec_q(4)
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
#include <opencv2/videoio.hpp>

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
  void push(T item) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_push_.wait(lk, [&]{ return q_.size() < cap_ || closed_; });
    if (closed_) return;
    q_.push(std::move(item));
    cv_pop_.notify_one();
  }
  // Non-blocking push: returns false immediately if queue is full or closed.
  bool try_push(T item) {
    std::lock_guard<std::mutex> lk(mu_);
    if (q_.size() >= cap_ || closed_) return false;
    q_.push(std::move(item));
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
private:
  std::queue<T> q_; std::mutex mu_;
  std::condition_variable cv_push_, cv_pop_;
  size_t cap_; bool closed_{false};
};

// ── Inter-stage data structs ──────────────────────────────────────────────────

// Thread A → Thread B: pushed frame record (display + timing context).
// Must stay FIFO-aligned with runner's internal push queue so that Thread B's
// pull() result matches the display frame popped from push_rec_q.
struct PushRecord {
  cv::Mat  display;      // rendered frame for boxes overlay
  cv::Mat  model_in;     // 1280×720 BGR copy kept for the push call
  double   read_ms{0}, preproc_ms{0};
  int      idx{0};
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
  for (const auto& b : boxes) {
    int x1 = std::clamp((int)std::round(b.x1), 0, frame.cols-1);
    int y1 = std::clamp((int)std::round(b.y1), 0, frame.rows-1);
    int x2 = std::clamp((int)std::round(b.x2), 0, frame.cols-1);
    int y2 = std::clamp((int)std::round(b.y2), 0, frame.rows-1);
    if (x2 <= x1 || y2 <= y1) continue;
    const cv::Scalar col = kColors[(size_t)std::abs(b.class_id) % kColors.size()];
    const std::string lbl = (b.class_id >= 0 && (size_t)b.class_id < labels.size()
                              ? labels[b.class_id] : std::to_string(b.class_id))
                          + " " + cv::format("%.2f", b.score);
    cv::rectangle(frame, {x1,y1}, {x2,y2}, col, 2);
    int bl = 0;
    cv::Size tsz = cv::getTextSize(lbl, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &bl);
    int yt = std::max(0, y1 - tsz.height - 4);
    cv::rectangle(frame, {x1,yt}, {x1+tsz.width, std::max(0,y1)}, col, cv::FILLED);
    cv::putText(frame, lbl, {x1, std::max(0,y1-2)},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,0,0}, 1, cv::LINE_AA);
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

// ── GStreamer encode pipeline (same as hw-pipeline) ───────────────────────────
static std::string build_hw_encode_gst(const std::string& host, int port,
                                       int w, int h, int fps) {
  std::ostringstream s;
  s << "appsrc ! videoconvert"
    << " ! video/x-raw,format=NV12,width=" << w << ",height=" << h
    << ",framerate=" << fps << "/1"
    << " ! neatencoder enc-type=h264 enc-fmt=NV12"
    << " enc-width=" << w << " enc-height=" << h
    << " enc-bitrate=3000 enc-frame-rate=" << fps
    << " enc-level=4.2 enc-profile=main"
    << " ! rtph264pay config-interval=1 pt=96"
    << " ! udpsink host=" << host << " port=" << port << " sync=false async=false";
  return s.str();
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  ::setenv("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY", "1", 0);

  // ── Argument parsing ──────────────────────────────────────────────────────
  std::string rtsp_url, insight_host, classes_arg;
  int  rtsp_frames = 99999, video_port = 9000, meta_port = 9100;
  bool use_zerocopy = false;
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
    else pos_args.push_back(a);
  }
  if (pos_args.size() < 2 || rtsp_url.empty()) {
    std::cerr << "Usage: " << argv[0]
              << " <model> <labels> --rtsp <URL>\n"
              << "  [--insight <host>] [--frames N] [--classes <csv>]\n"
              << "  [--video-port N] [--meta-port N] [--zerocopy]\n";
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
    model_opt.preprocess.input_max_width  = kInputW;
    model_opt.preprocess.input_max_height = kInputH;
    model_opt.preprocess.input_max_depth  = 3;
    model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;

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
    run_opt.queue_depth      = 3;
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
              << " BGR | async push/pull | queue_depth=3"
              << " | output=" << (use_zerocopy ? "ZeroCopy" : "Owned") << "\n";

    // ── SDK MeasureScope (plugin latency breakdown) ───────────────────────────
    // Wraps the entire push/pull workload. Reports avg_push_us / avg_pull_wait_us /
    // avg_copy_us / avg_alloc_us after threads join, revealing where the 8-10ms
    // overhead beyond MLA hw time (~6ms) actually goes.
    simaai::neat::MeasureOptions mopt;
    mopt.warmup_ms             = 2000;  // ~130 frames warm-up at 65fps
    mopt.include_plugin_latency = true; // per-node (EV74+MLA) breakdown
    auto mscope = runner.start_measurement(mopt);

    // ── FFmpeg low-latency RTSP decode ────────────────────────────────────────
    ::setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS",
             "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
             "|analyzeduration;100000|probesize;65536", 1);
    std::cout << "[DEC] opening FFmpeg RTSP: " << rtsp_url << "\n";
    cv::VideoCapture cap(rtsp_url, cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
      std::cerr << "[DEC] failed to open RTSP stream via FFmpeg\n";
      return 5;
    }
    const int stream_fps = std::max(1, static_cast<int>(cap.get(cv::CAP_PROP_FPS)));
    const int encode_fps = (stream_fps < 5 || stream_fps > 120) ? 60 : stream_fps;
    std::cout << "[DEC] opened " << kInputW << "x" << kInputH
              << "@" << stream_fps << " (encode fps=" << encode_fps << ")\n";

    // ── HW encode: neatencoder via GStreamer cv::VideoWriter ──────────────────
    cv::VideoWriter hw_writer;
    if (!insight_host.empty()) {
      const std::string encode_gst = build_hw_encode_gst(
          insight_host, video_port, kInputW, kInputH, encode_fps);
      std::cout << "[HW-ENC] pipeline:\n  " << encode_gst << "\n";
      hw_writer.open(encode_gst, cv::CAP_GSTREAMER, 0,
                     static_cast<double>(encode_fps),
                     cv::Size(kInputW, kInputH), true);
      if (hw_writer.isOpened())
        std::cout << "[insight] video  -> udp://" << insight_host << ":" << video_port << "\n";
      else
        std::cerr << "[HW-ENC] warn: neatencoder failed to open — video streaming disabled\n";
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

    // ── 5-thread async pipeline ───────────────────────────────────────────────
    //
    //  Thread A [FFmpeg-decode + preproc + runner.push()] → push_rec_q(4)
    //  Thread B [runner.pull() → tensors               ] → fast_post_q(4)
    //  Thread C [post-process + draw                   ] → fast_encode_q(16)
    //  Thread D [dispatch, non-blocking                ] → hw_write_q(1)
    //  Thread E [neatencoder write ~20ms               ]
    //
    // Thread A and Thread B share the runner: A calls push(), B calls pull().
    // push_rec_q carries the display frame and timing context from A to B,
    // FIFO-aligned with runner's internal queue (same ordering guaranteed by
    // OverflowPolicy::Block — no frames are silently dropped inside the runner).
    //
    PipelineQueue<PushRecord> push_rec_q(4);    // A → B: display + timing
    PipelineQueue<PostIn>     fast_post_q(4);   // B → C: tensors + display
    PipelineQueue<cv::Mat>    fast_encode_q(16);// C → D
    PipelineQueue<cv::Mat>    hw_write_q(1);    // D → E: single slot, drops if E busy

    std::atomic<int>  frames_done{0};
    std::atomic<int>  frames_dropped{0};
    Clock::time_point t_first_out{};
    std::atomic<bool> first_out_seen{false};

    // stage_acc: [0]=read, [1]=preproc, [2]=push, [3]=pull, [4]=post, [5]=encode
    std::array<TimingAcc, 6> stage_acc;

    // ── Thread A: FFmpeg decode + preproc + runner.push() ─────────────────────
    // Sole thread calling runner.push(). After push, saves display frame to
    // push_rec_q so Thread B can retrieve the matching frame after pull().
    std::thread push_th([&]() {
      for (int fi = 0; fi < rtsp_frames; ++fi) {
        cv::Mat raw;
        auto t0 = Clock::now();
        if (!cap.read(raw) || raw.empty()) {
          std::cerr << "[DEC] stream ended at frame " << fi << "\n";
          break;
        }
        double read_ms = Ms(Clock::now() - t0).count();
        if (fi > 0) stage_acc[0].add(read_ms);

        auto t1 = Clock::now();
        if (raw.type() != CV_8UC3) {
          cv::Mat tmp;
          if (raw.channels() == 4) cv::cvtColor(raw, tmp, cv::COLOR_BGRA2BGR);
          else                     raw.convertTo(tmp, CV_8UC3);
          raw = std::move(tmp);
        }
        cv::Mat model_in;
        if (raw.cols == kInputW && raw.rows == kInputH) model_in = raw.clone();
        else cv::resize(raw, model_in, {kInputW, kInputH}, 0, 0, cv::INTER_LINEAR);
        cv::Mat display   = raw.clone();
        double  preproc_ms = Ms(Clock::now() - t1).count();
        if (fi > 0) stage_acc[1].add(preproc_ms);

        // push() is non-blocking until the internal queue (queue_depth=3) is
        // full, then it blocks until Thread B's pull() frees a slot.
        auto t2 = Clock::now();
        if (!runner.push(std::vector<cv::Mat>{model_in})) {
          std::cerr << "[PUSH] runner.push() rejected frame " << fi << "\n";
          break;
        }
        double push_ms = Ms(Clock::now() - t2).count();
        if (fi > 0) stage_acc[2].add(push_ms);

        // Enqueue display record AFTER push so that push_rec_q order matches
        // the runner's internal FIFO order — ensuring pull/pop alignment.
        push_rec_q.push({std::move(display), std::move(model_in),
                         read_ms, preproc_ms, fi});
      }
      // Signal Thread B: no more frames.
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
        if (pr.idx > 0) stage_acc[3].add(pull_ms);

        // Extract TensorList from Sample using the SDK helper.
        // model.build().pull() may return a Bundle (kind==Bundle, fields contains
        // the per-output samples) rather than a plain TensorSet. stages::Tensors()
        // recursively collects tensors from all sample kinds (TensorSet, Tensor,
        // Bundle), giving the same TensorList that runner.run() returns internally.
        simaai::neat::TensorList tensors = simaai::neat::stages::Tensors(sample);

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
            auto all       = scale_boxes(boxes_720, pi.display.cols, pi.display.rows);
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

        draw_boxes(pi.display, boxes, labels);

        if (hw_writer.isOpened())
          fast_encode_q.push(std::move(pi.display));

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
      cv::Mat frame;
      while (fast_encode_q.pop(frame)) {
        if (hw_writer.isOpened()) {
          if (!hw_write_q.try_push(std::move(frame)))
            ++frames_dropped;
        }
      }
      hw_write_q.close();
    });

    // ── Thread E: neatencoder write ────────────────────────────────────────────
    std::thread encode_th([&]() {
      cv::Mat frame;
      while (hw_write_q.pop(frame)) {
        auto t0 = Clock::now();
        hw_writer.write(frame);
        stage_acc[5].add(Ms(Clock::now() - t0).count());
      }
    });

    push_th.join();
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

      const char* names[] = {"read", "preproc", "push", "pull", "post", "encode"};
      std::cout << "\n── Stage timings ─────────────────────────────────────────\n";
      std::cout << cv::format("%-10s  %8s  %8s  %8s  %8s\n",
                              "stage", "mean ms", "min ms", "max ms", "count");
      for (int i = 0; i < 6; ++i)
        if (stage_acc[i].n > 0)
          std::cout << cv::format("%-10s  %8.2f  %8.2f  %8.2f  %8d\n",
                                  names[i], stage_acc[i].mean(),
                                  stage_acc[i].mn, stage_acc[i].mx, stage_acc[i].n);
    }

    cap.release();
    if (hw_writer.isOpened()) hw_writer.release();
    runner.close();
    std::cout << "Done: " << processed << " frames\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 4;
  }
}
