/**
 * @example yolov11s-1280-detection-pipeline.cpp
 * YOLOv11 1280x720 detection pipeline (post-surgery model, EV74 genpreproc).
 *
 * Model: 1280.tar.gz — internal pipeline EV74-genpreproc -> MLA -> EV74-detessdequant
 *   Input:  1280x720 BGR (preproc letterboxes to 1280x1280 internally)
 *   Output: 6 NDHWC float tensors
 *     tensors[0:3] bbox (H,W,4)  cx,cy,w,h in model-space pixels (DFL decoded in model)
 *     tensors[3:6] cls  (H,W,80) probabilities (already sigmoid, class_is_prob=true)
 *
 * Usage:
 *   Folder mode: yolov11s-1280-detection-pipeline <model.tar.gz> <labels.txt> <input_dir> <output_dir>
 *   RTSP mode:   yolov11s-1280-detection-pipeline <model.tar.gz> <labels.txt> <output_dir> --rtsp <URL>
 *                  [--frames N] [--insight <host>] [--video-port N] [--meta-port N]
 *
 * OptiView / neat-insight streaming (RTSP mode only):
 *   --insight <host>   IP of the host running neat-insight (e.g. 172.16.1.207)
 *   --video-port N     UDP port for H.264 RTP video  (default 9000)
 *   --meta-port N      UDP port base for JSON metadata (default 9100)
 */

#include "neat.h"

#include "support/object_detection/obj_detection_utils.h"

#include <nodes/io/MetadataSender.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#ifdef __ARM_NEON
#  include <arm_neon.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

namespace {

struct FrameTiming {
  double read_ms    = 0.0;
  double preproc_ms = 0.0;
  double infer_ms   = 0.0;
  double post_ms    = 0.0;
  double render_ms  = 0.0;
  double save_ms    = 0.0;
  double total() const { return read_ms + preproc_ms + infer_ms + post_ms + render_ms + save_ms; }
};

struct TimingAcc {
  double sum = 0.0, mn = 1e18, mx = -1.0;
  int n = 0;
  void add(double v) { sum += v; mn = std::min(mn, v); mx = std::max(mx, v); ++n; }
  double mean() const { return n > 0 ? sum / n : 0.0; }
};

// Model input dimensions (what EV74 preproc expects)
constexpr int kInputW  = 1280;
constexpr int kInputH  = 720;
// Model internal dimensions (after preproc CENTER letterbox)
constexpr int kModelW  = 1280;
constexpr int kModelH  = 1280;

// CENTER letterbox params: 1280x720 -> 1280x1280
//   gain     = min(1280/720, 1280/1280) = 1.0
//   pad_top  = (1280 - 720 * 1.0) / 2   = 280.0
//   pad_left = (1280 - 1280 * 1.0) / 2  = 0.0
constexpr float kLbGain    = 1.0f;
constexpr float kLbPadTop  = 280.0f;
constexpr float kLbPadLeft = 0.0f;

constexpr float kMinScore  = 0.60f;
constexpr float kNmsIou    = 0.50f;
constexpr int   kMaxDet    = 24;   // topk from yolo_postproc.json
constexpr int   kTimeoutMs = 5000;

// Tensor data held as raw bytes (exactly as returned by copy_dense_bytes_tight).
// Avoids a second allocation and a memcpy: caller moves the byte vector in directly.
struct TensorHWC {
  int h = 0;
  int w = 0;
  int c = 0;
  std::vector<uint8_t> raw;   // float32 bytes, row-major HWC layout
  [[nodiscard]] const float* floats() const noexcept {
    return reinterpret_cast<const float*>(raw.data());
  }
};

#ifdef __ARM_NEON
// Vectorised argmax over exactly 80 float32 values.
// Uses 4 independent accumulators to hide vmaxq latency on A65's out-of-order engine.
// Pass 1: NEON horizontal-max (20 vmaxq_f32 total, then vmaxvq_f32 scalar reduction).
// Pass 2: scalar scan for the winning index (typically short-circuits very early).
static void argmax80_neon(const float* __restrict__ p, float& best_val, int& best_idx) {
  float32x4_t v0 = vld1q_f32(p +  0);
  float32x4_t v1 = vld1q_f32(p +  4);
  float32x4_t v2 = vld1q_f32(p +  8);
  float32x4_t v3 = vld1q_f32(p + 12);
  for (int i = 16; i < 80; i += 16) {
    v0 = vmaxq_f32(v0, vld1q_f32(p + i +  0));
    v1 = vmaxq_f32(v1, vld1q_f32(p + i +  4));
    v2 = vmaxq_f32(v2, vld1q_f32(p + i +  8));
    v3 = vmaxq_f32(v3, vld1q_f32(p + i + 12));
  }
  best_val = vmaxvq_f32(vmaxq_f32(vmaxq_f32(v0, v1), vmaxq_f32(v2, v3)));
  best_idx = -1;
  for (int i = 0; i < 80; ++i) {
    if (p[i] == best_val) { best_idx = i; break; }
  }
}
#endif

// Bounded blocking queue: thread-safe FIFO with close() for pipeline shutdown.
template<typename T>
class PipelineQueue {
public:
  explicit PipelineQueue(size_t cap) : cap_(cap) {}

  void push(T item) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_push_.wait(lk, [&] { return q_.size() < cap_ || closed_; });
    if (closed_) return;
    q_.push(std::move(item));
    cv_pop_.notify_one();
  }

  bool pop(T& out) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_pop_.wait(lk, [&] { return !q_.empty() || closed_; });
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop();
    cv_push_.notify_one();
    return true;
  }

  bool try_pop(T& out) {
    std::lock_guard<std::mutex> lk(mu_);
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop();
    cv_push_.notify_one();
    return true;
  }

  void close() {
    std::lock_guard<std::mutex> lk(mu_);
    closed_ = true;
    cv_pop_.notify_all();
    cv_push_.notify_all();
  }

private:
  std::queue<T>           q_;
  std::mutex              mu_;
  std::condition_variable cv_push_, cv_pop_;
  size_t                  cap_;
  bool                    closed_{false};
};

struct InferInput {
  cv::Mat bgr;
  double  read_ms{0.0};
  int     idx{0};
};

struct InferOutput {
  cv::Mat                  bgr;    // frame with boxes already drawn
  std::vector<objdet::Box> boxes;
  FrameTiming              ft;
  int                      idx{0};
};

// 3-stage fast RTSP pipeline structs:
//   ReadTh → [fast_infer_q] → InferTh → [fast_post_q] → PostTh
struct PreprocOut {
  cv::Mat  model_in;      // kInputW×kInputH BGR owned copy for runner
  cv::Mat  display;       // original decoded frame owned for rendering
  double   read_ms{0.0};
  double   preproc_ms{0.0};
  int      idx{0};
};

struct PostIn {
  cv::Mat                  display;
  simaai::neat::TensorList tensors;
  FrameTiming              ft;
  int                      idx{0};
};

// Build a low-latency GStreamer RTSP source pipeline for cv::VideoCapture.
//
// latency=0        : disables the rtspsrc jitter buffer — eliminates up to 200 ms
//                    of internal buffering that the default value (200 ms) adds.
// buffer-mode=auto : rtspsrc picks the best RTP buffer strategy for the network.
// avdec_h264       : libavcodec H.264 software decoder; max-threads=2 limits
//                    decoder threads to avoid starving the MLA infer thread.
// drop=true        : appsink drops old frames when the consumer is behind — the
//                    infer thread always receives the freshest decoded frame.
// max-buffers=2    : cap the appsink queue depth so latency stays bounded.
// sync=false       : do not clock-synchronise presentation; run as fast as possible.
//
// If out_w/out_h are non-zero, a videoscale element is inserted so GStreamer
// delivers frames already at the model input resolution, saving a cv::resize
// call on the A65 CPU.
static std::string gst_rtsp_read_pipeline(const std::string& url,
                                          int out_w = 0, int out_h = 0) {
  std::ostringstream s;
  // decodebin auto-selects the RTP de-payloader and H.264 decoder.
  // timeout=2000        : SDP/DESCRIBE timeout in ms — fail in 2 s if server unreachable.
  // tcp-timeout=2000000 : TCP connect timeout in µs (2 s).
  // These short timeouts make the background probe thread exit in ~2-3 s so we
  // can join() it cleanly without stalling SSH or the main loop.
  s << "rtspsrc location=" << url
    << " timeout=2000 tcp-timeout=2000000 latency=200"
    << " ! decodebin"
    << " ! videoconvert";
  if (out_w > 0 && out_h > 0) {
    // Pre-scale in GStreamer to model input size so cv::resize in infer_bgr is skipped.
    s << " ! videoscale"
      << " ! video/x-raw,format=BGR,width=" << out_w << ",height=" << out_h;
  } else {
    s << " ! video/x-raw,format=BGR";
  }
  // drop=true     : appsink drops stale frames — infer thread always gets the freshest one.
  // max-buffers=2 : bound queue depth to keep latency low.
  s << " ! appsink drop=true max-buffers=2 sync=false";
  return s.str();
}

bool is_image(const fs::path& p) {
  std::string ext = p.extension().string();
  for (char& c : ext) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
}

std::vector<std::string> load_labels(const fs::path& labels_path) {
  std::ifstream in(labels_path);
  if (!in.good()) {
    throw std::runtime_error("labels file does not exist: " + labels_path.string());
  }
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty()) {
      labels.push_back(line);
    }
  }
  if (labels.empty()) {
    throw std::runtime_error("labels file is empty: " + labels_path.string());
  }
  return labels;
}

TensorHWC tensor_to_hwc_f32(const simaai::neat::Tensor& t) {
  if (t.dtype != simaai::neat::TensorDType::Float32) {
    throw std::runtime_error("expected Float32 tensor for YOLO decode");
  }
  // Strip leading size-1 dims to handle NHWC (4D), NDHWC (5D), etc.
  auto shape = t.shape;
  while (shape.size() > 3 && shape.front() == 1) {
    shape.erase(shape.begin());
  }
  TensorHWC out;
  if (shape.size() == 3) {
    out.h = static_cast<int>(shape[0]);
    out.w = static_cast<int>(shape[1]);
    out.c = static_cast<int>(shape[2]);
  } else {
    throw std::runtime_error("unexpected tensor rank for HWC decode: " +
                             std::to_string(t.shape.size()));
  }
  const size_t elems =
      static_cast<size_t>(out.h) * static_cast<size_t>(out.w) * static_cast<size_t>(out.c);
  // Move the byte vector directly into TensorHWC::raw — zero extra allocation,
  // zero memcpy. The SDK already copied the tensor data once into this vector.
  out.raw = t.copy_dense_bytes_tight();
  if (out.raw.size() < elems * sizeof(float)) {
    throw std::runtime_error("tensor byte size is smaller than expected");
  }
  return out;
}

float iou_xyxy(const objdet::Box& a, const objdet::Box& b) {
  const float xx1 = std::max(a.x1, b.x1);
  const float yy1 = std::max(a.y1, b.y1);
  const float xx2 = std::min(a.x2, b.x2);
  const float yy2 = std::min(a.y2, b.y2);
  const float iw = std::max(0.0f, xx2 - xx1);
  const float ih = std::max(0.0f, yy2 - yy1);
  const float inter = iw * ih;
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  const float den = area_a + area_b - inter;
  return den > 0.0f ? (inter / den) : 0.0f;
}

// Decode post-surgery YOLOv11 tensors.
//   tensors[0:3] bbox (H,W,4)  cx,cy,w,h in model pixels (DFL already decoded in model)
//   tensors[3:6] cls  (H,W,80) probabilities (already sigmoid, class_is_prob=true)
// Returns boxes in kInputW x kInputH (1280x720) coordinate space.
std::vector<objdet::Box> decode_yolov11_postsurgery(
    const simaai::neat::TensorList& tensors, float min_score) {
  if (tensors.size() < 6) {
    throw std::runtime_error("expected 6 tensors from detessdequant, got " +
                             std::to_string(tensors.size()));
  }

  const float inv_gain = 1.0f / kLbGain;

  std::vector<objdet::Box> candidates;
  candidates.reserve(4000);

  for (size_t level = 0; level < 3; ++level) {
    const TensorHWC bbox = tensor_to_hwc_f32(tensors[level]);
    const TensorHWC cls  = tensor_to_hwc_f32(tensors[level + 3]);

    if (bbox.c != 4) {
      throw std::runtime_error("bbox tensor[" + std::to_string(level) +
                               "] has unexpected channel count " + std::to_string(bbox.c) +
                               " (expected 4: cx,cy,w,h)");
    }
    if (bbox.h != cls.h || bbox.w != cls.w) {
      throw std::runtime_error("bbox/cls tensor spatial size mismatch at level " +
                               std::to_string(level));
    }

    // Running pointers: advance by cls.c / 4 floats per cell — no per-cell multiply.
    const float* cls_ptr  = cls.floats();
    const float* bbox_ptr = bbox.floats();
    for (int y = 0; y < cls.h; ++y) {
      for (int x = 0; x < cls.w; ++x, cls_ptr += cls.c, bbox_ptr += 4) {
        // Class probs are already sigmoid (class_is_prob=true)
        int best_class = -1;
        float best_score = 0.0f;
#ifdef __ARM_NEON
        if (cls.c == 80) {
          argmax80_neon(cls_ptr, best_score, best_class);
        } else
#endif
        {
          for (int c = 0; c < cls.c; ++c) {
            if (cls_ptr[c] > best_score) { best_score = cls_ptr[c]; best_class = c; }
          }
        }
        // Most cells are background: skip quickly.
        if (best_score < min_score || best_class < 0) [[likely]]
          continue;

        const float cx = bbox_ptr[0];
        const float cy = bbox_ptr[1];
        const float bw = bbox_ptr[2];
        const float bh = bbox_ptr[3];

        // Inverse CENTER letterbox: model space (1280x1280) -> original space (1280x720)
        const float x1 = std::max(0.0f, std::min(static_cast<float>(kInputW),
                                                  (cx - bw * 0.5f - kLbPadLeft) * inv_gain));
        const float y1 = std::max(0.0f, std::min(static_cast<float>(kInputH),
                                                  (cy - bh * 0.5f - kLbPadTop) * inv_gain));
        const float x2 = std::max(0.0f, std::min(static_cast<float>(kInputW),
                                                  (cx + bw * 0.5f - kLbPadLeft) * inv_gain));
        const float y2 = std::max(0.0f, std::min(static_cast<float>(kInputH),
                                                  (cy + bh * 0.5f - kLbPadTop) * inv_gain));
        if (x2 > x1 && y2 > y1) [[unlikely]]
          candidates.emplace_back(x1, y1, x2, y2, best_score, best_class);
      }
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const objdet::Box& a, const objdet::Box& b) { return a.score > b.score; });

  std::vector<objdet::Box> keep;
  keep.reserve(static_cast<size_t>(kMaxDet));
  for (const auto& box : candidates) {
    bool suppressed = false;
    for (const auto& kept : keep) {
      if (kept.class_id == box.class_id && iou_xyxy(kept, box) > kNmsIou) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      keep.push_back(box);  // Box is a small POD — push_back and emplace_back are equivalent
      if (static_cast<int>(keep.size()) >= kMaxDet) [[unlikely]]
        break;
    }
  }
  return keep;
}

// Scale boxes from kInputW x kInputH (1280x720) to original image dimensions.
std::vector<objdet::Box> scale_boxes_to_original(const std::vector<objdet::Box>& boxes,
                                                  int orig_w, int orig_h) {
  const float sx = static_cast<float>(orig_w) / static_cast<float>(kInputW);
  const float sy = static_cast<float>(orig_h) / static_cast<float>(kInputH);
  std::vector<objdet::Box> out;
  out.reserve(boxes.size());
  for (const auto& b : boxes) {
    out.emplace_back(b.x1 * sx, b.y1 * sy, b.x2 * sx, b.y2 * sy, b.score, b.class_id);
  }
  return out;
}

std::string class_name(const std::vector<std::string>& labels, int class_id) {
  if (class_id >= 0 && static_cast<size_t>(class_id) < labels.size()) {
    return labels[static_cast<size_t>(class_id)];
  }
  return std::to_string(class_id);
}

cv::Scalar class_color(int class_id) {
  static const std::array<cv::Scalar, 8> kColors = {
      cv::Scalar(0, 255, 0),   cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255),
      cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255),
      cv::Scalar(128, 255, 0), cv::Scalar(255, 128, 0)};
  const size_t idx = static_cast<size_t>(class_id >= 0 ? class_id : -class_id) % kColors.size();
  return kColors[idx];
}

void draw_boxes(cv::Mat& frame, const std::vector<objdet::Box>& boxes,
                const std::vector<std::string>& labels) {
  for (const auto& b : boxes) {
    const int x1 = std::max(0, std::min(frame.cols - 1, static_cast<int>(std::round(b.x1))));
    const int y1 = std::max(0, std::min(frame.rows - 1, static_cast<int>(std::round(b.y1))));
    const int x2 = std::max(0, std::min(frame.cols - 1, static_cast<int>(std::round(b.x2))));
    const int y2 = std::max(0, std::min(frame.rows - 1, static_cast<int>(std::round(b.y2))));
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }
    const cv::Scalar color = class_color(b.class_id);
    const std::string text = class_name(labels, b.class_id) + " " + cv::format("%.2f", b.score);
    cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
    int baseline = 0;
    const cv::Size tsz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    const int y_text_top = std::max(0, y1 - tsz.height - 4);
    cv::rectangle(frame, cv::Point(x1, y_text_top), cv::Point(x1 + tsz.width, std::max(0, y1)),
                  color, cv::FILLED);
    cv::putText(frame, text, cv::Point(x1, std::max(0, y1 - 2)), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
  }
}

void draw_timing_overlay(cv::Mat& frame, const FrameTiming& t) {
  const std::string row0 =
      cv::format("read=%.1fms  preproc=%.1fms  infer=%.1fms",
                 t.read_ms, t.preproc_ms, t.infer_ms);
  const std::string row1 =
      cv::format("post=%.1fms  render=%.1fms  save=%.1fms  total=%.1fms",
                 t.post_ms, t.render_ms, t.save_ms, t.total());
  constexpr int    kFont = cv::FONT_HERSHEY_SIMPLEX;
  constexpr double kSc   = 0.42;
  constexpr int    kTh   = 1;
  int baseline = 0;
  int y = frame.rows - 4;
  for (const std::string* row : {&row1, &row0}) {
    const cv::Size sz = cv::getTextSize(*row, kFont, kSc, kTh, &baseline);
    cv::rectangle(frame,
                  cv::Point(2, y - sz.height - 2),
                  cv::Point(4 + sz.width, y + baseline + 2),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame, *row, cv::Point(4, y), kFont, kSc,
                cv::Scalar(0, 255, 255), kTh, cv::LINE_AA);
    y -= sz.height + baseline + 6;
  }
}

void print_timing_summary(const std::array<TimingAcc, 6>& acc, int count) {
  const std::array<const char*, 6> names = {"read", "preproc", "infer", "postproc", "render", "save"};
  TimingAcc total_acc;
  std::cout << "\n── Stage timings (" << count << " frames) ──────────────────────────\n";
  std::cout << cv::format("%-10s  %8s  %8s  %8s\n", "stage", "mean ms", "min ms", "max ms");
  std::cout << std::string(46, '-') << "\n";
  double total_sum = 0.0;
  double total_mn = 1e18, total_mx = -1.0;
  for (int i = 0; i < 6; ++i) {
    const auto& a = acc[i];
    total_sum += a.sum;
    total_mn  += a.mn;   // sum of per-frame stage mins = lower bound on total
    total_mx  += a.mx;
    std::cout << cv::format("%-10s  %8.2f  %8.2f  %8.2f\n",
                            names[i], a.mean(), a.mn, a.mx);
  }
  std::cout << std::string(46, '-') << "\n";
  const double mean_total = count > 0 ? total_sum / count : 0.0;
  std::cout << cv::format("%-10s  %8.2f\n", "total", mean_total);
  std::cout << std::string(46, '-') << "\n";
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  // Allow CPU tensor → EV74 DMA copy for EV74-genpreproc route (setenv 0 = don't overwrite).
  ::setenv("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY", "1", 0);

  // Parse args: collect positional args and optional flags.
  std::string rtsp_url;
  int rtsp_frames = 50;
  std::string insight_host;
  int video_port = 9000;
  int meta_port  = 9100;
  std::string classes_arg;   // comma-separated class whitelist, e.g. "person,car,truck"
  std::vector<std::string> pos_args;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--rtsp" && i + 1 < argc) {
      rtsp_url = argv[++i];
    } else if (a == "--frames" && i + 1 < argc) {
      rtsp_frames = std::stoi(argv[++i]);
    } else if (a == "--insight" && i + 1 < argc) {
      insight_host = argv[++i];
    } else if (a == "--video-port" && i + 1 < argc) {
      video_port = std::stoi(argv[++i]);
    } else if (a == "--meta-port" && i + 1 < argc) {
      meta_port = std::stoi(argv[++i]);
    } else if (a == "--classes" && i + 1 < argc) {
      classes_arg = argv[++i];
    } else {
      pos_args.push_back(a);
    }
  }

  const bool rtsp_mode = !rtsp_url.empty();
  // RTSP mode: model + labels required; output_dir is optional (omit to stream-only, no disk writes).
  // Folder mode: model + labels + input_dir + output_dir all required.
  const int required_pos = rtsp_mode ? 2 : 4;
  if (static_cast<int>(pos_args.size()) < required_pos) {
    std::cerr << "Usage:\n"
              << "  Folder mode:       " << argv[0] << " <model> <labels> <input_dir> <output_dir>\n"
              << "  RTSP stream+save:  " << argv[0] << " <model> <labels> <output_dir> --rtsp <URL>"
              << " [--insight <host>] [--frames N] [--video-port N] [--meta-port N]\n"
              << "  RTSP stream-only:  " << argv[0] << " <model> <labels> --rtsp <URL>"
              << " --insight <host> [--frames N] [--video-port N] [--meta-port N]\n";
    return 1;
  }

  const std::string model_path  = pos_args[0];
  const fs::path    labels_path = pos_args[1];
  const fs::path    input_dir   = rtsp_mode ? fs::path{} : fs::path{pos_args[2]};
  // RTSP: output_dir is optional (3rd positional arg). Empty → stream-only, no files written.
  const fs::path    output_dir  = rtsp_mode
      ? (pos_args.size() >= 3 ? fs::path{pos_args[2]} : fs::path{})
      : fs::path{pos_args[3]};

  if (!rtsp_mode && !fs::is_directory(input_dir)) {
    std::cerr << "Input directory does not exist: " << input_dir << "\n";
    return 2;
  }
  if (!output_dir.empty()) fs::create_directories(output_dir);

  std::vector<std::string> labels;
  try {
    labels = load_labels(labels_path);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }

  // Build class-id whitelist from --classes argument (empty = accept all).
  std::unordered_set<int> allowed_classes;
  if (!classes_arg.empty()) {
    std::istringstream ss(classes_arg);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      bool found = false;
      for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] == tok) { allowed_classes.insert(static_cast<int>(i)); found = true; break; }
      }
      if (!found) std::cerr << "[WARN] --classes: unknown label '" << tok << "' ignored\n";
    }
    std::cout << "[filter] class whitelist (" << allowed_classes.size() << "): " << classes_arg << "\n";
  }

  std::vector<fs::path> images;
  if (!rtsp_mode) {
    for (const auto& entry : fs::directory_iterator(input_dir)) {
      if (entry.is_regular_file() && is_image(entry.path())) {
        images.push_back(entry.path());
      }
    }
    std::sort(images.begin(), images.end());

    if (images.empty()) {
      std::cerr << "No images found in " << input_dir << "\n";
      return 3;
    }
    std::cout << "Found " << images.size() << " images\n";
  } else {
    if (output_dir.empty())
      std::cout << "[RTSP] stream-only mode (no disk writes) -> capturing " << rtsp_frames << " frames\n";
    else
      std::cout << "[RTSP] stream -> capturing " << rtsp_frames << " frames -> " << output_dir << "\n";
  }

  try {
    simaai::neat::Model::Options model_opt;

    model_opt.preprocess.input_max_width  = kInputW;
    model_opt.preprocess.input_max_height = kInputH;
    model_opt.preprocess.input_max_depth  = 3;
    model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;

    simaai::neat::Model model(model_path, model_opt);

    // model.build() runs an EV74 preflight that can time out on the first attempt due to
    // EV74 dispatch warm-up or an RPMsg lock held by another system process.
    // The SDK hint explicitly says this is intermittent — retry up to 3 times.
    auto runner = [&]() {
      for (int attempt = 1; ; ++attempt) {
        try {
          return model.build();
        } catch (const std::exception& e) {
          if (attempt >= 3) throw;
          std::cerr << "[BUILD] attempt " << attempt << "/3 failed: " << e.what()
                    << "\n[BUILD] retrying in 3 s (EV74 warm-up)...\n";
          std::this_thread::sleep_for(std::chrono::seconds(3));
        }
      }
    }();
    std::cout << "[BUILD] Model loaded: " << model_path << "\n";
    std::cout << "[BUILD] Input: " << kInputW << "x" << kInputH << " BGR -> 1280x1280 letterbox\n";

    // Per-stage timing accumulators: [read, preproc, infer, post, render, save]
    std::array<TimingAcc, 6> stage_acc;

    // infer_bgr: resize → runner.run → decode+NMS → scale to original coords.
    // Called exclusively from the infer thread — runner is not shared across threads.
    auto infer_bgr = [&](const cv::Mat& bgr_in,
                         double& preproc_ms, double& infer_ms, double& post_ms)
        -> std::vector<objdet::Box> {
      auto _t0 = Clock::now();
      // Ensure the frame is CV_8UC3 BGR — the only format EV74 preproc accepts.
      // GStreamer appsink is configured for BGR; FFmpeg VideoCapture also gives BGR.
      // The explicit check guards against unexpected BGRA from some decoders.
      cv::Mat bgr_src;
      if (bgr_in.type() == CV_8UC3) {
        bgr_src = bgr_in;  // already correct, borrow the header (read-only below)
      } else if (bgr_in.channels() == 4) {
        cv::cvtColor(bgr_in, bgr_src, cv::COLOR_BGRA2BGR);
      } else {
        bgr_in.convertTo(bgr_src, CV_8UC3);
      }
      // Resize to model input resolution.
      // Use clone() when already the right size so frame_720 owns independent data,
      // matching the cv::resize() path (both produce a deep copy for the runner).
      cv::Mat frame_720;
      if (bgr_src.cols == kInputW && bgr_src.rows == kInputH) {
        frame_720 = bgr_src.clone();
      } else {
        cv::resize(bgr_src, frame_720, cv::Size(kInputW, kInputH), 0, 0, cv::INTER_LINEAR);
      }
      preproc_ms = Ms(Clock::now() - _t0).count();

      simaai::neat::TensorList tensors_out;
      { auto _ti = Clock::now(); tensors_out = runner.run(std::vector<cv::Mat>{std::move(frame_720)}, kTimeoutMs); infer_ms = Ms(Clock::now() - _ti).count(); }
      auto _tp = Clock::now();
      auto boxes_720 = decode_yolov11_postsurgery(tensors_out, kMinScore);
      post_ms = Ms(Clock::now() - _tp).count();
      return scale_boxes_to_original(boxes_720, bgr_src.cols, bgr_src.rows);
    };

    int processed = 0;
    // save_to_disk: folder mode always true; RTSP mode true only when output_dir was supplied.
    const bool save_to_disk = !output_dir.empty();

    // ── Async PNG save thread (started only when save_to_disk) ───────────
    struct SaveTask { cv::Mat frame; fs::path path; };
    std::queue<SaveTask>      save_q;
    std::mutex                save_mu;
    std::condition_variable   save_cv_var;
    std::atomic<bool>         save_stop{false};
    std::thread save_thread;
    if (save_to_disk) {
      save_thread = std::thread([&]() {
        while (true) {
          std::unique_lock<std::mutex> lk(save_mu);
          save_cv_var.wait(lk, [&]{ return !save_q.empty() || save_stop.load(); });
          if (save_q.empty() && save_stop.load()) break;
          auto task = std::move(save_q.front()); save_q.pop();
          lk.unlock();
          cv::imwrite(task.path.string(), task.frame);
        }
      });
    }

    // ── Inference pipeline: infer_q → infer_th → result_q ────────────────
    // infer_q capacity 2: main reads one ahead while infer thread is busy.
    // result_q capacity 8: generous headroom so infer thread never stalls on push.
    PipelineQueue<InferInput>  infer_q(2);
    PipelineQueue<InferOutput> result_q(8);

    std::thread infer_th([&]() {
      InferInput inp;
      while (infer_q.pop(inp)) {
        FrameTiming ft;
        ft.read_ms = inp.read_ms;
        std::vector<objdet::Box> boxes;
        try {
          boxes = infer_bgr(inp.bgr, ft.preproc_ms, ft.infer_ms, ft.post_ms);
        } catch (const std::exception& e) {
          std::cerr << "Decode failed frame " << inp.idx << ": " << e.what() << "\n";
          result_q.push({std::move(inp.bgr), {}, ft, inp.idx});
          continue;
        }
        { auto _tr = Clock::now(); draw_boxes(inp.bgr, boxes, labels); ft.render_ms = Ms(Clock::now() - _tr).count(); }
        result_q.push({std::move(inp.bgr), std::move(boxes), ft, inp.idx});
      }
      result_q.close();
    });

    if (rtsp_mode) {
      // ── RTSP mode: 3-thread pipeline ──────────────────────────────────────
      //
      //   Thread A [read+preproc] → fast_infer_q(3) → Thread B [infer] → fast_post_q(4) → Thread C [post+render] → fast_encode_q(4) → Thread D [encode+stream]
      //
      //   read+preproc  : cap.read(10ms) + format/resize(0.5ms)
      //   infer         : runner.run(15ms)  ← MLA bottleneck → ceiling ~66 FPS
      //   post+render   : decode+NMS+draw (9ms)
      //   encode+stream : x264enc + udpsink (~15-20ms) — off the hot path on Thread D
      //
      // 4-thread pipeline: encode(N-1) overlaps post(N) which overlaps infer(N+1).
      // MLA never waits for CPU; x264 latency is hidden behind the next infer cycle.

      // Close the shared 2-thread infer_th (built for folder mode) — RTSP uses its own.
      infer_q.close();
      infer_th.join();

      // Open RTSP stream
      std::cout << "[RTSP] opening RTSP stream via FFmpeg (low-delay TCP): " << rtsp_url << "\n";
      setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS",
             "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
             "|analyzeduration;100000|probesize;65536", /*overwrite=*/1);
      cv::VideoCapture cap(rtsp_url, cv::CAP_FFMPEG);

      if (!cap.isOpened()) {
        std::cerr << "Failed to open RTSP stream: " << rtsp_url << "\n";
        { std::lock_guard<std::mutex> lk(save_mu); save_stop = true; }
        save_cv_var.notify_all();
        if (save_to_disk) save_thread.join();
        return 5;
      }
      const int stream_w   = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
      const int stream_h   = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
      const int stream_fps = std::max(1, static_cast<int>(cap.get(cv::CAP_PROP_FPS)));
      std::cout << "[RTSP] stream " << stream_w << "x" << stream_h << "@" << stream_fps
                << " -> model input " << kInputW << "x" << kInputH << "\n";

      // ── OptiView / neat-insight streaming setup ──────────────────────────
      cv::VideoWriter rtp_writer;
      std::unique_ptr<simaai::neat::MetadataSender> meta_sender;
      if (!insight_host.empty()) {
        std::ostringstream gst;
        gst << "appsrc ! videoconvert"
            << " ! video/x-raw,format=I420,width=" << stream_w
            << ",height=" << stream_h << ",framerate=" << stream_fps << "/1"
            << " ! x264enc speed-preset=ultrafast tune=zerolatency bitrate=1000"
            << " key-int-max=" << stream_fps
            << " ! rtph264pay config-interval=1 pt=96"
            << " ! udpsink host=" << insight_host
            << " port=" << video_port << " sync=false async=false";
        rtp_writer.open(gst.str(), cv::CAP_GSTREAMER, 0,
                        static_cast<double>(stream_fps), cv::Size(stream_w, stream_h), true);
        if (rtp_writer.isOpened())
          std::cout << "[insight] video  -> udp://" << insight_host << ":" << video_port << "\n";
        else
          std::cerr << "[insight] warn: failed to open GStreamer RTP writer — video disabled\n";

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
          std::cerr << "[insight] warn: MetadataSender: " << meta_err << " — metadata disabled\n";
          meta_sender.reset();
        }
      }

      // Inter-stage queues:
      //   fast_infer_q(3): read@10ms produces faster than infer@15ms consumes → allow 3 ahead
      //   fast_post_q(4):  infer@15ms is slower than post@7ms → post always drains quickly
      PipelineQueue<PreprocOut> fast_infer_q(3);
      PipelineQueue<PostIn>     fast_post_q(4);
      PipelineQueue<cv::Mat>    fast_encode_q(4);  // Thread C → Thread D (x264 encode)

      std::atomic<int>       frames_done{0};
      Clock::time_point      t_first_out{};
      std::atomic<bool>      first_out_seen{false};

      // ── Thread A: read + preproc ──────────────────────────────────────────
      // Reads from FFmpeg, normalises to CV_8UC3 BGR, makes two owned Mat copies:
      //   display   = original frame (moved, zero-copy) for box drawing + streaming
      //   model_in  = 1280x720 BGR clone for runner (1 memcpy per frame, 2.76 MB)
      // At 66 FPS this is 182 MB/s of memcpy — well within LPDDR4 bandwidth.
      std::thread read_th([&]() {
        for (int fi = 0; fi < rtsp_frames; ++fi) {
          cv::Mat raw;
          auto t0 = Clock::now();
          if (!cap.read(raw) || raw.empty()) {
            std::cerr << "[RTSP] stream ended at frame " << fi << "\n";
            break;
          }
          double read_ms = Ms(Clock::now() - t0).count();

          if (fi == 0)
            std::cout << "[RTSP] first frame: " << raw.cols << "x" << raw.rows
                      << "  CV_" << (raw.depth() == CV_8U ? "8U" : "other")
                      << "C" << raw.channels() << "\n";

          auto t1 = Clock::now();
          // Normalise to CV_8UC3 in-place (no-op for the common BGR path).
          if (raw.type() != CV_8UC3) {
            cv::Mat tmp;
            if (raw.channels() == 4) cv::cvtColor(raw, tmp, cv::COLOR_BGRA2BGR);
            else                     raw.convertTo(tmp, CV_8UC3);
            raw = std::move(tmp);
          }

          // model_in: independent copy at model resolution, runner takes ownership
          // display:  zero-copy move of raw (after above normalise raw is CV_8UC3)
          cv::Mat model_in;
          if (raw.cols == kInputW && raw.rows == kInputH) {
            model_in = raw.clone();
          } else {
            cv::resize(raw, model_in, cv::Size(kInputW, kInputH), 0, 0, cv::INTER_LINEAR);
          }
          cv::Mat display   = std::move(raw);
          double  preproc_ms = Ms(Clock::now() - t1).count();

          fast_infer_q.push({std::move(model_in), std::move(display), read_ms, preproc_ms, fi});
        }
        fast_infer_q.close();
      });

      // ── Thread B: MLA inference only ──────────────────────────────────────
      // Kept deliberately lean: no preproc, no postproc — just runner.run().
      // This ensures the MLA is always fed as fast as possible and never waits
      // for CPU postprocessing from the previous frame.
      std::thread infer_th_fast([&]() {
        PreprocOut pp;
        while (fast_infer_q.pop(pp)) {
          FrameTiming ft;
          ft.read_ms    = pp.read_ms;
          ft.preproc_ms = pp.preproc_ms;
          auto t0 = Clock::now();
          try {
            auto tensors = runner.run(
                std::vector<cv::Mat>{std::move(pp.model_in)}, kTimeoutMs);
            ft.infer_ms = Ms(Clock::now() - t0).count();
            fast_post_q.push({std::move(pp.display), std::move(tensors), ft, pp.idx});
          } catch (const std::exception& e) {
            ft.infer_ms = Ms(Clock::now() - t0).count();
            std::cerr << "MLA failed frame " << pp.idx << ": " << e.what() << "\n";
            fast_post_q.push({std::move(pp.display), {}, ft, pp.idx});
          }
        }
        fast_post_q.close();
      });

      // ── Thread C: post-process + render + output ──────────────────────────
      // Runs concurrently with infer(N+1): post(7ms) finishes well before the
      // next 15ms infer cycle, so the infer thread never stalls on fast_post_q.
      std::thread post_th([&]() {
        PostIn pi;
        while (fast_post_q.pop(pi)) {
          if (!first_out_seen.exchange(true)) t_first_out = Clock::now();

          FrameTiming& ft = pi.ft;

          auto t0 = Clock::now();
          std::vector<objdet::Box> boxes;
          if (!pi.tensors.empty()) {
            try {
              auto boxes_720 = decode_yolov11_postsurgery(pi.tensors, kMinScore);
              auto all = scale_boxes_to_original(boxes_720, pi.display.cols, pi.display.rows);
              if (allowed_classes.empty()) {
                boxes = std::move(all);
              } else {
                for (auto& b : all)
                  if (allowed_classes.count(b.class_id)) boxes.push_back(b);
              }
            } catch (const std::exception& e) {
              std::cerr << "Post failed frame " << pi.idx << ": " << e.what() << "\n";
            }
          }
          ft.post_ms = Ms(Clock::now() - t0).count();

          auto t1 = Clock::now();
          draw_boxes(pi.display, boxes, labels);
          ft.render_ms = Ms(Clock::now() - t1).count();

          // Hand rendered frame to Thread D (non-blocking push).
          // If save_to_disk is also needed, clone so both threads get their own copy.
          if (rtp_writer.isOpened()) {
            fast_encode_q.push(save_to_disk ? pi.display.clone() : std::move(pi.display));
          }

          if (save_to_disk) {
            const std::string fname = "frame_" + cv::format("%05d", pi.idx) + ".png";
            auto t2 = Clock::now();
            { std::lock_guard<std::mutex> lk(save_mu);
              save_q.push({std::move(pi.display), output_dir / fname});
              save_cv_var.notify_one(); }
            ft.save_ms = Ms(Clock::now() - t2).count();
          }

          int done = ++frames_done;
          stage_acc[0].add(ft.read_ms);   stage_acc[1].add(ft.preproc_ms);
          stage_acc[2].add(ft.infer_ms);  stage_acc[3].add(ft.post_ms);
          stage_acc[4].add(ft.render_ms); stage_acc[5].add(ft.save_ms);
          const std::string fname = "frame_" + cv::format("%05d", pi.idx) + ".png";
          std::cout << "[" << done << "/" << rtsp_frames << "] " << fname
                    << cv::format(" | read=%.1f inf=%.1f post=%.1f", ft.read_ms, ft.infer_ms, ft.post_ms);
          for (const auto& b : boxes)
            std::cout << " [" << class_name(labels, b.class_id)
                      << cv::format(" %.2f]", b.score);
          std::cout << "\n";
        }
        fast_encode_q.close();
      });

      // ── Thread D: x264 encode + UDP stream ───────────────────────────────
      // Pops rendered frames from fast_encode_q and calls rtp_writer.write().
      // Keeps x264 (~15-20ms) off Thread C so Thread C stays within the 15ms infer window.
      std::thread encode_th([&]() {
        cv::Mat frame;
        while (fast_encode_q.pop(frame)) {
          rtp_writer.write(frame);
        }
      });

      read_th.join();
      infer_th_fast.join();
      post_th.join();
      encode_th.join();
      processed = frames_done.load();

      if (processed > 1 && first_out_seen.load()) {
        double elapsed_ms = Ms(Clock::now() - t_first_out).count();
        double fps = (processed - 1) * 1000.0 / elapsed_ms;
        std::cout << cv::format("[pipeline] throughput: %.1f FPS  (%d frames, %.0f ms)\n",
                                fps, processed, elapsed_ms);
        std::cout << cv::format("[pipeline] MLA ceil:   %.1f FPS  (1000/%.1fms infer)\n",
                                1000.0 / stage_acc[2].mean(), stage_acc[2].mean());
      }

      cap.release();
      if (rtp_writer.isOpened()) rtp_writer.release();

    } else {
      // ── Folder mode ───────────────────────────────────────────────────────

      // ── Result processing: overlay, enqueue save, log ─────────────────────
      auto process_result = [&](InferOutput& out) {
        FrameTiming& ft = out.ft;
        draw_timing_overlay(out.bgr, ft);
        const fs::path out_path =
            output_dir / (images[static_cast<size_t>(out.idx)].stem().string() + ".png");
        { auto _ts = Clock::now();
          std::lock_guard<std::mutex> lk(save_mu);
          // Move the Mat — nothing uses out.bgr after draw_timing_overlay above.
          save_q.push({std::move(out.bgr), out_path});
          save_cv_var.notify_one();
          ft.save_ms = Ms(Clock::now() - _ts).count(); }
        ++processed;
        stage_acc[0].add(ft.read_ms);  stage_acc[1].add(ft.preproc_ms);
        stage_acc[2].add(ft.infer_ms); stage_acc[3].add(ft.post_ms);
        stage_acc[4].add(ft.render_ms); stage_acc[5].add(ft.save_ms);
        std::cout << "[" << processed << "/" << images.size() << "] "
                  << images[static_cast<size_t>(out.idx)].filename()
                  << " -> " << out_path.filename()
                  << " (" << out.boxes.size() << " det)"
                  << cv::format(" | read=%.1f pre=%.1f inf=%.1f post=%.1f rndr=%.1f save=%.1f | total=%.1fms\n",
                                ft.read_ms, ft.preproc_ms, ft.infer_ms,
                                ft.post_ms, ft.render_ms, ft.save_ms, ft.total());
      };

      // ── Read loop: push images to infer thread, drain ready results ────────
      for (size_t i = 0; i < images.size(); ++i) {
        cv::Mat bgr; double read_ms;
        { auto _tr = Clock::now(); bgr = cv::imread(images[i].string(), cv::IMREAD_COLOR); read_ms = Ms(Clock::now() - _tr).count(); }
        if (bgr.empty()) { std::cerr << "Skipping unreadable: " << images[i].filename() << "\n"; continue; }
        infer_q.push({std::move(bgr), read_ms, static_cast<int>(i)});
        { InferOutput out; while (result_q.try_pop(out)) process_result(out); }
      }
      infer_q.close();
      infer_th.join();
      { InferOutput out; while (result_q.pop(out)) process_result(out); }
    }

    // ── Drain save queue (only when save thread was started) ─────────────
    if (save_to_disk) {
      { std::lock_guard<std::mutex> lk(save_mu); save_stop = true; }
      save_cv_var.notify_all();
      save_thread.join();
    }

    if (processed > 0) {
      print_timing_summary(stage_acc, processed);
    }

    runner.close();

    std::cout << "Done: " << processed << " frames processed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 4;
  }
}
