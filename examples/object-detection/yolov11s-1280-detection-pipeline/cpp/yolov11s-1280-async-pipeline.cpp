/**
 * yolov11s-1280-async-pipeline.cpp
 *
 * RTSP detection pipeline using the SDK's Graph/Run API with VideoSender.
 *
 *   decode : SDK RtspDecodedInput node group (SiMa HW H.264)
 *   video  : SDK VideoSender node group (SiMa HW H.264 encode + RTP UDP, no box burn-in)
 *   infer  : SDK Model node with async push/pull (queue_depth=3, KeepLatest, ZeroCopy)
 *   post   : custom decode_yolov11_postsurgery (6-tensor output, NMS)
 *   meta   : MetadataSender → insight UDP (browser renders box overlay)
 *
 * Graph topology:
 *
 *   RtspDecodedInput → Branch → VideoSender  (clean video, no box overlay)
 *                            → Model → Output("detections")  (tensors)
 *
 * Main loop pulls "detections", runs custom post-processing, sends metadata.
 * No manual GStreamer, no manual decode/encode threads.
 *
 * Usage:
 *   yolov11s-1280-async-pipeline <model.tar.gz> <labels.txt>
 *     --rtsp <URL> [--insight <host>] [--frames N] [--fps N]
 *     [--classes <csv>] [--video-port N] [--meta-port N] [--save]
 */

#include "neat.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "support/object_detection/obj_detection_utils.h"
#include <nodes/io/MetadataSender.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#ifdef __ARM_NEON
#  include <arm_neon.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
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

// ── Timing ───────────────────────────────────────────────────────────────────
struct TimingAcc {
  double sum{0}, mn{1e18}, mx{-1.0}; int n{0};
  void add(double v) { sum += v; mn = std::min(mn,v); mx = std::max(mx,v); ++n; }
  double mean() const { return n > 0 ? sum/n : 0.0; }
};

// ── Tensor decode utilities ───────────────────────────────────────────────────
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
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1, cv::LINE_AA);
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
// Converts NV12 tensor to BGR (via copy_dense_bytes_tight + cv::cvtColor),
// draws boxes, saves every 100 frames.
static void save_tensor_frame(bool use_save, const simaai::neat::Tensor& nv12_t,
                               const std::vector<objdet::Box>& boxes,
                               const std::vector<std::string>& labels, int idx) {
  if (!use_save || idx == 0 || idx % 100 != 0) return;
  // Infer NV12 dims from tensor planes (Y plane shape = [H, W])
  if (nv12_t.planes.size() < 2) { std::cerr << "[save] unexpected tensor planes\n"; return; }
  const int h = static_cast<int>(nv12_t.planes[0].shape[0]);
  const int w = static_cast<int>(nv12_t.planes[0].shape[1]);
  const auto bytes = nv12_t.copy_dense_bytes_tight();
  if ((int)bytes.size() < w * h * 3 / 2) { std::cerr << "[save] tensor too small\n"; return; }
  cv::Mat nv12(h * 3 / 2, w, CV_8UC1,
               const_cast<uint8_t*>(bytes.data()));
  cv::Mat bgr;
  cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
  draw_boxes(bgr, boxes, labels);
  char path[128];
  std::snprintf(path, sizeof(path), "/tmp/save_box_%06d.jpg", idx);
  cv::imwrite(path, bgr, {cv::IMWRITE_JPEG_QUALITY, 90});
  std::cout << "[save] " << path << "\n";
}

// ── Metadata JSON builder ─────────────────────────────────────────────────────
static std::string build_metadata_json(const std::vector<objdet::Box>& boxes,
                                        const std::vector<std::string>& labels,
                                        int frame_w, int frame_h) {
  std::ostringstream j;
  j << "[";
  for (size_t i = 0; i < boxes.size(); ++i) {
    const auto& b = boxes[i];
    int x1 = std::max(0, (int)b.x1);
    int y1 = std::max(0, (int)b.y1);
    int w  = std::max(0, (int)(b.x2 - b.x1));
    int h  = std::max(0, (int)(b.y2 - b.y1));
    w = std::min(w, frame_w - x1);
    h = std::min(h, frame_h - y1);
    std::string lbl = (b.class_id >= 0 && (size_t)b.class_id < labels.size())
                      ? labels[b.class_id] : std::to_string(b.class_id);
    if (i > 0) j << ",";
    // Insight's object-detection overlay expects "bbox":[x,y,w,h] in absolute
    // frame pixels, NOT separate x/y/w/h fields. See neat-insight
    // tools/metadata_test.py generate_object_detection().
    j << "{\"id\":\"obj_" << (i+1) << "\""
      << ",\"label\":\"" << lbl << "\""
      << ",\"confidence\":" << cv::format("%.3f", b.score)
      << ",\"bbox\":[" << x1 << "," << y1 << "," << w << "," << h << "]}";
  }
  j << "]";
  return "{\"objects\":" + j.str() + "}";
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  ::setenv("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY", "1", 0);
  ::setenv("GST_PLUGIN_PATH_1_0",
           "/usr/lib/aarch64-linux-gnu/neat/gst-plugins", 0);

  // ── Argument parsing ──────────────────────────────────────────────────────
  std::string rtsp_url, insight_host, classes_arg;
  int  rtsp_frames = 99999, video_port = 9000, meta_port = 9100;
  int  stream_fps  = 30;
  bool use_save    = false;
  std::vector<std::string> pos_args;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if      (a == "--rtsp"       && i+1 < argc) rtsp_url     = argv[++i];
    else if (a == "--frames"     && i+1 < argc) rtsp_frames  = std::stoi(argv[++i]);
    else if (a == "--insight"    && i+1 < argc) insight_host = argv[++i];
    else if (a == "--video-port" && i+1 < argc) video_port   = std::stoi(argv[++i]);
    else if (a == "--meta-port"  && i+1 < argc) meta_port    = std::stoi(argv[++i]);
    else if (a == "--classes"    && i+1 < argc) classes_arg  = argv[++i];
    else if (a == "--fps"        && i+1 < argc) stream_fps   = std::stoi(argv[++i]);
    else if (a == "--save")                     use_save     = true;
    else pos_args.push_back(a);
  }
  if (pos_args.size() < 2 || rtsp_url.empty()) {
    std::cerr << "Usage: " << argv[0]
              << " <model> <labels> --rtsp <URL>\n"
              << "  [--insight <host>] [--frames N] [--fps N]\n"
              << "  [--classes <csv>] [--video-port N] [--meta-port N] [--save]\n";
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

    // Stream dims and FPS: use kInputW/kInputH constants + --fps arg.
    // RtspDecodedInput will auto-negotiate caps from the actual stream.
    const int frame_w    = kInputW;
    const int frame_h    = kInputH;
    const int output_fps = stream_fps;
    std::cout << "[stream] " << frame_w << "x" << frame_h << "@" << output_fps << " fps\n";

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

    // ── Build Graph ───────────────────────────────────────────────────────────
    // Source: SDK RtspDecodedInput (SiMa HW H.264 decoder)
    simaai::neat::nodes::groups::RtspDecodedInputOptions src_opt;
    src_opt.url                  = rtsp_url;
    src_opt.latency_ms           = 200;
    src_opt.tcp                  = true;
    src_opt.payload_type         = 96;
    src_opt.insert_queue         = true;
    src_opt.out_format           = "NV12";
    src_opt.decoder_name         = "decoder";
    src_opt.decoder_raw_output   = true;
    src_opt.auto_caps_from_stream = true;
    src_opt.fallback_h264_width  = frame_w;
    src_opt.fallback_h264_height = frame_h;
    src_opt.fallback_h264_fps    = output_fps;
    auto source = simaai::neat::nodes::groups::RtspDecodedInput(src_opt);

    // Branch: video (→ VideoSender) + model (→ Model)
    std::vector<std::string> branch_outputs = {"video", "model"};
    if (use_save) branch_outputs.push_back("debug_frame");
    auto branch = simaai::neat::graphs::Branch("source", branch_outputs);

    // VideoSender: SDK HW encode + RTP UDP (clean video, no box overlay)
    auto vid_opt = simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(
        frame_w, frame_h, output_fps);
    vid_opt.host             = insight_host.empty() ? "127.0.0.1" : insight_host;
    vid_opt.channel          = 0;
    vid_opt.video_port_base  = video_port;
    vid_opt.encoder.bitrate_kbps = 4000;
    simaai::neat::Graph video_graph("video");
    video_graph.connect(simaai::neat::nodes::Input("video"),
                        simaai::neat::nodes::groups::VideoSender(vid_opt));

    // Model graph
    simaai::neat::Graph model_graph("model");
    model_graph.connect(simaai::neat::nodes::Input("model"), model);

    // Detections output
    simaai::neat::Graph det_graph("detections");
    det_graph.add(
        simaai::neat::nodes::Output("detections", simaai::neat::OutputOptions::EveryFrame(4)));

    // Wire the graph
    simaai::neat::Graph graph;
    graph.connect(source, branch);
    graph.connect(branch, video_graph);
    graph.connect(branch, model_graph);
    graph.connect(model_graph, det_graph);

    // Optional debug_frame output for --save (combines raw NV12 + detections)
    std::string pull_name = "detections";
    if (use_save) {
      simaai::neat::Graph dbg_graph("debug_frame");
      dbg_graph.add(simaai::neat::nodes::Output(
          "debug_frame", simaai::neat::OutputOptions::EveryFrame(4)));
      auto dbg_join = simaai::neat::graphs::Combine(
          {"debug_frame", "detections"}, "debug_output",
          simaai::neat::CombinePolicy::ByPts);
      graph.connect(branch, dbg_graph);
      graph.connect(dbg_graph, dbg_join);
      graph.connect(det_graph, dbg_join);
      pull_name = "debug_output";
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
        std::cout << "[insight] video    -> udp://" << insight_host << ":" << vid_opt.video_port() << "\n"
                  << "[insight] metadata -> udp://" << insight_host << ":" << meta_sender->metadata_port() << "\n";
      else {
        std::cerr << "[insight] warn: MetadataSender: " << meta_err << "\n";
        meta_sender.reset();
      }
    }

    // ── Build Run ─────────────────────────────────────────────────────────────
    simaai::neat::RunOptions run_opt;
    run_opt.preset                              = simaai::neat::RunPreset::Realtime;
    run_opt.queue_depth                         = 3;
    run_opt.overflow_policy                     = simaai::neat::OverflowPolicy::KeepLatest;
    run_opt.output_memory                       = simaai::neat::OutputMemory::ZeroCopy;
    run_opt.advanced.prepare_output_cpu_visible = true;

    auto run = graph.build(run_opt);
    std::cout << "[BUILD] " << model_path << " ready — "
              << frame_w << "x" << frame_h
              << "@" << output_fps << " fps | VideoSender | ZeroCopy\n";

    // ── MeasureScope ──────────────────────────────────────────────────────────
    simaai::neat::MeasureOptions mopt;
    mopt.warmup_ms              = 2000;
    mopt.include_plugin_latency = true;
    auto mscope = run.start_measurement(mopt);

    // ── Main loop ─────────────────────────────────────────────────────────────
    int processed = 0;
    Clock::time_point t_first_out{};
    bool first_seen = false;
    TimingAcc pull_acc, post_acc, meta_acc;

    while (processed < rtsp_frames) {
      simaai::neat::Sample sample;
      simaai::neat::PullError pull_err;
      auto t0 = Clock::now();
      auto status = run.pull(pull_name, 20000, sample, &pull_err);
      double pull_ms = Ms(Clock::now() - t0).count();

      if (status == simaai::neat::PullStatus::Timeout) {
        std::cerr << "[warn] pull timeout\n";
        continue;
      }
      if (status == simaai::neat::PullStatus::Closed) break;
      if (status != simaai::neat::PullStatus::Ok)
        throw std::runtime_error("pull failed: " + pull_err.message);

      if (!first_seen) { first_seen = true; t_first_out = Clock::now(); }

      // Extract detection tensors
      simaai::neat::TensorList tensors;
      simaai::neat::Tensor     frame_tensor;  // only populated when --save + Bundle
      if (use_save && sample.kind == simaai::neat::SampleKind::Bundle) {
        for (const auto& field : sample.fields) {
          auto ft = simaai::neat::tensors_from_sample(field, false);
          if (ft.size() == 6 && tensors.empty()) {
            tensors = std::move(ft);
          } else if (ft.size() == 1 && !frame_tensor.storage) {
            frame_tensor = ft.front();
          }
        }
        if (tensors.empty())
          tensors = simaai::neat::tensors_from_sample(sample, false);
      } else {
        tensors = simaai::neat::tensors_from_sample(sample, false);
      }

      // Post-processing: decode tensors → NMS
      auto t1 = Clock::now();
      std::vector<objdet::Box> boxes;
      if (!tensors.empty()) {
        try {
          auto boxes_720 = decode_yolov11_postsurgery(tensors, kMinScore);
          auto all = scale_boxes(boxes_720, frame_w, frame_h);
          if (allowed_classes.empty()) {
            boxes = std::move(all);
          } else {
            for (auto& b : all)
              if (allowed_classes.count(b.class_id)) boxes.push_back(b);
          }
        } catch (const std::exception& e) {
          std::cerr << "[post] failed frame " << processed << ": " << e.what() << "\n";
        }
      }
      double post_ms = Ms(Clock::now() - t1).count();

      // Send metadata to insight (browser renders box overlay on clean video)
      if (meta_sender) {
        auto t2 = Clock::now();
        const std::string data_json = build_metadata_json(boxes, labels, frame_w, frame_h);
        const int64_t ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        const int64_t fid = sample.frame_id >= 0 ? sample.frame_id : processed;
        std::string merr;
        if (!meta_sender->send_metadata("object-detection", data_json, ts_ms,
                                        std::to_string(fid), &merr))
          std::cerr << "[meta] send failed: " << merr << "\n";
        meta_acc.add(Ms(Clock::now() - t2).count());
      }

      // Save debug frame (only when --save and frame tensor available)
      if (use_save && frame_tensor.storage)
        save_tensor_frame(true, frame_tensor, boxes, labels, processed);

      ++processed;
      if (processed > 0) { pull_acc.add(pull_ms); post_acc.add(post_ms); }

      if (processed % 100 == 0) {
        std::cout << "[" << processed << "/" << rtsp_frames << "]"
                  << cv::format(" pull=%.1fms post=%.1fms boxes=%zu\n",
                                pull_ms, post_ms, boxes.size());
      }
    }

    // ── SDK report ────────────────────────────────────────────────────────────
    auto report = mscope.stop();
    std::cout << "\n── SDK MeasureReport ─────────────────────────────────────\n"
              << report.to_text() << "\n";

    // ── Pipeline stats ────────────────────────────────────────────────────────
    if (processed > 1 && first_seen) {
      double elapsed = Ms(Clock::now() - t_first_out).count();
      double fps     = (processed - 1) * 1000.0 / elapsed;
      std::cout << cv::format("\n[pipeline] throughput: %.1f FPS  (%d frames, %.0f ms)\n",
                              fps, processed, elapsed);
      std::cout << cv::format("  pull  mean=%.2f ms  min=%.2f  max=%.2f\n",
                              pull_acc.mean(), pull_acc.mn, pull_acc.mx);
      std::cout << cv::format("  post  mean=%.2f ms  min=%.2f  max=%.2f\n",
                              post_acc.mean(), post_acc.mn, post_acc.mx);
      if (meta_acc.n > 0)
        std::cout << cv::format("  meta  mean=%.2f ms  min=%.2f  max=%.2f\n",
                                meta_acc.mean(), meta_acc.mn, meta_acc.mx);
    }

    run.close();
    std::cout << "Done: " << processed << " frames\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 4;
  }
}
