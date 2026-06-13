// Copyright 2026 SiMa Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "neat.h"
#include "neat/models.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "support/object_detection/obj_detection_utils.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"

#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kStreamLimit = 16;
constexpr int kInitialFrameTimeoutMs = 30000;
constexpr int kDetectorResultTimeoutMs = 5000;
constexpr int kDecoderBuffers = 8;

volatile std::sig_atomic_t g_stop_requested = 0;

void request_stop(int) {
  g_stop_requested = 1;
}

struct AppConfig {
  std::string model_path;
  std::string decode_type = "yolo26";
  fs::path labels_path;
  std::vector<std::string> rtsp_urls;
  int workers = 1;
  int queue_depth = 16;
  int latency_ms = 100;
  bool tcp = true;
  double min_score = 0.55;
  double nms_iou = 0.60;
  int max_detections = 50;
  bool profile = false;
  int warmup_frames = 30;
  std::string insight_host = "127.0.0.1";
  int video_port_base = 9000;
  int metadata_port_base = 9100;
  bool video_enabled = true;
  fs::path save_dir;
  int save_every = 0;
};

simaai::neat::BoxDecodeType parse_box_decode_type(const std::string& token) {
  if (token == "yolo26" || token == "yolov26")
    return simaai::neat::BoxDecodeType::YoloV26;
  if (token == "yolov8")
    return simaai::neat::BoxDecodeType::YoloV8;
  throw std::runtime_error("model.decode_type must be one of: yolo26, yolov8");
}

struct CliOptions {
  fs::path config_path;
  bool validate_config_only = false;
};

struct StreamProfile {
  bool enabled = false;
  int stream_index = 0;
  int interval = 100;
  int frames = 0;
  int boxes = 0;
  double start_ms = 0.0;
  double parse_ms = 0.0;
  double metadata_send_ms = 0.0;

  void add(double parse, double metadata_send, int box_count) {
    if (!enabled)
      return;
    if (frames == 0)
      start_ms = sima_examples::time_ms();
    ++frames;
    boxes += box_count;
    parse_ms += parse;
    metadata_send_ms += metadata_send;
    if (frames >= interval)
      flush();
  }

  void flush() {
    if (!enabled || frames == 0)
      return;
    const double elapsed = sima_examples::time_ms() - start_ms;
    const double metadata_fps =
        elapsed > 0.0 ? static_cast<double>(frames) * 1000.0 / elapsed : 0.0;
    const auto avg = [this](double value) { return value / static_cast<double>(frames); };
    std::cout << "[profile stream=" << stream_index << "] frames=" << frames
              << " metadata_fps=" << metadata_fps << " avg_parse_ms=" << avg(parse_ms)
              << " avg_metadata_send_ms=" << avg(metadata_send_ms)
              << " avg_boxes=" << static_cast<double>(boxes) / static_cast<double>(frames) << "\n";
    frames = 0;
    boxes = 0;
    start_ms = 0.0;
    parse_ms = 0.0;
    metadata_send_ms = 0.0;
  }
};
struct AggregateProfile {
  bool enabled = false;
  int stream_count = 0;
  std::uint64_t interval_frames = 400;
  std::uint64_t frames = 0;
  double start_ms = 0.0;
  std::mutex mu;

  void add() {
    if (!enabled)
      return;
    std::lock_guard<std::mutex> lock(mu);
    if (frames == 0)
      start_ms = sima_examples::time_ms();
    ++frames;
    if (frames >= interval_frames)
      flush_locked();
  }

  void flush() {
    if (!enabled)
      return;
    std::lock_guard<std::mutex> lock(mu);
    flush_locked();
  }

private:
  void flush_locked() {
    if (frames == 0)
      return;
    const double elapsed_s = std::max(0.001, (sima_examples::time_ms() - start_ms) / 1000.0);
    const double aggregate_fps = static_cast<double>(frames) / elapsed_s;
    std::cout << "[profile aggregate] frames=" << frames << " elapsed_s=" << elapsed_s
              << " aggregate_fps=" << aggregate_fps << " avg_per_stream_fps="
              << (stream_count > 0 ? aggregate_fps / static_cast<double>(stream_count) : 0.0)
              << "\n";
    frames = 0;
    start_ms = 0.0;
  }
};

struct SourceRuntime {
  int index = 0;
  std::string url;
  std::unique_ptr<simaai::neat::MetadataSender> metadata_sender;
  std::vector<std::string> labels;
  simaai::neat::nodes::groups::RtspDecodedInputOptions source_options;
  StreamProfile profile;
  int frame_w = 0;
  int frame_h = 0;
  int source_fps = 0;
  int video_port = 0;
  int processed = 0;
};
struct AppRuntime {
  std::unique_ptr<simaai::neat::Model> model;
  simaai::neat::Graph graph;
  simaai::neat::Run run;
  std::vector<SourceRuntime> sources;
};
CliOptions parse_args(int argc, char** argv) {
  CliOptions options;
  options.config_path = sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--config requires a path");
      }
      options.config_path = argv[++i];
    } else if (arg == "--validate-config-only") {
      options.validate_config_only = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [--config <path>] [--validate-config-only]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return options;
}

std::string strip_inline_comment(const std::string& line) {
  bool in_single = false;
  bool in_double = false;
  std::string out;
  out.reserve(line.size());
  for (char c : line) {
    if (c == '\'' && !in_double) {
      in_single = !in_single;
    } else if (c == '"' && !in_single) {
      in_double = !in_double;
    } else if (c == '#' && !in_single && !in_double) {
      break;
    }
    out.push_back(c);
  }
  return out;
}

std::string unquote(std::string value) {
  value = sima_examples::trim_copy(value);
  if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                            (value.front() == '\'' && value.back() == '\''))) {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

int leading_indent(const std::string& line) {
  int indent = 0;
  while (indent < static_cast<int>(line.size()) &&
         (line[static_cast<std::size_t>(indent)] == ' ' ||
          line[static_cast<std::size_t>(indent)] == '\t')) {
    ++indent;
  }
  return indent;
}

std::vector<std::string> parse_streams(const fs::path& config_path) {
  std::ifstream input(config_path);
  if (!input.is_open()) {
    throw std::runtime_error("failed to open config file: " + config_path.string());
  }

  std::vector<std::string> streams;
  bool in_streams = false;
  int streams_indent = -1;
  std::string raw_line;
  while (std::getline(input, raw_line)) {
    const std::string line_without_comment = strip_inline_comment(raw_line);
    if (sima_examples::trim_copy(line_without_comment).empty()) {
      continue;
    }

    const int indent = leading_indent(line_without_comment);
    const std::string line = sima_examples::trim_copy(line_without_comment);

    if (in_streams && indent <= streams_indent && line.rfind("- ", 0) != 0) {
      in_streams = false;
    }
    if (!in_streams && line == "streams:") {
      in_streams = true;
      streams_indent = indent;
      continue;
    }
    if (!in_streams) {
      continue;
    }

    if (line.rfind("- ", 0) == 0) {
      const std::string url = unquote(line.substr(2));
      if (url.empty()) {
        throw std::runtime_error("streams entries must be non-empty strings");
      }
      streams.push_back(url);
    }
  }
  if (streams.empty()) {
    throw std::runtime_error("streams must be a non-empty list");
  }
  return streams;
}

void validate_config(const AppConfig& cfg) {
  sima_examples::require(!cfg.model_path.empty(), "model.path must be set");
  (void)parse_box_decode_type(cfg.decode_type);
  sima_examples::require(!cfg.labels_path.empty(), "model.labels must be set");
  sima_examples::require(!cfg.rtsp_urls.empty(), "streams must be set");
  sima_examples::require(cfg.rtsp_urls.size() <= kStreamLimit,
                         "this example supports up to 16 streams");
  sima_examples::require(
      cfg.workers == 1,
      "graph-linked 16-stream detector uses one shared worker; set inference.workers to 1");
  sima_examples::require(cfg.queue_depth > 0, "inference.queue_depth must be > 0");
  sima_examples::require(cfg.queue_depth <= 32, "inference.queue_depth must be <= 32");
  sima_examples::require(!cfg.insight_host.empty(), "output.insight.host must be set");
  sima_examples::require(cfg.latency_ms >= 0, "input.latency_ms must be >= 0");
  sima_examples::require(cfg.min_score >= 0.0 && cfg.min_score <= 1.0,
                         "inference.min_score must be between 0 and 1");
  sima_examples::require(cfg.nms_iou >= 0.0 && cfg.nms_iou <= 1.0,
                         "inference.nms_iou must be between 0 and 1");
  sima_examples::require(cfg.max_detections > 0, "inference.max_detections must be > 0");
  sima_examples::require(cfg.warmup_frames >= 0, "runtime.warmup_frames must be >= 0");
  sima_examples::require(cfg.video_port_base > 0, "output.insight.video_port_base must be > 0");
  sima_examples::require(cfg.metadata_port_base > 0,
                         "output.insight.metadata_port_base must be > 0");
  sima_examples::require(cfg.save_every >= 0, "output.save_every must be >= 0");
}

AppConfig load_app_config(const fs::path& config_path) {
  const auto raw = sima_examples::ScalarConfig::load(config_path);
  sima_examples::require(!raw.string_value("inference.fps").has_value(),
                         "inference.fps is not supported; set stream FPS at the RTSP source");
  sima_examples::require(
      !raw.string_value("inference.target_fps").has_value(),
      "inference.target_fps is not supported; set stream FPS at the RTSP source");
  const auto default_labels =
      fs::path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR).parent_path() / "common" / "coco_label.txt";

  AppConfig cfg;
  cfg.model_path = raw.string_or("model.path", "");
  cfg.decode_type = raw.string_or("model.decode_type", "yolo26");
  cfg.labels_path = raw.string_or("model.labels", default_labels.string());
  cfg.rtsp_urls = parse_streams(config_path);
  cfg.tcp = raw.bool_or("input.tcp", true);
  cfg.latency_ms = raw.int_or("input.latency_ms", 100);
  cfg.workers = raw.int_or("inference.workers", 1);
  cfg.queue_depth = raw.int_or("inference.queue_depth", 16);
  if (!raw.string_value("inference.queue_depth").has_value()) {
    cfg.queue_depth =
        raw.int_or("inference.inflight", raw.int_or("inference.detector_queue_depth", 16));
  }
  cfg.min_score = raw.double_or("inference.min_score", 0.55);
  cfg.nms_iou = raw.double_or("inference.nms_iou", 0.60);
  cfg.max_detections = raw.int_or("inference.max_detections", 50);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.warmup_frames = raw.int_or("runtime.warmup_frames", 30);
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port_base = raw.int_or("output.insight.video_port_base", 9000);
  cfg.metadata_port_base = raw.int_or("output.insight.metadata_port_base", 9100);
  cfg.video_enabled = raw.bool_or("output.video_enabled", true);
  cfg.save_dir = raw.string_or("output.debug_dir", "");
  cfg.save_every = raw.int_or("output.save_every", 0);
  validate_config(cfg);
  return cfg;
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

bool extract_bbox_payload(const simaai::neat::Sample& sample, std::vector<std::uint8_t>& payload,
                          std::string& err) {
  if (sample.kind == simaai::neat::SampleKind::Bundle) {
    for (const auto& field : sample.fields) {
      if (extract_bbox_payload(field, payload, err)) {
        return true;
      }
    }
    err = "bundle missing BBOX field";
    return false;
  }
  if (sample.kind == simaai::neat::SampleKind::TensorSet && !sample.tensors.empty()) {
    simaai::neat::Sample tensor_sample = sample;
    tensor_sample.kind = simaai::neat::SampleKind::Tensor;
    tensor_sample.tensor = sample.tensors.front();
    tensor_sample.tensors.clear();
    return objdet::extract_bbox_payload(tensor_sample, payload, err);
  }
  return objdet::extract_bbox_payload(sample, payload, err);
}

std::vector<sima_examples::MetadataBox> build_metadata_boxes(const std::vector<objdet::Box>& boxes,
                                                             const std::vector<std::string>& labels,
                                                             int frame_w, int frame_h) {
  std::vector<sima_examples::MetadataBox> metadata_boxes;
  metadata_boxes.reserve(boxes.size());
  int object_index = 1;
  for (const auto& box : boxes) {
    int x1 = std::max(0, static_cast<int>(box.x1));
    int y1 = std::max(0, static_cast<int>(box.y1));
    int w = std::max(0, static_cast<int>(box.x2 - box.x1));
    int h = std::max(0, static_cast<int>(box.y2 - box.y1));
    if (x1 + w > frame_w)
      w = frame_w - x1;
    if (y1 + h > frame_h)
      h = frame_h - y1;

    sima_examples::MetadataBox obj;
    obj.id = "obj_" + std::to_string(object_index++);
    obj.label = (box.class_id >= 0 && box.class_id < static_cast<int>(labels.size()))
                    ? labels[box.class_id]
                    : "unknown";
    obj.confidence = box.score;
    obj.x = static_cast<float>(x1);
    obj.y = static_cast<float>(y1);
    obj.w = static_cast<float>(std::max(0, w));
    obj.h = static_cast<float>(std::max(0, h));
    metadata_boxes.push_back(obj);
  }
  return metadata_boxes;
}

int64_t fallback_timestamp_ms() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

bool env_bool(const char* key, bool fallback = false) {
  const char* value = std::getenv(key);
  if (!value || !*value)
    return fallback;
  std::string v(value);
  std::transform(v.begin(), v.end(), v.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return v == "1" || v == "true" || v == "yes" || v == "on";
}

bool app_verbose() {
  return env_bool("APP16_VERBOSE", false);
}

bool app_print_backend() {
  return app_verbose() || env_bool("APP16_PRINT_BACKEND", false);
}

bool app_gst_profile() {
  return env_bool("APP16_GST_PROFILE", false);
}

std::string stream_id_for(int stream_index) {
  return "stream" + std::to_string(stream_index);
}

int stream_index_from_detection(const simaai::neat::Sample& sample, int stream_count) {
  const std::string prefix = "stream";
  if (sample.stream_id.rfind(prefix, 0) != 0) {
    if (stream_count == 1)
      return 0;
    throw std::runtime_error("detection sample missing stream id: " + sample.stream_id);
  }
  const std::string suffix = sample.stream_id.substr(prefix.size());
  if (suffix.empty() ||
      !std::all_of(suffix.begin(), suffix.end(), [](unsigned char c) { return std::isdigit(c); })) {
    throw std::runtime_error("invalid detection stream id: " + sample.stream_id);
  }
  const int index = std::stoi(suffix);
  if (index < 0 || index >= stream_count) {
    throw std::runtime_error("detection stream id out of range: " + sample.stream_id);
  }
  return index;
}

simaai::neat::RunOptions realtime_options(
    int queue_depth = 3,
    simaai::neat::OverflowPolicy overflow_policy = simaai::neat::OverflowPolicy::KeepLatest) {
  simaai::neat::RunOptions run_options;
  run_options.preset = simaai::neat::RunPreset::Realtime;
  run_options.queue_depth = queue_depth;
  run_options.overflow_policy = overflow_policy;
  run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  return run_options;
}

simaai::neat::nodes::groups::RtspDecodedInputOptions
make_source_options(const AppConfig& cfg, const std::string& url, int& fps_out, int& width_out,
                    int& height_out) {
  sima_examples::RtspStreamInfo probe;
  sima_examples::RtspProbeOptions probe_options;
  probe_options.payload_type = 96;
  probe_options.latency_ms = cfg.latency_ms;
  probe_options.rtsp_tcp = cfg.tcp;
  probe_options.debug = cfg.profile;
  (void)sima_examples::probe_rtsp_stream_info(url, probe_options, probe);

  simaai::neat::nodes::groups::RtspDecodedInputOptions opt;
  opt.url = url;
  opt.latency_ms = cfg.latency_ms;
  opt.tcp = cfg.tcp;
  opt.payload_type = 96;
  opt.insert_queue = true;
  opt.out_format = "NV12";
  opt.decoder_name = "decoder";
  opt.decoder_raw_output = true;
  opt.decoder_next_element = "CVU";
  opt.auto_caps_from_stream = true;
  if (probe.width > 0 && probe.height > 0) {
    opt.fallback_h264_width = probe.width;
    opt.fallback_h264_height = probe.height;
    opt.output_caps.width = probe.width;
    opt.output_caps.height = probe.height;
    width_out = probe.width;
    height_out = probe.height;
  }
  if (probe.fps > 0) {
    opt.fallback_h264_fps = probe.fps;
    opt.output_caps.fps = probe.fps;
    fps_out = probe.fps;
  }
  opt.output_caps.enable = true;
  opt.output_caps.format = simaai::neat::FormatTag::NV12;
  opt.output_caps.memory = simaai::neat::CapsMemory::Any;
  return opt;
}

simaai::neat::Graph
make_rtsp_decoded_input(const simaai::neat::nodes::groups::RtspDecodedInputOptions& opt,
                        int decoder_buffers) {
  const bool use_auto_caps = opt.auto_caps_from_stream &&
                             (opt.h264_fps <= 0 || opt.h264_width <= 0 || opt.h264_height <= 0);
  const int dec_w = (opt.h264_width > 0) ? opt.h264_width : opt.fallback_h264_width;
  const int dec_h = (opt.h264_height > 0) ? opt.h264_height : opt.fallback_h264_height;
  const int dec_fps = (opt.h264_fps > 0) ? opt.h264_fps : opt.fallback_h264_fps;

  simaai::neat::Graph graph;
  graph.add(simaai::neat::nodes::RTSPInput(opt.url, opt.latency_ms, opt.tcp));
  if (opt.insert_queue)
    graph.add(simaai::neat::nodes::Queue());
  graph.add(simaai::neat::nodes::H264Depacketize(opt.payload_type, opt.h264_parse_config_interval,
                                                 opt.h264_fps, opt.h264_width, opt.h264_height,
                                                 /*enforce_h264_caps=*/!use_auto_caps));
  if (opt.insert_queue)
    graph.add(simaai::neat::nodes::Queue());
  if (use_auto_caps) {
    graph.add(simaai::neat::nodes::H264CapsFixup(opt.fallback_h264_fps, opt.fallback_h264_width,
                                                 opt.fallback_h264_height));
  }
  graph.add(simaai::neat::nodes::H264Decode(opt.sima_allocator_type, "NV12", opt.decoder_name,
                                            opt.decoder_raw_output, opt.decoder_next_element, dec_w,
                                            dec_h, dec_fps, decoder_buffers));
  if (opt.output_caps.enable) {
    graph.add(simaai::neat::nodes::CapsRaw("NV12", opt.output_caps.width, opt.output_caps.height,
                                           opt.output_caps.fps, opt.output_caps.memory));
  }
  return graph;
}

std::unique_ptr<simaai::neat::Model> make_model(const AppConfig& cfg) {
  simaai::neat::Model::Options model_opt;
  if (!app_verbose()) {
    model_opt.verbose = simaai::neat::VerboseOptions::quiet();
  }
  model_opt.preprocess.kind = simaai::neat::InputKind::Image;
  model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
  model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::NV12;
  model_opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
  model_opt.decode_type = parse_box_decode_type(cfg.decode_type);
  model_opt.score_threshold = cfg.min_score;
  model_opt.nms_iou_threshold = cfg.nms_iou;
  model_opt.top_k = cfg.max_detections;
  return std::make_unique<simaai::neat::Model>(cfg.model_path, model_opt);
}

simaai::neat::nodes::groups::VideoSenderOptions make_video_options(const AppConfig& cfg,
                                                                   const SourceRuntime& source) {
  auto video_options = simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(
      source.frame_w, source.frame_h, source.source_fps);
  video_options.host = cfg.insight_host;
  video_options.channel = source.index;
  video_options.video_port_base = cfg.video_port_base;
  video_options.encoder.bitrate_kbps = 1000;
  return video_options;
}

SourceRuntime make_source_runtime(const AppConfig& cfg, int stream_index,
                                  const std::vector<std::string>& labels) {
  SourceRuntime source;
  source.index = stream_index;
  source.url = cfg.rtsp_urls[static_cast<std::size_t>(stream_index)];
  source.source_options =
      make_source_options(cfg, source.url, source.source_fps, source.frame_w, source.frame_h);
  sima_examples::require(source.frame_w > 0 && source.frame_h > 0,
                         "failed to probe RTSP frame dimensions");
  sima_examples::require(source.source_fps > 0, "failed to probe RTSP frame rate");
  source.labels = labels;
  source.profile.enabled = cfg.profile;
  source.profile.stream_index = stream_index;

  simaai::neat::MetadataSenderOptions metadata_options;
  metadata_options.host = cfg.insight_host;
  metadata_options.channel = stream_index;
  metadata_options.metadata_port_base = cfg.metadata_port_base;
  std::string metadata_err;
  source.metadata_sender =
      std::make_unique<simaai::neat::MetadataSender>(metadata_options, &metadata_err);
  sima_examples::require(source.metadata_sender->ok(), metadata_err);
  return source;
}

simaai::neat::Graph build_detector_graph(const AppConfig& cfg,
                                         std::unique_ptr<simaai::neat::Model>& model) {
  model = make_model(cfg);

  simaai::neat::Graph input_graph;
  input_graph.add(simaai::neat::nodes::Input("detector_frame", model->input_appsrc_options(false)));

  simaai::neat::Graph model_graph = model->graph();

  simaai::neat::Graph detections_graph;
  detections_graph.add(simaai::neat::nodes::Output(
      "detections", simaai::neat::OutputOptions::EveryFrame(cfg.queue_depth)));

  simaai::neat::Graph graph;
  graph.connect(input_graph, model_graph);
  graph.connect(model_graph, detections_graph);
  return graph;
}

void connect_source_graph(AppRuntime& app, const AppConfig& cfg, SourceRuntime& source,
                          const simaai::neat::Graph& detector_graph) {
  auto rtsp = make_rtsp_decoded_input(source.source_options, kDecoderBuffers);
  std::vector<std::string> outputs = {"detector_frame"};
  if (cfg.video_enabled) {
    outputs.push_back("video");
  }
  auto branch = simaai::neat::graphs::Branch("source", outputs);

  app.graph.connect(rtsp, branch);

  simaai::neat::GraphLinkOptions detector_link;
  detector_link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
  detector_link.queue_depth = cfg.queue_depth;
  detector_link.stream_id = stream_id_for(source.index);
  app.graph.connect(branch, detector_graph, detector_link);

  if (cfg.video_enabled) {
    auto video_options = make_video_options(cfg, source);
    source.video_port = video_options.video_port();
    simaai::neat::Graph video_graph;
    video_graph.connect(simaai::neat::nodes::Input("video"),
                        simaai::neat::nodes::groups::VideoSender(video_options));
    app.graph.connect(branch, video_graph);
  }

  std::cout << "[stream " << source.index << "] rtsp=" << source.url << " stream=" << source.frame_w
            << "x" << source.frame_h << "@" << source.source_fps << " insight=" << cfg.insight_host
            << " video=";
  if (cfg.video_enabled) {
    std::cout << source.video_port;
  } else {
    std::cout << "disabled";
  }
  std::cout << " metadata=" << source.metadata_sender->metadata_port() << "\n";
}

void validate_worker_pool_geometry(const std::vector<SourceRuntime>& sources) {
  if (sources.empty()) {
    return;
  }
  const int frame_w = sources.front().frame_w;
  const int frame_h = sources.front().frame_h;
  for (const auto& source : sources) {
    if (source.frame_w != frame_w || source.frame_h != frame_h) {
      throw std::runtime_error("detector graph requires all streams to share frame size");
    }
  }
}

void send_metadata(SourceRuntime& source, const simaai::neat::Sample& frame,
                   const std::vector<objdet::Box>& boxes) {
  const auto metadata_boxes =
      build_metadata_boxes(boxes, source.labels, source.frame_w, source.frame_h);
  const std::string data_json = sima_examples::metadata_boxes_data_json("objects", metadata_boxes);
  const int64_t timestamp_ms = frame.pts_ns >= 0 ? frame.pts_ns / 1000000 : fallback_timestamp_ms();
  const int64_t frame_id = frame.frame_id >= 0 ? frame.frame_id : 0;
  std::string err;
  if (!source.metadata_sender->send_metadata("object-detection", data_json, timestamp_ms,
                                             std::to_string(frame_id), &err)) {
    std::cerr << "[warn] stream " << source.index << " metadata send failed: " << err << "\n";
  }
}

void complete_detection(SourceRuntime& source, const AppConfig& cfg,
                        AggregateProfile& aggregate_profile,
                        const simaai::neat::Sample& detections) {
  const double parse_start = sima_examples::time_ms();
  std::vector<std::uint8_t> payload;
  std::string err;
  if (!extract_bbox_payload(detections, payload, err)) {
    throw std::runtime_error("stream " + std::to_string(source.index) +
                             " bbox extract failed: " + err);
  }
  const auto boxes = objdet::parse_boxes_strict(payload, source.frame_w, source.frame_h,
                                                cfg.max_detections, false);
  const double parse_end = sima_examples::time_ms();

  ++source.processed;
  const bool warming_up = source.processed <= cfg.warmup_frames;
  if (!warming_up) {
    const double metadata_start = sima_examples::time_ms();
    send_metadata(source, detections, boxes);
    const double metadata_end = sima_examples::time_ms();
    source.profile.add(parse_end - parse_start, metadata_end - metadata_start,
                       static_cast<int>(boxes.size()));
    aggregate_profile.add();
  }
}

void pull_detections(AppRuntime& app, const AppConfig& cfg, AggregateProfile& aggregate_profile) {
  bool saw_detection = false;
  while (g_stop_requested == 0) {
    simaai::neat::Sample detections;
    simaai::neat::PullError err;
    const int timeout_ms = saw_detection ? kDetectorResultTimeoutMs : kInitialFrameTimeoutMs;
    const auto status = app.run.pull("detections", timeout_ms, detections, &err);
    if (status == simaai::neat::PullStatus::Closed) {
      break;
    }
    if (status == simaai::neat::PullStatus::Timeout) {
      throw std::runtime_error(saw_detection ? "timed out waiting for detections"
                                             : "timed out waiting for initial detection");
    }
    if (status == simaai::neat::PullStatus::Error) {
      throw std::runtime_error("failed to pull detections: " + err.message);
    }
    if (status != simaai::neat::PullStatus::Ok) {
      continue;
    }

    saw_detection = true;
    const int stream_index =
        stream_index_from_detection(detections, static_cast<int>(app.sources.size()));
    complete_detection(app.sources[static_cast<std::size_t>(stream_index)], cfg, aggregate_profile,
                       detections);
  }
}

void run_app(const AppConfig& cfg) {
  g_stop_requested = 0;
  auto previous_sigint = std::signal(SIGINT, request_stop);
  if (!app_print_backend() && !app_gst_profile()) {
    setenv("SIMA_PULL_TIMEOUT_DIAG", "0", 0);
  }
  if (app_gst_profile()) {
    setenv("SIMA_GST_ELEMENT_TIMINGS", "1", 0);
    setenv("SIMA_GST_FLOW_DEBUG", "1", 0);
    setenv("SIMA_GST_BOUNDARY_PROBES", "1", 0);
  }
  if (!cfg.save_dir.empty() && cfg.save_every > 0) {
    std::cout << "[warn] output.debug_dir is ignored in graph-linked realtime mode\n";
  }

  const auto labels = load_labels(cfg.labels_path);
  AggregateProfile aggregate_profile;
  aggregate_profile.enabled = cfg.profile;
  aggregate_profile.stream_count = static_cast<int>(cfg.rtsp_urls.size());

  AppRuntime app;
  app.sources.reserve(cfg.rtsp_urls.size());
  auto detector_graph = build_detector_graph(cfg, app.model);
  for (std::size_t index = 0; index < cfg.rtsp_urls.size(); ++index) {
    app.sources.push_back(make_source_runtime(cfg, static_cast<int>(index), labels));
    connect_source_graph(app, cfg, app.sources.back(), detector_graph);
  }
  validate_worker_pool_geometry(app.sources);

  if (cfg.profile && app_print_backend()) {
    std::cout << "Application backend:\n" << app.graph.describe_backend() << "\n";
  }

  app.run = app.graph.build(realtime_options(cfg.queue_depth));

  try {
    pull_detections(app, cfg, aggregate_profile);
  } catch (...) {
    app.run.close();
    std::signal(SIGINT, previous_sigint);
    throw;
  }

  app.run.close();
  aggregate_profile.flush();
  for (auto& source : app.sources) {
    source.profile.flush();
    std::cout << "[stream " << source.index << "] processed=" << source.processed << "\n";
  }
  std::signal(SIGINT, previous_sigint);
}

} // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions cli = parse_args(argc, argv);
    if (!fs::exists(cli.config_path)) {
      std::cerr << "Error: config file not found: " << cli.config_path << "\n";
      return 2;
    }

    const AppConfig cfg = load_app_config(cli.config_path);
    if (cli.validate_config_only) {
      std::cout << "Config validated: " << cli.config_path << " (streams=" << cfg.rtsp_urls.size()
                << ", workers=" << cfg.workers << ", queue_depth=" << cfg.queue_depth << ")\n";
      return 0;
    }
    run_app(cfg);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
