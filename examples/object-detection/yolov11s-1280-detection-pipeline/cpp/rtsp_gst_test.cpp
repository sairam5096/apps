/**
 * rtsp_gst_test.cpp
 * Standalone GStreamer RTSP reader test.
 *
 * Tries four pipeline variants in order. For each variant it:
 *   1. Prints the pipeline string
 *   2. Opens cv::VideoCapture with a 4-second timeout
 *   3. Reads up to 10 frames and prints format/size + read latency
 *   4. Reports which variant worked
 *
 * Usage:
 *   rtsp_gst_test <rtsp_url>
 *   e.g. rtsp_gst_test rtsp://192.168.132.103:8554/mystream
 */

#include <opencv2/videoio.hpp>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// Try to open cap in a thread; return true if opened within timeout_ms.
static bool try_open(cv::VideoCapture& cap, const std::string& src,
                     int backend, int timeout_ms) {
  std::atomic<bool> done{false};
  std::thread t([&]() {
    cap.open(src, backend);
    done.store(true, std::memory_order_release);
  });
  auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
  while (!done.load(std::memory_order_acquire) && Clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  if (done.load()) { t.join(); return cap.isOpened(); }
  t.detach();  // timed out — abandon
  return false;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <rtsp_url>\n";
    return 1;
  }
  const std::string url = argv[1];

  // --- Pipeline variants to try ---
  struct Variant {
    std::string name;
    std::string src;
    int         backend;
    int         timeout_ms;
  };

  const std::vector<Variant> variants = {
    {
      "GStreamer decodebin (2s internal timeout)",
      "rtspsrc location=" + url + " timeout=2000 tcp-timeout=2000000 latency=200"
      " ! decodebin"
      " ! videoconvert"
      " ! video/x-raw,format=BGR"
      " ! appsink drop=true max-buffers=2 sync=false",
      cv::CAP_GSTREAMER, 4000
    },
    {
      "GStreamer explicit H264 (2s internal timeout)",
      "rtspsrc location=" + url + " timeout=2000 tcp-timeout=2000000 latency=200"
      " ! rtph264depay ! h264parse ! avdec_h264 max-threads=2"
      " ! videoconvert"
      " ! video/x-raw,format=BGR"
      " ! appsink drop=true max-buffers=2 sync=false",
      cv::CAP_GSTREAMER, 4000
    },
    {
      "GStreamer explicit H264 protocols=tcp",
      "rtspsrc location=" + url + " protocols=tcp timeout=2000 tcp-timeout=2000000 latency=200"
      " ! rtph264depay ! h264parse ! avdec_h264 max-threads=2"
      " ! videoconvert"
      " ! video/x-raw,format=BGR"
      " ! appsink drop=true max-buffers=2 sync=false",
      cv::CAP_GSTREAMER, 4000
    },
    {
      "FFmpeg low-latency (no-buffer TCP)",
      url,  // FFmpeg reads native RTSP URL
      cv::CAP_FFMPEG, 8000
    },
  };

  // Set FFmpeg low-latency env for the last variant
  setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS",
         "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
         "|analyzeduration;100000|probesize;65536", 1);

  for (const auto& v : variants) {
    std::cout << "\n====================================================\n";
    std::cout << "Trying: " << v.name << "\n";
    std::cout << "  src: " << v.src << "\n";

    cv::VideoCapture cap;
    bool ok = try_open(cap, v.src, v.backend, v.timeout_ms);
    if (!ok) {
      std::cout << "  RESULT: FAILED to open within " << v.timeout_ms << " ms\n";
      continue;
    }

    const int w   = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int h   = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    const int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    std::cout << "  OPENED: " << w << "x" << h << " @ " << fps << " fps\n";

    std::cout << "  Reading 10 frames...\n";
    double total_read = 0.0;
    int frames_ok = 0;
    for (int i = 0; i < 10; ++i) {
      cv::Mat frame;
      auto t0 = Clock::now();
      bool fok = cap.read(frame);
      double ms = Ms(Clock::now() - t0).count();
      if (!fok || frame.empty()) {
        std::cout << "  frame " << i << ": READ FAILED\n";
        break;
      }
      total_read += ms;
      ++frames_ok;
      std::cout << "  frame " << i << ": " << frame.cols << "x" << frame.rows
                << " channels=" << frame.channels()
                << " depth=" << frame.depth()
                << " read=" << ms << " ms\n";
    }
    if (frames_ok > 0)
      std::cout << "  RESULT: OK — " << frames_ok << "/10 frames, avg read="
                << total_read / frames_ok << " ms\n";

    cap.release();
    if (frames_ok >= 5) {
      std::cout << "\n>>> WINNER: " << v.name << " <<<\n";
      return 0;
    }
  }

  std::cout << "\nAll GStreamer variants failed. FFmpeg is the working backend.\n";
  return 1;
}
