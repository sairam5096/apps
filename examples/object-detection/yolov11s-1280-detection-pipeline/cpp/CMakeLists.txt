cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(yolov11s-1280-detection-pipeline LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/ExampleModule.cmake")
sima_neat_apps_module("yolov11s-1280-detection-pipeline" OUTPUT_TARGET_VAR _main_target)

# Explicitly enable ARM NEON (Advanced SIMD) and A65-supported extensions.
# The SDK sets -mcpu=cortex-a65 -march=armv8.2-a+crypto globally; these flags
# override -march for this target to also enable:
#   +simd      — Advanced SIMD / NEON (explicit; ensures vectoriser uses NEON)
#   +dotprod   — 8-bit integer dot-product NEON instructions (A65 hw feature)
#   +fp16      — half-precision float arithmetic (FP16 FMLA etc.)
# -fno-math-errno allows NEON maths builtins without errno writes.
if (_main_target)
  target_compile_options(${_main_target} PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:
      -march=armv8.2-a+simd+dotprod+fp16+crypto
      -ftree-vectorize
      -fno-math-errno
    >
  )
endif()

# ── HW pipeline: FFmpeg decode + neatencoder (SiMa HW H.264 encode) ──────────
# Second binary alongside yolov11s-1280-detection-pipeline that keeps our custom
# MLA post-processing but replaces the SW x264 encoder with SiMa neatencoder,
# freeing the A65 CPU from encode. Decode uses FFmpeg (rtspsrc hangs on device).
if (_main_target)
  add_executable(yolov11s-1280-hw-pipeline
    "${CMAKE_CURRENT_LIST_DIR}/yolov11s-1280-hw-pipeline.cpp")
  target_link_libraries(yolov11s-1280-hw-pipeline
    PRIVATE $<TARGET_PROPERTY:${_main_target},LINK_LIBRARIES>)
  target_include_directories(yolov11s-1280-hw-pipeline
    PRIVATE $<TARGET_PROPERTY:${_main_target},INCLUDE_DIRECTORIES>)
  target_compile_options(yolov11s-1280-hw-pipeline PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:
      -march=armv8.2-a+simd+dotprod+fp16+crypto
      -ftree-vectorize
      -fno-math-errno
    >)
  target_compile_features(yolov11s-1280-hw-pipeline PRIVATE cxx_std_20)
  message(STATUS "yolov11s-1280-hw-pipeline: enabled (FFmpeg decode + neatencoder HW encode)")
endif()

# ── Async pipeline: SDK push/pull API + neatencoder HW encode ─────────────────
# Third binary that demonstrates model.build(RunOptions) + runner.push() /
# runner.pull() in separate threads. queue_depth=3 keeps the MLA fed continuously.
# Decode: FFmpeg (rtspsrc hangs). Encode: neatencoder with non-blocking dispatch.
if (_main_target)
  add_executable(yolov11s-1280-async-pipeline
    "${CMAKE_CURRENT_LIST_DIR}/yolov11s-1280-async-pipeline.cpp")
  target_link_libraries(yolov11s-1280-async-pipeline
    PRIVATE $<TARGET_PROPERTY:${_main_target},LINK_LIBRARIES>)
  target_include_directories(yolov11s-1280-async-pipeline
    PRIVATE $<TARGET_PROPERTY:${_main_target},INCLUDE_DIRECTORIES>)
  target_compile_options(yolov11s-1280-async-pipeline PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:
      -march=armv8.2-a+simd+dotprod+fp16+crypto
      -ftree-vectorize
      -fno-math-errno
    >)
  target_compile_features(yolov11s-1280-async-pipeline PRIVATE cxx_std_20)
  message(STATUS "yolov11s-1280-async-pipeline: enabled (FFmpeg + async push/pull + neatencoder)")
endif()

# Standalone GStreamer RTSP reader test — diagnostic tool to verify which read
# backend works on a given device, before integrating into the main pipeline.
find_package(PkgConfig QUIET)
if (PkgConfig_FOUND)
  # The SDK toolchain keeps its .pc files under a non-default path.
  set(ENV{PKG_CONFIG_PATH}
    "/opt/toolchain/aarch64/modalix/usr/lib/aarch64-linux-gnu/pkgconfig:$ENV{PKG_CONFIG_PATH}")
  pkg_check_modules(OPENCV4 QUIET IMPORTED_TARGET opencv4)
  if (OPENCV4_FOUND)
    add_executable(rtsp_gst_test
      "${CMAKE_CURRENT_LIST_DIR}/rtsp_gst_test.cpp")
    target_link_libraries(rtsp_gst_test PRIVATE PkgConfig::OPENCV4)
    target_compile_features(rtsp_gst_test PRIVATE cxx_std_17)
    target_compile_options(rtsp_gst_test PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang>:
        -march=armv8.2-a+simd+dotprod+fp16+crypto
      >
    )
    message(STATUS "rtsp_gst_test: enabled")
  else()
    message(STATUS "rtsp_gst_test: skipped (opencv4 not found via pkg-config)")
  endif()
endif()

if (BUILD_TESTING AND TARGET SimaNeatApps::support_testing)
  get_filename_component(_apps_root "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
  get_filename_component(_pl_dir    "${CMAKE_CURRENT_LIST_DIR}/.."          ABSOLUTE)

  add_executable(yolov11s-1280-detection-pipeline_unit_test
    "${_pl_dir}/tests/cpp/test_unit.cpp")
  target_link_libraries(yolov11s-1280-detection-pipeline_unit_test
    PRIVATE SimaNeatApps::support_testing)
  target_include_directories(yolov11s-1280-detection-pipeline_unit_test
    PRIVATE "${_apps_root}")
  add_test(
    NAME yolov11s-1280-detection-pipeline.unit
    COMMAND $<TARGET_FILE:yolov11s-1280-detection-pipeline_unit_test>
            $<TARGET_FILE:${_main_target}>)
  set_tests_properties(yolov11s-1280-detection-pipeline.unit PROPERTIES
    SKIP_RETURN_CODE 77 LABELS "unit" WORKING_DIRECTORY "${_apps_root}")

  add_executable(yolov11s-1280-detection-pipeline_e2e_test
    "${_pl_dir}/tests/cpp/test_e2e.cpp")
  target_link_libraries(yolov11s-1280-detection-pipeline_e2e_test
    PRIVATE SimaNeatApps::support_testing)
  target_include_directories(yolov11s-1280-detection-pipeline_e2e_test
    PRIVATE "${_apps_root}")
  add_test(
    NAME yolov11s-1280-detection-pipeline.e2e
    COMMAND $<TARGET_FILE:yolov11s-1280-detection-pipeline_e2e_test>
            $<TARGET_FILE:${_main_target}>)
  set_tests_properties(yolov11s-1280-detection-pipeline.e2e PROPERTIES
    SKIP_RETURN_CODE 77 LABELS "e2e" WORKING_DIRECTORY "${_apps_root}")
endif()
