# 16-Stream Object Detector

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Advanced |
| Tags | object-detection, rtsp, multistream, insight, yolo26 |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | 16-stream-object-detector |
| Model | yolo26m-det-int8-b1 |

## Concept
This example runs up to 16 RTSP streams through one shared YOLO26 detector graph. Each stream owns RTSP decode and clean Insight video output. The detector input uses Neat Graph realtime latest-by-stream links, so source graphs never build stale detector backlogs; detections publish `object-detection` metadata to the matching Insight channel.

## Preview
Snippet from a pipeline run:

![16-stream object detector preview](../../../assets/portal/object-detection/16-stream-object-detector/image.png)

Architecture:
- one composed Neat Graph for all RTSP streams
- one shared detector graph: `Input(detector_frame) -> model.graph() -> Output(detections)`
- per-stream `RtspDecodedInput -> Branch(detector_frame, video)`
- `detector_frame` connects to the detector with `RealtimeLatestByStream`
- `video -> VideoSender(H.264 RTP/UDP)` remains independent from detector scheduling

## Prerequisites
- Installed Neat SDK.
- Reachable RTSP camera URLs encoded at the intended detector candidate FPS.
- A YOLO26 INT8 batch-1 model pack downloaded into `assets/models/`.
- Edit `src/common/config.yaml` before running with real streams.
- On Modalix DevKit, run `bash /usr/bin/fix_devkit_runtime.sh` before starting the example if the runtime has been used by earlier ML/video apps.

## Command-line options
- `--config <path>`: path to the YAML configuration file
- `--validate-config-only`: validate the config and exit without opening RTSP streams
- `--help`: print the CLI help text

## Download Models
The default model is `yolo26m-det-int8-b1.tar.gz`.

```bash
mkdir -p assets/models
cd assets/models
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.0.0/models/modalix/yolo26-detection/yolo26m-det-int8-b1.tar.gz
cd ../..
```

## Build
### Build From The Apps Repo
```bash
cd <apps-repo-root>
./build.sh
```

### Build This Example Directly With CMake
```bash
cd <apps-repo-root>
cmake -S examples/object-detection/16-stream-object-detector/src/cpp -B build/16-stream-object-detector
cmake --build build/16-stream-object-detector -j
```

## Run
### Validate Config Only
This is useful for a quick smoke test without opening RTSP streams.

```bash
./build/examples/object-detection/16-stream-object-detector/16-stream-object-detector \
  --config examples/object-detection/16-stream-object-detector/src/common/config.yaml \
  --validate-config-only
```

### C++
```bash
SIMA_GST_RUN_INPUT_TIMEOUT_MS=120000 \
./build/examples/object-detection/16-stream-object-detector/16-stream-object-detector \
  --config examples/object-detection/16-stream-object-detector/src/common/config.yaml
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/16-stream-object-detector/src/python/requirements.txt
SIMA_GST_RUN_INPUT_TIMEOUT_MS=120000 python3 examples/object-detection/16-stream-object-detector/src/python/main.py \
  --config examples/object-detection/16-stream-object-detector/src/common/config.yaml
```

## Debugging notes
- The checked-in `src/common/config.yaml` includes placeholder stream URLs and Insight host values. Replace them before running.
- Set the intended detector candidate FPS at the RTSP source. This example does not downsample in the scheduler and does not use `VideoRate`.
- `inference.workers` must be `1`. The graph-linked C++ path uses one shared detector graph; adding app-level detector workers is intentionally not part of this example.
- `inference.queue_depth` sets the realtime link/downstream detector queue depth. Leave it at the default unless you are profiling throughput.
- `output.debug_dir`/`output.save_every` are ignored by the graph-linked C++ path so the live path stays zero-copy and customer-facing.
- Profiling prints aggregate FPS plus per-stream metadata, parse, send, and detection-count summaries. It does not enable heavyweight GStreamer/plugin traces by default; set `APP16_GST_PROFILE=1` only when you need those traces.

## Notes
- Point `model.path` at a YOLO26 detection pack. This example does not use a `model.family` config key.
- The C++ path uses public Graph composition: `RtspDecodedInput`, `graphs::Branch`, `VideoSender`, realtime graph links, `Input -> model.graph() -> Output`, and `MetadataSender`.
- The live video path is graph-owned and independent from detector scheduling.
- Metadata uses the graph-preserved stream id, timestamp, and frame id so Insight overlays detections on the intended video stream.
- `output.video_enabled: false` disables per-stream H.264 video output. Metadata still runs.

## Source Files
- C++: `src/cpp/main.cpp`
- C++ tests: `tests/cpp/test_unit.cpp`, `tests/cpp/test_e2e.cpp`
- Python: `src/python/main.py`
- Python tests: `tests/python/test_unit.py`, `tests/python/test_e2e.py`
- Shared assets: `src/common/`
