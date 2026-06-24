# YOLOv11s 1280 Detection Pipeline

## Metadata

| Field         | Value                                                        |
|---------------|--------------------------------------------------------------|
| Category      | object-detection                                             |
| Tags          | object-detection, yolov11s, 1280×1280, model-surgery, NEON  |
| Languages     | C++20, Python 3                                              |
| Status        | experimental                                                 |
| Binary name   | `yolov11s-1280-detection-pipeline`                           |
| Model         | `1280.tar.gz` (post-surgery YOLOv11s, EV74 genpreproc)      |

---

## Overview

End-to-end YOLOv11s object-detection pipeline running on the SiMa Modalix SoC.

**Processing chain:**
```
Input (BGR)
  └─ cv::resize → 1280×720
       └─ EV74 genpreproc → CENTER letterbox → 1280×1280
            └─ MLA inference → 6 NDHWC float tensors
                 └─ postproc decode + NMS (NEON-accelerated)
                      └─ draw bounding boxes
                           └─ UDP RTP stream  and/or  PNG save
```

**Supported run modes:**

| Mode | Description |
|------|-------------|
| **Folder** | Read images from a directory, annotate, save to output dir |
| **RTSP + save** | Capture live RTSP stream, annotate, stream via RTP and save PNG frames |
| **RTSP stream-only** | Capture live RTSP stream, annotate, stream via RTP — no disk writes |

Both C++ and Python implementations share the same model and labels.

---

## Model Architecture

The model is a **post-surgery YOLOv11s** compiled for the Modalix MLA at 1280×1280 resolution.

- **EV74 genpreproc stage**: handles color conversion and CENTER letterbox (1280×720 → 1280×1280, gain=1.0, pad_top=280px)
- **MLA stage**: full YOLOv11s backbone + head
- **EV74 detessdequant stage**: dequantizes and outputs 6 raw tensors

**Output tensors** (6 total, NDHWC float32):

| Indices | Shape | Content |
|---------|-------|---------|
| `tensors[0:3]` | (H, W, 4) | Bounding box — cx, cy, w, h in model pixels (DFL already decoded) |
| `tensors[3:6]` | (H, W, 80) | Class probabilities (sigmoid applied, 80 COCO classes) |

Feature map levels and grid sizes:

| Level | Grid | Receptive field |
|-------|------|-----------------|
| 0     | 160×80 | small objects |
| 1     | 80×40  | medium objects |
| 2     | 40×20  | large objects |

---

## Prerequisites

- SiMa NEAT SDK installed and SDK container running
- Compiled model package (`1280.tar.gz`) in `assets/models/`
- COCO labels file: `common/coco_label.txt` (80 classes, one per line)

---

## Build

### Build from the apps repo

```bash
# Inside the NEAT SDK container
cd /workspace/sima-neat/apps
cmake -S . -B build
cmake --build build --target \
  app__workspace_sima_neat_apps_examples_object_detection_yolov11s_1280_detection_pipeline_cpp_yolov11s_1280_detection_pipeline \
  -j4
```

Binary location:
```
apps/build/examples/object-detection/yolov11s-1280-detection-pipeline/yolov11s-1280-detection-pipeline
```

### Build this example standalone

```bash
cd apps
cmake -S examples/object-detection/yolov11s-1280-detection-pipeline/cpp \
      -B build/yolov11s-1280-detection-pipeline
cmake --build build/yolov11s-1280-detection-pipeline -j4
```

---

## Running — C++

### Folder mode

Process all images in a directory; output annotated PNGs to another directory.

```bash
./yolov11s-1280-detection-pipeline \
  assets/models/1280.tar.gz \
  examples/object-detection/yolov11s-1280-detection-pipeline/common/coco_label.txt \
  /path/to/input/images \
  /path/to/output/dir
```

### RTSP mode — stream + save to disk

Capture live RTSP stream, send annotated frames via H.264 RTP to a viewer, and also save each frame as a PNG.

```bash
./yolov11s-1280-detection-pipeline \
  assets/models/1280.tar.gz \
  examples/object-detection/yolov11s-1280-detection-pipeline/common/coco_label.txt \
  /path/to/output/dir \
  --rtsp rtsp://<camera-ip>:<port>/<stream> \
  --frames 300 \
  --insight <viewer-host-ip> \
  --video-port 9000 \
  --meta-port 9100
```

### RTSP mode — stream only (no disk writes)

Omit the `<output_dir>` positional argument. No PNG files are written; annotated frames go directly to the RTP UDP sink.

```bash
./yolov11s-1280-detection-pipeline \
  assets/models/1280.tar.gz \
  examples/object-detection/yolov11s-1280-detection-pipeline/common/coco_label.txt \
  --rtsp rtsp://<camera-ip>:<port>/<stream> \
  --frames 300 \
  --insight <viewer-host-ip> \
  --video-port 9000 \
  --meta-port 9100
```

### C++ CLI reference

```
Folder mode:
  yolov11s-1280-detection-pipeline <model> <labels> <input_dir> <output_dir>

RTSP stream + save:
  yolov11s-1280-detection-pipeline <model> <labels> <output_dir> --rtsp <URL>
    [--frames N] [--insight <host>] [--video-port N] [--meta-port N]

RTSP stream-only (no disk writes):
  yolov11s-1280-detection-pipeline <model> <labels> --rtsp <URL>
    --insight <host> [--frames N] [--video-port N] [--meta-port N]
```

| Argument / Flag | Default | Description |
|-----------------|---------|-------------|
| `<model>` | — | Path to compiled model package (`.tar.gz`) |
| `<labels>` | — | Path to labels text file (one class per line) |
| `<input_dir>` | — | Directory of input images (folder mode only) |
| `<output_dir>` | — | Output directory for annotated PNGs (optional in RTSP mode) |
| `--rtsp <URL>` | — | RTSP stream URL; enables RTSP mode |
| `--frames N` | 50 | Number of RTSP frames to capture |
| `--insight <host>` | — | IP of the neat-insight/OptiView viewer host |
| `--video-port N` | 9000 | UDP port for H.264 RTP video stream |
| `--meta-port N` | 9100 | UDP port base for JSON metadata stream |

---

## Running — Python

### Folder mode

```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/yolov11s-1280-detection-pipeline/python/requirements.txt

python examples/object-detection/yolov11s-1280-detection-pipeline/python/main.py \
  assets/models/1280.tar.gz \
  examples/object-detection/yolov11s-1280-detection-pipeline/common/coco_label.txt \
  /path/to/input/images \
  /path/to/output/dir \
  --min-score 0.45
```

### RTSP mode

```bash
python examples/object-detection/yolov11s-1280-detection-pipeline/python/main.py \
  assets/models/1280.tar.gz \
  examples/object-detection/yolov11s-1280-detection-pipeline/common/coco_label.txt \
  /path/to/output/dir \
  --rtsp rtsp://<camera-ip>:<port>/<stream> \
  --frames 300 \
  --min-score 0.45
```

### Python CLI reference

| Argument / Flag | Default | Description |
|-----------------|---------|-------------|
| `<model>` | — | Path to compiled model package (`.tar.gz`) |
| `<labels_file>` | — | Path to labels text file |
| `<input_dir>` | — | Input images directory (folder mode) or ignored when `--rtsp` given |
| `<output_dir>` | None | Output directory (optional) |
| `--rtsp <URL>` | — | RTSP stream URL; enables RTSP mode |
| `--frames N` | 300 | Number of RTSP frames to capture |
| `--min-score F` | 0.45 | Detection confidence threshold |

---

## OptiView / neat-insight Streaming Setup

The C++ pipeline can stream annotated video and per-frame JSON detections to the **neat-insight** OptiView viewer.

**On the viewer host:**

1. Start `neat-insight` with the device IP:
   ```bash
   CONTAINER_HOST_IP=<device-ip> docker compose up
   ```

2. Open the OptiView web UI (typically `http://<viewer-host>:3000`)

**On the device / pipeline side**, pass `--insight <viewer-host-ip>`:
```bash
./yolov11s-1280-detection-pipeline \
  assets/models/1280.tar.gz \
  common/coco_label.txt \
  --rtsp rtsp://... \
  --insight 192.168.29.127 \
  --video-port 9000 \
  --meta-port 9100
```

The pipeline sends:
- **Video**: H.264 RTP over UDP → `udpsink host=<insight-host> port=9000`
- **Metadata**: JSON object list per frame over UDP → port 9100

JSON format per frame:
```json
{
  "objects": [
    {
      "id": "0",
      "label": "person",
      "confidence": "0.8921",
      "x": "0.1234",
      "y": "0.4567",
      "w": "0.0890",
      "h": "0.2100"
    }
  ]
}
```
Coordinates are normalised to `[0, 1]` relative to stream width/height.

---

## Performance

Measured on **SiMa Modalix** (aarch64) with 21 COCO validation images at 1920×1080 input, served from NFS.

### Stage timing breakdown

| Stage | Mean (ms) | Min (ms) | Max (ms) | Notes |
|-------|-----------|----------|----------|-------|
| **read** | 90.9 | 87.4 | 96.9 | NFS read; overlaps with infer in pipelined mode |
| **preproc** | 2.8 | 2.4 | 7.8 | `cv::resize` to 1280×720 on CPU |
| **infer** | 16.2 | 15.2 | 31.1 | MLA inference (EV74 preproc + MLA + EV74 dequant) |
| **postproc** | 18.2 | 17.8 | 19.4 | Tensor decode + argmax + NMS (ARM NEON A65 vectorised) |
| **render** | 0.2 | 0.0 | 0.6 | `cv::rectangle` + `cv::putText` per detection |
| **save enq** | 0.4 | 0.2 | 1.6 | Async enqueue to save thread (non-blocking) |

> **Note:** `read` is dominated by NFS latency. Local SSD or in-memory dataset reduces this to ~5–10 ms.

### Throughput summary

| Mode | Frame time | FPS |
|------|------------|-----|
| Sequential (before pipeline threading) | ~128 ms | ~7.8 |
| **Pipelined** (read ‖ infer, current) | ~90 ms effective | **~11** |
| Compute-bound only (no read) | ~37 ms | **~27** |

Pipeline threading overlaps the NFS read of frame N+1 with the inference of frame N, so effective wall-clock per frame = `max(read_ms, infer+post+render)` = `max(90, 37)` ≈ 90 ms.

### NEON optimisation impact (ARM Cortex-A65)

The argmax over 80 class scores is the hottest loop in postproc (runs ~1.34 M times per frame across three feature levels). Replacing the scalar loop with 4-accumulator NEON `vmaxq_f32` + `vmaxvq_f32` gave:

| | Before | After |
|--|--------|-------|
| postproc | ~23 ms | ~18 ms |
| Speedup | — | **~22%** |

### Inference latency (MLA steady-state)

| Metric | Value |
|--------|-------|
| MLA inference (steady state) | **15–16 ms** |
| First-frame MLA (cold start) | ~31 ms |
| Model load + graph build | ~400–700 ms (one-time) |
| NMS + decode (3 scales, NEON) | ~18 ms |

---

## Inference Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Input size to model | 1280×720 BGR | Resized from original before EV74 preproc |
| Internal model size | 1280×1280 | After CENTER letterbox (pad top/bottom 280 px) |
| Confidence threshold | 0.45 | Filters class score before NMS |
| NMS IoU threshold | 0.50 | Suppression overlap threshold |
| Max detections (topk) | 24 | Matches `yolo_postproc.json` topk setting |
| Timeout | 5000 ms | `runner.run()` timeout per frame |

---

## Pipeline Architecture

### Thread model

```
Main thread                     Infer thread              Save thread
───────────                     ────────────              ───────────
cap.read() / cv::imread()   →   infer_q.pop()
push to infer_q             ←   cv::resize (preproc)
try_pop result_q (non-block)    runner.run() (MLA)
...read next frame...           decode + NMS (NEON)
                                draw_boxes
                                push to result_q      →   save_q.pop()
draw_timing_overlay                                       cv::imwrite (async)
RTP write (insight)
push to save_q
```

- **`infer_q`** capacity = 2: main thread reads one frame ahead while inference is running
- **`result_q`** capacity = 8: generous headroom so the infer thread never stalls on push
- Disk saves are fully async — PNG write time does not appear on the critical path

### Shutdown order

```
infer_q.close()   →  infer_th.join()  →  drain result_q  →  save_thread.join()  →  runner.close()
```

---

## Source Files

| File | Description |
|------|-------------|
| `cpp/main.cpp` | C++ pipeline — all modes, pipeline threading, NEON decode |
| `cpp/CMakeLists.txt` | CMake build with manual test registration |
| `python/main.py` | Python pipeline — folder and RTSP modes |
| `python/requirements.txt` | Python dependencies |
| `common/coco_label.txt` | 80 COCO class names |
| `tests/cpp/test_unit.cpp` | C++ unit tests (decode + NMS correctness) |
| `tests/cpp/test_e2e.cpp` | C++ end-to-end test (runs binary on real model) |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `archive path does not exist` | Wrong model path | Check that `1280.tar.gz` exists under `assets/models/` |
| No detections | Score threshold too high | Lower `--min-score` (try `0.25`) |
| Detections on wrong classes | Wrong labels file | Verify `coco_label.txt` is 80-class COCO order |
| RTSP stream failed to open | Camera not reachable | Verify `rtsp://` URL with `ffprobe <URL>` |
| RTP video missing in viewer | GStreamer not built with x264 | Check `gst-inspect-1.0 x264enc` on device |
| `CONTAINER_HOST_IP` ICE error | Wrong viewer host IP in supervisord | Set `CONTAINER_HOST_IP=<host-ip>` in supervisord.conf |
| High postproc on non-COCO model | `cls.c != 80` | NEON path falls back to scalar for non-80-class models |
