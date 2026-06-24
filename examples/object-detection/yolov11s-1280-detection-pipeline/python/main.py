"""YOLOv11 1280x720 detection pipeline using pyneat.

Model: 1280.tar.gz (yolo11 post-surgery)
  Internal pipeline: EV74-genpreproc -> MLA -> EV74-detessdequant
  Input:  1280x720 BGR (preproc letterboxes to 1280x1280 internally)
  Output: 6 NDHWC float tensors
    tensors[0:3] bbox (H,W,4)  cx,cy,w,h in model-space pixels (DFL decoded in model)
    tensors[3:6] cls  (H,W,80) probabilities (already sigmoid)

Usage:
  Folder mode: main.py <model> <labels> <input_dir> <output_dir> [--min-score F]
  RTSP mode:   main.py <model> <labels> <output_dir> --rtsp <URL> [--frames N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyneat

# Model input dimensions (what preproc expects)
INPUT_W = 1280
INPUT_H = 720
# Model internal dimensions (after preproc letterbox)
MODEL_W = 1280
MODEL_H = 1280

# CENTER letterbox params: 1280x720 -> 1280x1280
# gain = min(1280/720, 1280/1280) = 1.0
# pad_top  = (1280 - 720 * 1.0) / 2 = 280.0
# pad_left = (1280 - 1280 * 1.0) / 2 = 0.0
_LB_GAIN      = min(MODEL_H / INPUT_H, MODEL_W / INPUT_W)   # 1.0
_LB_PAD_TOP   = (MODEL_H - INPUT_H * _LB_GAIN) / 2.0       # 280.0
_LB_PAD_LEFT  = (MODEL_W - INPUT_W * _LB_GAIN) / 2.0       # 0.0
_LB_INV_GAIN  = 1.0 / _LB_GAIN                              # 1.0

MIN_SCORE = 0.45
NMS_IOU   = 0.50
MAX_DET   = 24   # matches topk in yolo_postproc.json

DEFAULT_RTSP_FRAMES = 50

BOX_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
]


# ─── Utility ────────────────────────────────────────────────────────────────

def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Labels file does not exist: {path}")
    labels = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not labels:
        raise ValueError(f"Labels file is empty: {path}")
    return labels


def tensor_to_numpy(t: pyneat.Tensor) -> np.ndarray:
    return np.asarray(t.to_numpy(copy=True))


def iter_tensors(sample: pyneat.Sample):
    if sample.kind == pyneat.SampleKind.TensorSet:
        yield from sample.tensors
        return
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        yield sample.tensor
    for field in sample.fields:
        yield from iter_tensors(field)


def tensor_to_hwc_f32(t: pyneat.Tensor) -> np.ndarray:
    """Convert tensor to HWC float32, stripping leading size-1 dims (NHWC, NDHWC, etc.)."""
    arr = tensor_to_numpy(t).astype(np.float32)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"unexpected tensor rank {arr.ndim}, shape {arr.shape}")
    return arr


# ─── Post-surgery YOLOv11 decode ─────────────────────────────────────────────

def iou_xyxy(a: dict, b: dict) -> float:
    xx1 = max(a["x1"], b["x1"]); yy1 = max(a["y1"], b["y1"])
    xx2 = min(a["x2"], b["x2"]); yy2 = min(a["y2"], b["y2"])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    den = area_a + area_b - inter
    return inter / den if den > 0 else 0.0


def decode_yolov11_tensors(tensors: list, min_score: float) -> list[dict]:
    """Decode post-surgery YOLOv11 output from 6 NDHWC float tensors.

    tensors[0:3] bbox: shape (H,W,4)  channels = [cx, cy, w, h] in model pixels
    tensors[3:6] cls:  shape (H,W,80) class probabilities (already sigmoid)

    Applies CENTER letterbox inverse:
        x_orig = (cx ± w/2 - pad_left) / gain
        y_orig = (cy ± h/2 - pad_top)  / gain
    Returns boxes in INPUT_W x INPUT_H (1280x720) coordinate space.
    """
    all_x1: list[np.ndarray] = []
    all_y1: list[np.ndarray] = []
    all_x2: list[np.ndarray] = []
    all_y2: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_classes: list[np.ndarray] = []

    for hi in range(3):
        bbox = tensor_to_hwc_f32(tensors[hi])     # (H, W, 4)
        cls  = tensor_to_hwc_f32(tensors[hi + 3]) # (H, W, 80)
        H, W = cls.shape[:2]

        # Vectorised per-anchor decode
        cls_flat  = cls.reshape(-1, cls.shape[2])  # (H*W, 80)
        bbox_flat = bbox.reshape(-1, 4)             # (H*W, 4)

        best_cls   = np.argmax(cls_flat, axis=1).astype(np.int32)  # (H*W,)
        best_score = cls_flat[np.arange(H * W), best_cls]          # (H*W,) already sigmoid

        mask = best_score >= min_score
        if not np.any(mask):
            continue

        bs = best_score[mask].astype(np.float32)
        bc = best_cls[mask]
        bb = bbox_flat[mask]  # (K, 4)

        cx, cy = bb[:, 0], bb[:, 1]
        w,  h  = bb[:, 2], bb[:, 3]

        # Inverse letterbox → 1280×720 original space
        x1 = np.clip((cx - w * 0.5 - _LB_PAD_LEFT) * _LB_INV_GAIN, 0.0, INPUT_W)
        y1 = np.clip((cy - h * 0.5 - _LB_PAD_TOP)  * _LB_INV_GAIN, 0.0, INPUT_H)
        x2 = np.clip((cx + w * 0.5 - _LB_PAD_LEFT) * _LB_INV_GAIN, 0.0, INPUT_W)
        y2 = np.clip((cy + h * 0.5 - _LB_PAD_TOP)  * _LB_INV_GAIN, 0.0, INPUT_H)

        valid = (x2 > x1) & (y2 > y1)
        if not np.any(valid):
            continue

        all_x1.append(x1[valid]); all_y1.append(y1[valid])
        all_x2.append(x2[valid]); all_y2.append(y2[valid])
        all_scores.append(bs[valid])
        all_classes.append(bc[valid])

    if not all_x1:
        return []

    boxes   = np.stack([np.concatenate(all_x1), np.concatenate(all_y1),
                        np.concatenate(all_x2), np.concatenate(all_y2)], axis=1)
    scores  = np.concatenate(all_scores)
    classes = np.concatenate(all_classes)

    order = np.argsort(-scores, kind="stable")

    keep: list[dict] = []
    suppressed = np.zeros(len(order), dtype=bool)
    for idx in range(len(order)):
        if suppressed[idx]:
            continue
        i = order[idx]
        d = dict(x1=float(boxes[i, 0]), y1=float(boxes[i, 1]),
                 x2=float(boxes[i, 2]), y2=float(boxes[i, 3]),
                 score=float(scores[i]), class_id=int(classes[i]))
        keep.append(d)
        if len(keep) >= MAX_DET:
            break
        # Suppress overlapping same-class boxes
        for jdx in range(idx + 1, len(order)):
            if suppressed[jdx]:
                continue
            j = order[jdx]
            if int(classes[j]) == d["class_id"]:
                dj = dict(x1=float(boxes[j, 0]), y1=float(boxes[j, 1]),
                          x2=float(boxes[j, 2]), y2=float(boxes[j, 3]),
                          score=0, class_id=0)
                if iou_xyxy(d, dj) > NMS_IOU:
                    suppressed[jdx] = True

    return keep


def decode_yolo_boxes_from_sample(sample: pyneat.Sample, min_score: float) -> list[dict]:
    tensors = list(iter_tensors(sample))
    if not tensors:
        raise ValueError("no output tensors found in sample")
    if len(tensors) < 6:
        raise ValueError(f"expected 6 tensors, got {len(tensors)}")
    return decode_yolov11_tensors(tensors, min_score)


# ─── Drawing ─────────────────────────────────────────────────────────────────

def draw_boxes(frame: np.ndarray, boxes: list[dict], labels: list[str]) -> None:
    for b in boxes:
        x1, y1 = int(b["x1"]), int(b["y1"])
        x2, y2 = int(b["x2"]), int(b["y2"])
        cls_id = b["class_id"]
        color  = BOX_COLORS[cls_id % len(BOX_COLORS)]
        label  = labels[cls_id] if cls_id < len(labels) else str(cls_id)
        text   = f"{label} {b['score']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def scale_boxes(boxes: list[dict], from_w: int, from_h: int, to_w: int, to_h: int) -> list[dict]:
    sx, sy = to_w / float(from_w), to_h / float(from_h)
    return [dict(x1=b["x1"]*sx, y1=b["y1"]*sy, x2=b["x2"]*sx, y2=b["y2"]*sy,
                 score=b["score"], class_id=b["class_id"]) for b in boxes]


# ─── NEAT session ────────────────────────────────────────────────────────────

def build_neat_session(model_path: str):
    """Build pyneat runner. Model expects INPUT_W x INPUT_H BGR input."""
    import os
    # Allow CPU tensor → EV74 DMA copy (avoids zero-copy requirement on EV74-genpreproc route).
    os.environ.setdefault("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY", "1")

    opt = pyneat.ModelOptions()
    opt.preprocess.input_max_width  = INPUT_W   # 1280
    opt.preprocess.input_max_height = INPUT_H   # 720
    opt.preprocess.input_max_depth  = 3
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR

    model = pyneat.Model(model_path, opt)
    runner = model.build()
    print(f"[BUILD] Model loaded: {model_path}")
    print(f"[BUILD] Input: {INPUT_W}x{INPUT_H} BGR -> {MODEL_W}x{MODEL_H} letterbox")
    return model, runner


def infer_frame(runner, bgr: np.ndarray, min_score: float) -> tuple:
    """Resize frame to INPUT_W×INPUT_H, infer, decode bboxes in 1280×720 space.

    Returns (boxes, preproc_ms, infer_ms, post_ms).
    """
    t0 = time.perf_counter()
    frame_720 = cv2.resize(bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    frame_720 = np.ascontiguousarray(frame_720, dtype=np.uint8)
    t_in = pyneat.Tensor.from_numpy(frame_720, copy=True, image_format=pyneat.PixelFormat.BGR)
    t1 = time.perf_counter()

    if not runner.push([t_in]):
        raise RuntimeError("push failed")
    out = runner.pull(timeout_ms=5000)
    if out is None:
        raise RuntimeError("pull timed out")
    t2 = time.perf_counter()

    boxes = decode_yolo_boxes_from_sample(out, min_score)
    t3 = time.perf_counter()

    return boxes, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3


# ─── Timing helpers ──────────────────────────────────────────────────────────

def draw_timing_overlay(frame: np.ndarray,
                        read_ms: float, preproc_ms: float, infer_ms: float,
                        post_ms: float, render_ms: float, save_ms: float) -> None:
    total = read_ms + preproc_ms + infer_ms + post_ms + render_ms + save_ms
    row0 = f"read={read_ms:.1f}ms  preproc={preproc_ms:.1f}ms  infer={infer_ms:.1f}ms"
    row1 = f"post={post_ms:.1f}ms  render={render_ms:.1f}ms  save={save_ms:.1f}ms  total={total:.1f}ms"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thick = 1
    y = frame.shape[0] - 4
    for row in (row1, row0):
        (w, h), base = cv2.getTextSize(row, font, scale, thick)
        cv2.rectangle(frame, (2, y - h - 2), (4 + w, y + base + 2), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, row, (4, y), font, scale, (0, 255, 255), thick, cv2.LINE_AA)
        y -= h + base + 6


def print_timing_summary(timings: list, n: int) -> None:
    names = ["read", "preproc", "infer", "postproc", "render", "save"]
    sep = "─" * 46
    print(f"\n── Stage timings ({n} frames) ──────────────────────────")
    print(f"{'stage':<10}  {'mean ms':>8}  {'min ms':>8}  {'max ms':>8}")
    print(sep)
    total_mean = 0.0
    for i, name in enumerate(names):
        vals = [t[i] for t in timings]
        mean_v, min_v, max_v = sum(vals) / n, min(vals), max(vals)
        total_mean += mean_v
        print(f"{name:<10}  {mean_v:>8.2f}  {min_v:>8.2f}  {max_v:>8.2f}")
    print(sep)
    print(f"{'total':<10}  {total_mean:>8.2f}")
    print(sep)


# ─── Run modes ───────────────────────────────────────────────────────────────

def run_folder_mode(args: argparse.Namespace, labels: list[str]) -> int:
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in input_dir.iterdir() if p.is_file() and is_image(p))
    if not images:
        print(f"No images found in {input_dir}", file=sys.stderr)
        return 3
    print(f"Found {len(images)} images")

    try:
        _, run = build_neat_session(args.model)
    except Exception as e:
        print(f"Error building pipeline: {e}", file=sys.stderr)
        return 4

    processed = 0
    frame_timings: list = []
    for img_path in images:
        t_read0 = time.perf_counter()
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        read_ms = (time.perf_counter() - t_read0) * 1e3
        if bgr is None:
            print(f"Skipping unreadable: {img_path.name}", file=sys.stderr)
            continue
        orig_h, orig_w = bgr.shape[:2]
        try:
            boxes_720, preproc_ms, infer_ms, post_ms = infer_frame(run, bgr, args.min_score)
        except Exception as e:
            print(f"Inference failed for {img_path.name}: {e}", file=sys.stderr)
            continue

        t_rndr0 = time.perf_counter()
        boxes = scale_boxes(boxes_720, INPUT_W, INPUT_H, orig_w, orig_h)
        draw_boxes(bgr, boxes, labels)
        render_ms = (time.perf_counter() - t_rndr0) * 1e3

        # Encode to PNG buffer first (measures compression cost), then write to disk.
        t_save0 = time.perf_counter()
        ok, buf = cv2.imencode(".png", bgr)
        encode_ms = (time.perf_counter() - t_save0) * 1e3

        draw_timing_overlay(bgr, read_ms, preproc_ms, infer_ms, post_ms, render_ms, encode_ms)
        ok2, buf2 = cv2.imencode(".png", bgr)   # re-encode with overlay baked in

        out_path = output_dir / f"{img_path.stem}.png"
        t_write0 = time.perf_counter()
        out_path.write_bytes(buf2.tobytes() if ok2 else buf.tobytes())
        save_ms = encode_ms + (time.perf_counter() - t_write0) * 1e3

        processed += 1
        frame_timings.append((read_ms, preproc_ms, infer_ms, post_ms, render_ms, save_ms))
        total_ms = sum(frame_timings[-1])
        print(f"[{processed}/{len(images)}] {img_path.name} -> {out_path.name} ({len(boxes)} det)"
              f" | read={read_ms:.1f} pre={preproc_ms:.1f} inf={infer_ms:.1f}"
              f" post={post_ms:.1f} rndr={render_ms:.1f} save={save_ms:.1f} | total={total_ms:.1f}ms")

    run.close()
    if frame_timings:
        print_timing_summary(frame_timings, processed)
    print(f"Done: {processed} images processed")
    return 0


def run_rtsp_mode(args: argparse.Namespace, labels: list[str]) -> int:
    output_dir = Path(args.input_dir)   # 3rd positional used as output in RTSP mode
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.rtsp)
    if not cap.isOpened():
        print(f"Failed to open RTSP stream: {args.rtsp}", file=sys.stderr)
        return 2

    stream_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or INPUT_W)
    stream_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or INPUT_H)
    print(f"[RTSP] stream {stream_w}x{stream_h} -> resizing to {INPUT_W}x{INPUT_H} for model")
    print(f"[RTSP] capturing {args.frames} frames -> {output_dir}")

    try:
        _, run = build_neat_session(args.model)
    except Exception as e:
        cap.release()
        print(f"Error building pipeline: {e}", file=sys.stderr)
        return 4

    processed = 0
    errors    = 0
    frame_timings: list = []
    try:
        while processed < args.frames:
            t_read0 = time.perf_counter()
            ok, bgr = cap.read()
            read_ms = (time.perf_counter() - t_read0) * 1e3
            if not ok:
                print("RTSP stream ended or read failed", file=sys.stderr)
                break
            try:
                boxes_720, preproc_ms, infer_ms, post_ms = infer_frame(run, bgr, args.min_score)
            except Exception as e:
                errors += 1
                print(f"Inference error frame {processed}: {e}", file=sys.stderr)
                if errors > 5:
                    print("Too many errors, aborting", file=sys.stderr)
                    break
                continue

            t_rndr0 = time.perf_counter()
            boxes = scale_boxes(boxes_720, INPUT_W, INPUT_H, stream_w, stream_h)
            draw_boxes(bgr, boxes, labels)
            render_ms = (time.perf_counter() - t_rndr0) * 1e3

            t_save0 = time.perf_counter()
            _, buf = cv2.imencode(".png", bgr)
            encode_ms = (time.perf_counter() - t_save0) * 1e3

            draw_timing_overlay(bgr, read_ms, preproc_ms, infer_ms, post_ms, render_ms, encode_ms)
            _, buf2 = cv2.imencode(".png", bgr)

            out_path = output_dir / f"frame_{processed:05d}.png"
            t_write0 = time.perf_counter()
            out_path.write_bytes(buf2.tobytes())
            save_ms = encode_ms + (time.perf_counter() - t_write0) * 1e3

            processed += 1
            frame_timings.append((read_ms, preproc_ms, infer_ms, post_ms, render_ms, save_ms))
            total_ms = sum(frame_timings[-1])
            print(f"[{processed}/{args.frames}] frame_{processed-1:05d}.png ({len(boxes)} det)"
                  f" | read={read_ms:.1f} pre={preproc_ms:.1f} inf={infer_ms:.1f}"
                  f" post={post_ms:.1f} rndr={render_ms:.1f} save={save_ms:.1f} | total={total_ms:.1f}ms")
    finally:
        cap.release()
        run.close()

    if frame_timings:
        print_timing_summary(frame_timings, processed)
    print(f"Done: {processed} frames, {errors} errors")
    return 0 if processed > 0 else 3


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="YOLOv11 1280x720 detection pipeline (post-surgery, EV74-preproc)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Folder mode:\n"
            "  main.py model labels input_dir output_dir [--min-score F]\n\n"
            "RTSP mode:\n"
            "  main.py model labels output_dir --rtsp URL [--frames N] [--min-score F]"
        ),
    )
    parser.add_argument("model",      type=str, help="Path to compiled model package (.tar.gz)")
    parser.add_argument("labels_file", type=str, help="Path to labels txt file")
    parser.add_argument("input_dir",  type=str,
                        help="Input image directory (folder mode) or output directory (RTSP mode)")
    parser.add_argument("output_dir", type=str, nargs="?", default=None,
                        help="Output directory (folder mode only)")
    parser.add_argument("--rtsp",    metavar="URL", default=None, help="RTSP stream URL")
    parser.add_argument("--frames",  type=int, default=DEFAULT_RTSP_FRAMES,
                        help=f"Frames to capture in RTSP mode (default: {DEFAULT_RTSP_FRAMES})")
    parser.add_argument("--min-score", type=float, default=MIN_SCORE,
                        help=f"Confidence threshold (default: {MIN_SCORE})")
    args = parser.parse_args()

    try:
        labels = load_labels(Path(args.labels_file))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if args.rtsp:
        return run_rtsp_mode(args, labels)
    else:
        if not args.output_dir:
            parser.error("output_dir is required in folder mode (use --rtsp for RTSP mode)")
        return run_folder_mode(args, labels)


if __name__ == "__main__":
    raise SystemExit(main())
