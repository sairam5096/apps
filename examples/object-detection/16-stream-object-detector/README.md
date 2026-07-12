# 16-Stream Object Detector

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Advanced |
| Tags | object-detection, RTSP, multistream, Insight, YOLO26 |
| Language | C++ (validated), Python (reference only) |
| Binary | `16-stream-object-detector` |
| Validated profile | 24 × 1280×720 at 20 FPS |

## What the validated C++ application does

Despite its historical name, this example's checked-in profile processes 24 RTSP/H.264
cameras. Each source is depacketized once and then follows two paths:

```text
RTSP/H.264 -> AU-aligned H.264
  detector: SimaDecode -> latest-by-stream C++ mux -> Preproc -> YOLO26n -> Boxdecode
  video:    owned CPU copy of encoded AU -> bounded delay -> VideoSender -> Insight channel N
  boxes:    detection JSON held by PTS epoch -> released with matching delayed video -> channel N
```

The video path does **not** copy decoded frames to CPU and does not allocate a second EV74
frame queue or 24 encoders. Core's encoded-AU callback owns the H.264 bytes before returning,
so App16 can retain them safely while the decoder consumes the original buffer. One fair
round-robin dispatcher uses nonblocking `Run::try_push`; a busy VideoSender leaves its AU at
the head of only that channel's queue while other channels continue.

Metadata UDP is also nonblocking. Local UDP congestion can drop a metadata datagram, but it
cannot stall every channel's video/inference dispatcher. App16 rate-limits warnings and reports
`metadata_send_ok`, `metadata_send_fail`, `metadata_would_block`, and
`metadata_no_buffer_space` in liveness/final statistics.

### Buffer configuration

For each stream the CPU-owned encoded queue uses:

```text
frame capacity = ceil(sync_delay_ms * source_fps / 1000) + max(64, 2 * source_fps)
byte capacity  = 16 MiB
maximum VideoSender input AU = 16 MiB
```

At 20 FPS with the checked-in 400 ms delay this is 72 encoded AUs per stream. The frame
reserve absorbs a legal RTSP catch-up burst; the byte limit is the actual memory bound and
also guarantees that an AU accepted by the delay queue fits the VideoSender input. `Sample`,
caps, deque, and metadata bookkeeping add a small amount of ordinary CPU memory. No part of
this queue consumes EV74 memory.

The checked-in decoder configuration is 16 output surfaces and two compressed-input buffers
per stream. Those are codec pool surfaces, not application staging queues. The detector link
admits one decoded frame per stream. `inference.internal_queue_depth` remains an explicit
tuning input; the checked-in value is the current handoff candidate, not a claim that depth one
or two is universally optimal. Pin the final value only after the clean-install 24×20 gate.

PTS is matched within an epoch rather than as one never-ending numeric timeline. The ordered,
no-B-frame encoded branch is the only authority that starts a new epoch after a large backward
PTS discontinuity. A bounded CPU-only history retains exact `(PTS, epoch)` identities after an
AU leaves the delay queue. Out-of-order inference completions therefore resolve against the
original encoded AU instead of advancing an epoch from metadata completion order. An ambiguous
or expired identity after a reconnect is rejected and counted as
`metadata_pts_epoch_unresolved`; guessing could draw a valid box on the wrong loop.

## Prerequisites

- Apps, Core, and Internals built and installed from the matching handoff revisions.
- A Modalix DevKit with the decoder service running.
- Insight reachable from the DevKit and configured for channels 0 through 23.
- Twenty-four H.264 RTSP sources at the caps declared in the config.
- A YOLO26n model pack.

`model.path` and `model.labels` are resolved relative to the YAML file, not the caller's
working directory. Put the model beside `src/common/config.yaml` using its checked-in name,
or change the relative path. Do not add a build-tree `RPATH` or rely on `LD_LIBRARY_PATH`;
install the matching Core/Internals artifacts into their normal target locations.

### Validated source set

Do not mix these revisions with another handoff:

- Apps implementation: `d6e9e4b887ff06f160c0186a1e1ca2406e9dad63`
- Core: `33aedfd2de7f46c7b904d9f36ac09dc2c302bee9`
- Internals: `79c856c8d4f013982a211fc4f498e42f9ca13737`

The Apps SHA deliberately names the functional commit before this non-self-referential
documentation pin. For a clean install, use the matching revisions and artifact checksums in the
delivered handoff bundle's `source-handoff/manifest.json` and `SHA256SUMS`; that manifest records
the final Apps documentation commit as well as this implementation revision.

## Build and run

From the Apps repository root:

```bash
./build.sh

./build/examples/object-detection/16-stream-object-detector/16-stream-object-detector \
  --config examples/object-detection/16-stream-object-detector/src/common/config.yaml \
  --validate-config-only

./build/examples/object-detection/16-stream-object-detector/16-stream-object-detector \
  --config examples/object-detection/16-stream-object-detector/src/common/config.yaml
```

The config check should report:

```text
streams=24, model_path=..., labels_path=..., workers=1, queue_depth=1, internal_queue_depth=1,
max_inflight_per_stream=1,
insight_visible_streams=24, sync_delay_ms=400, decoder_admission=core
```

No private `SIMA_*` feature switch is used or required. Useful application-owned diagnostics
are:

- `APP16_LIVENESS_MS=5000`: periodic per-stream progress and send-failure counters.
- `APP16_VERBOSE=1`: verbose application logs; leave unset for FPS acceptance.
- `APP16_PRINT_BACKEND=1`: print the selected graph backend.
- `APP16_FRAMES_PER_STREAM=N`: stop after every stream produces `N` detections.

Important YAML settings:

- `input.skip_rtsp_probe: true` requires correct width, height, and FPS.
- `input.drop_on_latency: true` prevents a stale RTSP backlog.
- `inference.workers` must be `1`.
- `inference.queue_depth: 1` keeps the per-stream detector input current.
- `inference.internal_queue_depth` is a small global model-stage handoff, not a per-camera
  decoded-frame queue. Keep it explicit and confirm it with the final 24×20 clean-install run.
- `inference.max_inflight_per_stream` is the public latest-mux credit for each camera. Keep the
  checked-in safety value until measuring 1, 2, 3, and 4 on the clean 24×20 install; then pin the
  winner that sustains approximately 20 detections/s per channel without stale boxes or EV74
  pressure. Do not replace this setting with a private environment override.
- `output.insight.max_visible_streams` publishes the first N configured streams. Streams after
  N still run inference but intentionally have no Insight video or metadata sender. Set it to
  the stream count for acceptance. Core's encoded callback is graph-global: with partial
  visibility it still copies each hidden branch's encoded AU to CPU before App16 discards it.
  No callback is installed when zero video senders are visible.
- `output.insight.sync_delay_ms` is the CPU encoded-video/metadata synchronization window.
- `output.video_enabled: false` is a metadata-only diagnostic, not the visual acceptance path.

The removed `output.hidden_streams`, `output.debug_dir`, and `output.save_every` settings were
abandoned experimental paths and are no longer supported.

## Use real moving H.264 without B-frames

A repeated still image cannot prove smoothness, ordering, or that frames are fresh. Convert a
real moving source to constant-frame-rate H.264 with no B-frames before publishing it:

```bash
mkdir -p /root/.simaai/neat-insight/media/app16
ffmpeg -y -i /path/to/real-moving-source.mp4 \
  -an -vf 'fps=20,scale=1280:720:flags=lanczos,format=yuv420p' \
  -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p \
  -g 20 -keyint_min 20 -bf 0 -refs 1 -sc_threshold 0 \
  -x264-params 'bframes=0:repeat-headers=1' -fps_mode cfr \
  /root/.simaai/neat-insight/media/app16/moving-720p20-no-b.mp4
```

Verify the asset rather than trusting its filename:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate,has_b_frames \
  -of default=nw=1 \
  /root/.simaai/neat-insight/media/app16/moving-720p20-no-b.mp4
# Expected: h264, 1280x720, 20/1, has_b_frames=0
```

Publish independent RTSP sessions to MediaMTX (adjust its port/config as needed):

```bash
mkdir -p /tmp/app16-rtsp-pubs
for i in $(seq 1 24); do
  setsid -f ffmpeg -nostdin -hide_banner -loglevel warning \
    -re -stream_loop -1 \
    -i /root/.simaai/neat-insight/media/app16/moving-720p20-no-b.mp4 \
    -map 0:v:0 -an -c:v copy -f rtsp -rtsp_transport tcp \
    "rtsp://127.0.0.1:8554/src${i}" \
    >"/tmp/app16-rtsp-pubs/src${i}.out" \
    2>"/tmp/app16-rtsp-pubs/src${i}.err" </dev/null
done
```

The loop above proves per-tile motion and boxes, but every channel intentionally carries the
same media. It therefore cannot prove that channel 7 was not accidentally wired to channel 3.
For channel-to-media mapping acceptance, generate real moving fixtures with a stable unique
color marker and human-readable `CHxx` label:

```bash
python3 stress/app16_make_identity_fixtures.py \
  --input /path/to/real-moving-source.mp4 \
  --output-dir /root/.simaai/neat-insight/media/app16/identity-24x20 \
  --channel-ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 \
  --width 1280 --height 720 --fps 20 --duration-seconds 120 --overwrite
```

The generator adds both the stable channel marker and an LSB-first binary frame counter. It
verifies every output with `ffprobe` (`h264`, requested caps/FPS, and `has_b_frames=0`), decodes
the first temporal-code sequence back to `0..15`, and writes `identity-manifest.json`. Publish
the matching file to each RTSP source; App16 channel 0 reads `src1`, channel 1 reads `src2`, and
so on:

```bash
mkdir -p /tmp/app16-rtsp-pubs
for channel in $(seq 0 23); do
  source_number=$((channel + 1))
  file=$(printf \
    '/root/.simaai/neat-insight/media/app16/identity-24x20/channel-%02d-1280x720p20-no-b.mp4' \
    "$channel")
  setsid -f ffmpeg -nostdin -hide_banner -loglevel warning \
    -re -stream_loop -1 -i "$file" -map 0:v:0 -an -c:v copy \
    -f rtsp -rtsp_transport tcp "rtsp://127.0.0.1:8554/src${source_number}" \
    >"/tmp/app16-rtsp-pubs/src${source_number}.out" \
    2>"/tmp/app16-rtsp-pubs/src${source_number}.err" </dev/null
done
```

## Insight acceptance gates

Use the controlled visual/temporal gate for pixel direction and box synchronization, and the
read-only gate for independent operator-viewer continuity/rate evidence. Neither substitutes for
the other.

### 1. Short visual/content gate

`stress/app16_insight_visual_gate.py` creates a **dedicated** Chromium target through CDP. It
does not navigate an existing operator tab and always closes its target. For every requested
channel it requires:

- the exact, unique channel ID and active WebRTC tile;
- correct decoded dimensions;
- advancing playback and at least 90% of the expected decoded FPS over the sample;
- a changing 32×18 video-pixel signature (not merely `currentTime`);
- visible overlay pixels and increasing real overlay paint operations.

With `--identity-manifest`, it additionally samples the stable marker patch, maps the measured
RGB to the nearest expected channel marker, and requires an exact unique assignment for every
tile. This optional mode catches duplicated or cross-wired media without making normal customer
streams depend on a synthetic pixel signature. When the manifest includes the generated temporal
marker, the gate samples it seven times by default and also requires:

- every binary frame-code segment, captured from one composited video-frame draw, to advance
  forward at the requested rate (backward/repeated content fails without per-bit tearing);
- metadata message count and metadata PTS-derived frame code to advance on every segment;
- metadata `rtp_timestamp` to equal the 90 kHz value derived from its PTS; and
- the displayed frame's browser `VideoFrameMetadata.rtpTimestamp` and App metadata
  `rtp_timestamp` to remain within `sync_tolerance_frames * 90000 / fps` ticks. A stale or large
  RTP lag fails, as do stale repeated boxes.

The fixture frame code and pipeline PTS can have an arbitrary channel-specific origin offset when
the publisher was already running before App16 joined. The gate records that offset and its
stability as a diagnostic but never requires it to be near zero; direct displayed-video/metadata
RTP distance is the synchronization authority.

The pre-document hook observes both viewer-created DataChannels and remotely created channels
delivered through the browser's `datachannel` event, before Insight installs its own handler. It
also starts a `requestVideoFrameCallback` loop for each video tile and retains the latest presented
frame RTP timestamp and callback count.

Insight currently assigns each channel's metadata DataChannel to one viewer. Close the
operator viewer before this active gate, then refresh it after the gate closes. Running two
viewers is not a valid metadata test.

```bash
python3 -m pip install websocket-client
python3 stress/app16_insight_visual_gate.py \
  --cdp-host 127.0.0.1 --cdp-port 9222 \
  --viewer-url 'https://192.168.1.9:8081/static/viewer.html?mode=light&src=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23' \
  --channel-ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 \
  --layout 24 --expected-fps 20 --width 1280 --height 720 \
  --identity-manifest /root/.simaai/neat-insight/media/app16/identity-24x20/identity-manifest.json \
  --output-prefix runs/app16-visual-24x20
```

Omit `--identity-manifest` only when validating unmarked customer media; that still proves that
all tiles decode fresh moving pixels and live overlays, but it does not prove channel-to-media
uniqueness.

The gate writes a screenshot and machine-readable JSON.

For an authoritative 30-minute temporal/synchronization gate, let the dedicated target own the
single metadata DataChannel for the full run. The 31-minute sample window below leaves enough
margin to start the independent 30-minute read-only observer after every peer appears:

```bash
python3 stress/app16_insight_visual_gate.py \
  --cdp-host 127.0.0.1 --cdp-port 9222 \
  --viewer-url 'https://192.168.1.9:8081/static/viewer.html?mode=light&src=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23' \
  --channel-ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 \
  --layout 24 --expected-fps 20 --width 1280 --height 720 \
  --identity-manifest /root/.simaai/neat-insight/media/app16/identity-24x20/identity-manifest.json \
  --wait-seconds 30 --sample-seconds 1860 --temporal-samples 373 \
  --output-prefix runs/app16-temporal-sync-24x20-30m
```

Run that command in the background, wait until Insight reports all 24 peers, and start the
read-only gate below while the dedicated target remains open. Do not open an operator viewer at
the same time. The read-only gate then observes this same dedicated browser peer without owning a
second DataChannel. The controlled gate closes its target afterward; refresh the operator viewer
only after both commands exit.

### 2. Read-only 30-minute stability gate

Have exactly one browser peer open for all 24 channels: either the controlled temporal gate above
or an operator viewer. Run `stress/app16_insight_stability_gate.py` on the Insight host. It issues
only GET requests to Insight's ingest/egress stats APIs; it never calls `/offer`, creates a WebRTC
peer, or steals metadata ownership. Every five-second sample requires all channels to advance at
Insight ingest and at the existing browser, including browser-decoded frames and metadata
messages.
Those read-only APIs expose counters, not metadata payload PTS or decoded video pixels. This
gate proves continuity and rate only; it must not be cited as proof of forward-only content or
box/video synchronization. Use the controlled temporal gate above for those claims.

```bash
python3 stress/app16_insight_stability_gate.py \
  --base-url https://127.0.0.1:9900 \
  --channel-ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 \
  --duration-seconds 1800 --interval-seconds 5 \
  --expected-video-fps 20 --expected-metadata-fps 20 \
  --minimum-fps-ratio 0.9 --minimum-median-fps-ratio 0.95 --max-bad-samples 0 \
  --output-prefix runs/app16-stability-24x20-30m
```

It writes one JSONL record per interval plus a summary JSON. The summary includes each
channel's minimum, p05, median, p95, maximum, and rate-miss count for browser video, Insight
ingest metadata, and browser metadata. A pass requires the full duration, no bad sample, and
at least 95% of the requested rate at each channel's median. The 90% per-sample floor tolerates
five-second browser-report quantization; it does not redefine an 18 FPS run as a 20 FPS result.
Use the reported distributions to show that observed rates are actually near 20. Because this
gate observes the existing viewer, an initial `missing_operator_peers` error means the dedicated
target/operator viewer is absent, stale, or does not include all requested channels.

## Tests

The supported automated tests are the C++ and Python unit tests plus the two Insight gates
above. Old debug-frame E2E tests and profile-driven experimental stress harnesses were removed
because their CLI/config contract no longer matched the validated C++ application.

The C++ unit tests cover configuration, CPU AU capacity/overflow, retry ordering, delayed
metadata matching, and realistic PTS-reset interleaving. Core and Internals carry the focused
encoded-callback lifetime, PTS restoration, terminal decoder-loan release, and stage-queue
tests.

## Python scope

`src/python/main.py` remains a small graph-native API reference. It resolves config-relative
assets and shares basic config validation, but it does **not** implement the validated C++
encoded-AU delay/PTS-epoch synchronization or the proven 24-channel dispatch architecture.
Do not use the Python path for the 24×20 handoff or Insight acceptance result.

## Source files

- Validated C++ app: `src/cpp/main.cpp`
- CPU delivery queue: `src/cpp/encoded_delivery_queue.h`
- Default profile/assets: `src/common/`
- C++ unit tests: `tests/cpp/test_unit.cpp`
- Python reference/tests: `src/python/main.py`, `tests/python/test_unit.py`
- Supported live gates: `stress/app16_insight_visual_gate.py`,
  `stress/app16_insight_stability_gate.py`
- Optional identity fixture generator: `stress/app16_make_identity_fixtures.py`
