# Fall Detection

Pose-based fall detection pipeline: video → MediaPipe keypoints → normalized
features → sliding windows → BiLSTM + attention classifier.

## Meta-rule

Update this README after each milestone (M1, M2, M3, ...) with what was
completed, what decisions were made, and what's next.

## Milestones

- **M1** — dataset inventory / split decision (`src/falldet/inventory.py`)
- **M2** — pose extraction pipeline: MediaPipe wrapper, normalization,
  features, windowing, orchestration (`src/falldet/pose.py`,
  `normalize.py`, `features.py`, `windowing.py`, `pipeline.py`)
- **M3** — model: dataset/loader, BiLSTM + attention, training loop,
  evaluation (`src/falldet/dataset.py`, `model.py`, `train.py`,
  `evaluate.py`)
- **M3.5** — experiment grid over configs (`scripts/run_experiment_grid.py`)

## Status

**M1 done.** Dataset: [Le2i fall detection](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia)
(Kaggle mirror of the Le2i UMR6306 release), 190 videos, 320x240 @ 25fps,
across six scenes:

| Scene | Videos | Labels |
|---|---|---|
| Coffee_room_01 | 48 | fall (per-frame annotated) |
| Coffee_room_02 | 22 | fall (per-frame annotated) |
| Home_01 | 30 | fall (per-frame annotated) |
| Home_02 | 30 | fall (per-frame annotated) |
| Lecture_room | 27 | ADL, unannotated |
| Office | 33 | ADL, unannotated |

Real-data quirks `src/falldet/inventory.py` handles:
- `Annotation_files` vs. `Annotations_files` (typo in Coffee_room_02 only)
- `Lecture_room`/`Office` ship no annotation folder at all and no `Videos/`
  subfolder - every video in the annotated scenes contains exactly one fall
  (no 0/0 headers found), so these two scenes are treated as pure ADL.
- Annotation row format is `frame,activity_code,x1,y1,x2,y2` - confirmed by
  checking coordinate ranges against the 320x240 frame, not by the
  dataset's own (misleadingly worded) README.

**Split decision:** stratified per (scene, label) group at the video level,
not a full scene holdout - with only 6 scenes (4 fall-only, 2 ADL-only) a
scene holdout would either drop an entire environment from training or
break class balance. `configs/default.yaml` train/val/test ratios apply
within each group so every split sees every scene and every class.

Verified end-to-end against a real (partial) mirror of the dataset in
`data/raw/` — see `scripts/download_le2i.py` to fetch it yourself
(`--annotations_only` for a lightweight ~1MB pull, no flag for the full
~17GB video set; both require a Kaggle API key at `~/.kaggle/kaggle.json`).

**M2 done.** Pose extraction pipeline (`src/falldet/pose.py`,
`normalize.py`, `features.py`, `windowing.py`, `pipeline.py`):

- `pose.py` wraps MediaPipe's legacy `solutions.pose` API (33 landmarks,
  x/y/z/visibility per frame). **Pinned to `mediapipe==0.10.21`** —
  mediapipe>=1.0 replaced this with a Tasks-based `PoseLandmarker` whose
  `TensorsToDetectionsCalculator` crashes on macOS (`Metal GraphService
  unavailable`) even when forced to the CPU delegate; confirmed on this
  machine (Apple Silicon) before pinning down.
- `normalize.py` hip-centers and torso-scales every frame (position/
  camera-distance invariant across the six Le2i scenes), and linearly
  interpolates frames with no detection (occlusion/motion-blur during the
  fall itself is exactly when detections are most likely to drop — an
  interpolated frame beats losing it from the window).
- `features.py` builds a 167-dim per-frame vector: normalized xy (66) +
  visibility (33) + xy velocity (66) + torso angle vs. vertical (1) +
  torso angular velocity (1). Torso angle is 0 when upright, ~90° when
  lying flat — the single strongest per-frame fall signal.
- `windowing.py` slides fixed-size windows (`configs/default.yaml:
  windowing.window_size/stride`) over the feature sequence and labels a
  window 1 if it overlaps the annotated fall interval at all (0 for ADL
  videos, including the two unannotated scenes).
- `pipeline.py: process_video()` chains all of the above; `scripts/
  run_extraction.py` runs it over every video in `data/inventory.json`
  and writes one `.npz` (windows/labels/frame_ranges) per video to
  `data/processed/`.

Verified against the same real 6-video Le2i sample as M1: e.g.
`Coffee_room_01/video (44)` (fall_start=54, fall_end=86 of 133 frames,
99% pose detection rate) produced 7 windows correctly labeled
`[0,0,1,1,1,1,0]`.

**M3 built, not yet trained on the real dataset.** Model/training/eval
(`src/falldet/dataset.py`, `model.py`, `train.py`, `evaluate.py`):

- `dataset.py`: `FallWindowDataset` concatenates the per-video `.npz`
  windows for one split; `make_class_weights()` inverse-weights the loss
  since falls are a small fraction of frames even inside fall-labeled
  videos.
- `model.py`: BiLSTM over the 167-dim per-frame features, with a learned
  attention-pooling head (`AttentionPooling`) instead of last-hidden-state
  or mean pooling, so the classifier weighs the frames that actually look
  like a fall rather than averaging them out over a mostly-idle window.
- `train.py`: Adam + weighted cross-entropy, early stopping on val loss,
  MLflow param/metric logging and a best-checkpoint artifact. **Note:**
  mlflow's plain-directory filestore is now in maintenance mode and
  refuses to open by default — `configs/default.yaml` points
  `mlflow_uri` at `sqlite:///mlruns.db` instead.
- `evaluate.py`: classification report, confusion matrix, PR-AUC on the
  test split via sklearn.
- `scripts/run_training.py` chains train → evaluate and prints both.

Verified two ways since the local machine has no GPU and only 6 real
sample videos (not enough to populate val/test — M1's per-scene split
puts each scene's lone video into train):
1. **Synthetic separable data** (train/val/test all populated): full
   loop runs, early-stopping and checkpointing work, and the model
   correctly reaches 100% test accuracy / PR-AUC on the separable
   signal — confirms the training/eval wiring is correct.
2. **Real Le2i data** (train-only, 67 windows across the 6 sample
   videos): trains without crashing, loss decreases epoch over epoch,
   evaluation is correctly skipped with a clear message when a split is
   empty rather than silently reporting garbage metrics.

**Update**: trained end to end on the full real 190-video Le2i dataset,
locally on CPU (no GPU needed — small model, finished in minutes). Results,
threshold-tuning, and every subsequent run's config/metrics are tracked in
[`EXPERIMENTS.md`](EXPERIMENTS.md) — check there before starting a new
run, and add an entry after one that changes config, data, or code.

`scripts/threshold_sweep.py` scores an existing checkpoint at every
decision threshold instead of the default argmax (0.5) — useful since
fall is a rare class and the right recall/precision tradeoff is a
judgment call, not something a fixed 0.5 cutoff should decide for you.

There is also `notebooks/fall_detection_colab.ipynb` — a single,
self-contained notebook (every module inlined as a cell) for running this
on a hosted GPU notebook service instead of locally.

## Layout

See `configs/default.yaml` for all hyperparameters and paths — nothing
should be hardcoded in code.

## Usage

```bash
python scripts/run_inventory.py --raw_dir <path>
python scripts/run_extraction.py
python scripts/run_training.py
python scripts/run_experiment_grid.py
```

## M5 - serving (API + Docker)

`api/` exposes a `/predict` endpoint that reuses `src/falldet/pipeline.py`'s
`process_video()` completely unchanged from training - `api/inference.py`
pulls every pose/windowing parameter straight from the deployed
checkpoint's own saved config, so there is no separate config file that
could drift out of sync with what the model actually trained on.
Errors are centralized in `api/errors/` (typed exceptions + one FastAPI
handler any endpoint can reuse) rather than scattered per-route.

### Run locally

```bash
pip install -r requirements-serve.txt
python scripts/run_api.py
# in another terminal:
curl -F "file=@path/to/clip.mp4" http://localhost:8000/predict
```

### Run in Docker

Tested end-to-end on this machine (Apple Silicon, Docker Desktop) exactly
as written below - not from memory:

```bash
docker build -f docker/Dockerfile -t falldet .
docker run -d -p 8000:8000 --name falldet-test falldet
curl http://localhost:8000/health
curl -F "file=@path/to/clip.mp4" http://localhost:8000/predict
```

**Real things this surfaced, not just theory:**
- `mediapipe==0.10.21` (what `requirements.txt`/local dev use) has **no
  published wheel for `linux/aarch64`** - the actual build failed with
  `ERROR: Could not find a version that satisfies the requirement
  mediapipe==0.10.21`. `requirements-serve.txt` pins `mediapipe==0.10.18`
  instead (the newest 0.10.x with a wheel for this platform); the legacy
  `solutions.pose` API used by `pose.py` is unchanged across 0.10.x patch
  releases, confirmed by comparing predictions below.
- Confidence on a known test video came back **0.9704** in Docker vs.
  **0.9700** locally - a real, tiny, and expected discrepancy from the
  mediapipe patch-version difference (0.10.18 vs 0.10.21), not a bug. Same
  verdict (`fall`), same window count (64).
- Image is ~3GB - mediapipe pulls in `jax`/`jaxlib`/`scipy` as its own
  transitive dependencies, which dominate the size; not something this
  project's own code controls.
- `libgl1`/`libglib2.0-0` are required at runtime (not just build time) or
  OpenCV/MediaPipe fail with `ImportError: libGL.so.1` - both stages of
  the Dockerfile install them for this reason.

If building on `linux/amd64` instead (e.g. a typical cloud VM, not Apple
Silicon), `mediapipe==0.10.21` likely does have a wheel there and
`requirements-serve.txt` could be updated to match `requirements.txt`
exactly - untested on that platform from here.

### Live webcam demo (WebSocket)

`api/ws.py` exposes `/ws/predict` for real-time inference; `static/ws_client.html`
(served at `/demo`) is a bare browser harness around it - `getUserMedia()` for the
webcam, sends JPEG frames at 15fps, shows the live prediction as an overlay.

```bash
python scripts/run_api.py
# then open http://localhost:8000/demo in a browser and allow camera access
```

**The model is bidirectional, so it needs the full window before it can predict
at all** - live detection has a built-in lag of `window_size` frames at whatever
rate the client sends them (currently 45 frames @ 15fps sent =~ 3 seconds before
the first prediction, and every prediction after that reflects events up to
roughly one window-length ago). This is a structural property of the
architecture, stated here rather than left for someone to discover and assume
is a bug.

**Verified for real** (not just unit tests): streamed a real fall video frame by
frame through a live `/ws/predict` connection - confidence climbed to ~0.97
exactly during the real fall interval (frames 211-238), matching `/predict`'s
own aggregated result (0.9704) on the same clip almost exactly, then decayed as
the sliding window moved past the event - the same qualitative pattern found
during M4's error analysis and the original `run_demo.py` check, now reproduced
live over a real websocket connection.

**Recording an actual demo clip (webcam) is a "you, not Claude Code" step** -
needs a real camera and someone acting out a fall + normal movement, which
can't be done from here.

### Monitoring backend: Postgres (or CSV fallback)

`api/monitoring.py` logs every inference to Postgres if `FALLDET_DATABASE_URL`
is set, else falls back to the CSV file used earlier in development. Postgres
fixes the one genuine correctness gap in the CSV approach: multiple uvicorn
workers appending to the same file with plain `open(..., "a")` is a real race
condition, not just "less proper."

```bash
export FALLDET_DATABASE_URL="postgresql://user:pass@host/db?sslmode=require"
python scripts/run_api.py
# ... send some /predict requests ...
python scripts/plot_drift.py   # reads from Postgres automatically when the env var is set
```

**Isolation note**: if pointing this at a shared/existing Postgres instance,
the app creates and only ever touches its own schema (`falldet.inference_log`)
- verified live against a real shared Neon database that had 15 unrelated
tables (a Nakama game-backend schema, 43 real user rows) already in `public`:
sent real predictions through, confirmed the new rows landed correctly in
`falldet.inference_log`, and re-queried `public.users`/`user_device`/
`leaderboard_record` afterward to confirm zero rows changed there.

**Never commit a real `FALLDET_DATABASE_URL`** (or any credential) into this
repo or into chat with an assistant - `.env`/`.env.*` are gitignored for this
reason; set it as a real environment variable or secret instead.

### MLOps additions: CI, alerting, upload limits, auth, model registry

- **CI** (`.github/workflows/ci.yml`): runs the full pytest suite on every push/PR to `main`.
- **Alerting** (`scripts/plot_drift.py --alert_threshold X`): exits 1 and prints a
  warning to stderr if the latest rolling-mean confidence drops below `X` - cheap,
  automatable (cron/CI) drift signal, explicitly NOT real alerting infrastructure
  (no paging, no dashboards).
- **Upload limits** (`FALLDET_MAX_UPLOAD_BYTES`, default 200MB): `/predict` reads
  uploads in 1MB chunks and rejects (413) as soon as the limit is crossed, rather
  than buffering an arbitrarily large file into memory first.
- **API key auth** (`FALLDET_API_KEY`): if set, `/predict` requires a matching
  `X-API-Key` header (401 otherwise). Unset by default (open), matching every
  earlier assumption in this project. Doesn't cover `/ws/predict` - the browser
  `WebSocket` API can't send custom headers at all, so that endpoint stays
  unauthenticated (it's already documented as a demo harness, not production).
- **Model registry** (`api/registry.py`, `scripts/register_model.py`): a Postgres
  table (`falldet.models`) mapping a tag (e.g. `production`) to a checkpoint's
  file path + sha256 hash + key config fields. `FALLDET_MODEL_TAG` resolves the
  model to serve via this registry instead of a hardcoded `FALLDET_MODEL_PATH`.
  Deliberately minimal - "one active checkpoint per tag," not a full
  versioning/rollback/promotion system.

```bash
python scripts/register_model.py --checkpoint checkpoints_window45/best_model.pt --tag production
python scripts/register_model.py --list
FALLDET_MODEL_TAG=production FALLDET_DATABASE_URL=... python scripts/run_api.py
```

All five verified live: registered the real production model, confirmed
`/health` reports the registry-resolved absolute path, confirmed a real
prediction through that path matches every earlier run exactly
(confidence 0.9700656533241272 on the same test video).
