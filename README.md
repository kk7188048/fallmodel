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
