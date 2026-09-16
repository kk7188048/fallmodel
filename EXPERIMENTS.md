# Experiment log

Running history of training runs and tuning decisions. Update after each
run that changes config, data, or code in a way that affects results.

## 2026-09-13 — baseline, full Le2i dataset, local CPU

**Setup:** full 190-video Le2i dataset downloaded locally (`scripts/download_le2i.py`),
default `configs/default.yaml` (window_size=30, stride=15, hidden_dim=128,
num_layers=2, bidirectional, dropout=0.3, lr=3e-4, batch_size=32, CPU
training — no GPU used, still finished in a few minutes).

**Data:** 4,781 windows total (335 fall / 4,446 ADL) across train=133,
val=28, test=29 videos (67/66, 14/14, 15/14 fall/adl respectively).

**Result:** early stopped at epoch 29, best val_loss 0.2256.

Test set (636 windows, argmax/0.5 threshold):

| | precision | recall | f1 |
|---|---|---|---|
| ADL | 0.98 | 0.90 | 0.94 |
| Fall | 0.42 | 0.85 | 0.57 |

Accuracy 0.89, PR-AUC 0.58. Confusion matrix `[[TN=522, FP=61], [FN=8, TP=45]]`.

**Takeaway:** recall is good (85%, only 8/53 falls missed) but precision is
weak (61 false alarms) — expected given only 335 positive windows total.
Checkpoint: `checkpoints/best_model.pt`.

### Threshold sweep (`scripts/threshold_sweep.py`, same checkpoint, test split)

No retraining - just moving the decision threshold off the default 0.5:

| threshold | precision | recall | f1 |
|---|---|---|---|
| 0.35 | 0.32 | 0.93 | 0.48 |
| 0.50 (default) | 0.43 | 0.85 | 0.57 |
| **0.55** | **0.45** | 0.81 | **0.58 (best F1)** |
| 0.75 | 0.51 | 0.68 | 0.58 |
| 0.90 | 0.58 | 0.53 | 0.55 |

- Best F1 at **threshold=0.55** (barely above default, marginal gain).
- If prioritizing recall (fewer missed falls, safety-first): **threshold=0.35**
  gets recall to 0.93 (only ~4/53 missed) at the cost of precision dropping
  to 0.32 (more false alarms).
- Threshold tuning alone can't close the gap much further — the real
  bottleneck is too few positive (fall) windows, not the decision boundary.

**Next things to try** (see chat for full list): smaller stride specifically
around annotated fall intervals to generate more positive windows without
new data; heavier class weighting; `run_experiment_grid.py` hyperparameter
sweep; more fall videos if available.

## 2026-09-13 — stride=5 (was 15) — promoted to new default

**Hypothesis:** a smaller sliding-window stride gives more (denser,
overlapping) examples of each fall event without needing new video data.
Config: `configs/exp_stride5.yaml`, same as baseline except
`windowing.stride: 5`. Note: this scales both classes proportionally
(positive windows 335->998, but the positive/total ratio stayed ~7% in
both runs) - so this tests whether more overlapping crops help the model
generalize, not whether it fixes class imbalance.

**Data:** 14,160 windows (998 fall / 13,162 ADL), same train/val/test
video split as baseline (only windowing changed, not which videos are in
which split).

**Result:** early stopped at epoch 14, best val_loss 0.2253 (~same as
baseline's 0.2256).

| | baseline (stride=15) | stride=5 |
|---|---|---|
| PR-AUC | 0.585 | **0.640** |
| Best-threshold F1 | 0.581 (t=0.55) | **0.653** (t=0.75) |
| → precision / recall at that threshold | 0.45 / 0.81 | **0.59 / 0.73** |
| Precision at recall >= 0.9 | 0.32 (t=0.35) | **0.36** (t=0.45) |

**Verdict: promoted to default.** Better PR-AUC and better F1 at every
recall level tested, with the same amount of source video. `configs/default.yaml`
now uses `stride: 5`. Old baseline artifacts kept for reference at
`data/processed_stride15_baseline/`, `checkpoints_stride15_baseline/`,
`mlruns_stride15_baseline.db`.

**Still true from baseline:** recall/precision tradeoff is still a choice,
not solved - use `scripts/threshold_sweep.py --min_recall <x>` to pick an
operating point. Absolute fall-window count (998) is still small for deep
learning; more source video would likely help more than further stride
tuning.

## 2026-09-15 — window_size=45 (was 30), testing the M4 "truncated context" hypothesis

**Hypothesis (from M4 error analysis):** 80% of false negatives (20/25) had their
highest-attention frame within 2 frames of the window boundary, suggesting the model
wasn't seeing the full fall event. Prediction: a longer window should reduce missed
falls.

**Setup:** `configs/exp_window45.yaml` - window_size=45 (was 30), stride=5 unchanged,
same model/train hyperparameters. Re-extracted all 190 videos (13,590 windows, 1,266
fall / 9.3% - up from 7.0% at window=30, a nice side effect of longer windows
naturally overlapping fall intervals more often).

**Result:** early stopped epoch 13, best val_loss 0.2761 (worse than window=30's
0.2253 - some evidence of a harder optimization problem with longer sequences).

| | window=30 (current default) | window=45 |
|---|---|---|
| PR-AUC | 0.640 | 0.656 |
| Best F1 | 0.653 (t=0.75) | 0.668 (t=0.80) |
| Precision/recall at best F1 | 0.59 / 0.73 | 0.69 / 0.65 |
| Precision at recall>=0.9 | 0.36 (t=0.45) | 0.40 (t=0.45) |
| Recall at threshold=0.5 | 0.879 | 0.863 (slightly worse) |

**Verdict: modest, mixed improvement - NOT the clean fix the edge-attention
statistic predicted.** PR-AUC and best-F1 both improved a little; precision at a
matched high-recall (0.9) operating point improved meaningfully (0.36->0.40). But at
the default threshold, recall got slightly *worse*, and window=45's own best-F1
threshold trades recall down to 0.65 (worse than window=30's 0.73 at its best-F1
point) in exchange for higher precision. This is the opposite tradeoff direction
from what the safety framing wants if you're optimizing for "don't miss falls."

**Takeaway for the truncated-context theory:** probably a real contributing factor,
not the dominant cause of false negatives - window=45 gives longer context but
doesn't resolve the fundamental data-scarcity problem (only ~1,266 real positive
windows even now). Worth keeping as a genuine, if modest, improvement; not worth
overselling as "the fix."

**Decision on which model to deploy (M5): pending - see chat.**
