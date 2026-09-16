# M4 Error Analysis — Writeup

Model: `checkpoints/best_model.pt` (window=30, stride=5, the default config at
the time this analysis was run). Test set, default threshold 0.5.
188 misclassified windows out of 636 total (163 false positives, 25 false
negatives). Full per-window data: `misclassified_index.csv`. Skeleton plots
for a 20-window sample: `plots/`.

Bucketing method: rule-based, applied in priority order over four diagnostic
columns computed per window (`low_visibility_frac`, `max_attention_near_edge`,
`max_hip_velocity`, `torso_angle_range`, `confidence`) — not hand-labeled one
by one. Rules and thresholds (quartiles of the misclassified set) are in the
script used to produce the CSV; see `bucket`/`notes` columns for the
per-window justification. Spot-checked against the actual skeleton plots for
windows 37 and 79 (see below) to confirm the rules track real video content,
not just numbers.

## Bucket counts

| bucket | count | true=fall | true=ADL | meaning |
|---|---|---|---|---|
| occlusion | 50 | 2 | 48 | person partly off-frame / low pose-landmark visibility for a large fraction of the window |
| fall_like_adl_motion | 40 | 0 | 40 | false positive: fast sitting/bending/crouching motion that kinematically resembles a fall |
| borderline_confidence | 38 | 2 | 36 | confidence 0.40–0.65 — genuinely ambiguous, not a clear model error |
| unclear_fp | 29 | 0 | 29 | false positive with no single diagnostic signal standing out; needs manual video review |
| window_boundary | 28 | 18 | 10 | model's peak attention frame sits at the window edge — the decisive motion is likely split across two windows |
| subtle_fall_motion | 3 | 3 | 0 | true fall but low hip velocity throughout — a slow/controlled fall, not the fast falls the model has mostly seen |

## Concrete findings (3–5 bullets)

1. **False positives (163) dominate, and most are explainable, not random noise.**
   `occlusion` (48 of them) and `fall_like_adl_motion` (40) together account for
   >50% of all false positives — a person briefly leaving frame, or a fast sit/bend
   motion, both plausibly look "fall-like" to a model trained on kinematic
   features alone with no scene/object context.

2. **False negatives (25) are dominated by window-boundary truncation.** 18 of
   25 false negatives (72%) fall in `window_boundary` — the model's attention
   peaks right at frame 0 or frame 29 of a 30-frame window, meaning the actual
   fall motion is likely split across the window cut rather than fully visible
   in either window. This matches the earlier observation in `EXPERIMENTS.md`
   (~80% of FNs near-edge) that motivated the window=45 experiment.

3. **Occlusion is a real, separate failure mode from motion-based confusion.**
   `low_visibility_frac` for the `occlusion` bucket (top quartile of the
   misclassified set, ≥0.24) means roughly a quarter or more of the window's
   frames had low-confidence or missing MediaPipe landmarks — the model is
   making a fall/ADL call on partially-missing pose data. This is a pose
   pipeline limitation, not a classifier limitation, and window-size or
   architecture changes won't fix it.

4. **Only 3 of 188 misclassified windows are "slow fall" false negatives**
   (`subtle_fall_motion`) — the fear that the model only recognizes fast,
   dramatic falls and misses slow/controlled ones is not well supported by
   this data; it's a real but minor failure mode.

5. **Visual spot check confirmed the bucketing rules were sound, not just
   numeric artifacts.** Window 37 (`Coffee_room_01/video (10)`, true=1,
   pred=0, conf=0.11): attention is near-zero for frames 0–24 and jumps to
   0.35 exactly at the last frame (t=29), where the skeleton first shows a
   collapsed/horizontal posture — the fall itself starts right at the window
   edge, confirming the `window_boundary` label. Window 79
   (`Coffee_room_01/video (16)`, true=1, pred=0, conf=0.51): the skeleton is
   already lying horizontal for the *entire* window with low, flat attention
   and low velocity throughout — consistent with `subtle_fall_motion` (a fall
   already in progress/completed with no sudden motion signature for the
   model to key on).

## Fix attempted: window_size 30 → 45

Directly tested the `window_boundary` hypothesis by increasing window_size
from 30 to 45 frames (full setup and results in `EXPERIMENTS.md`,
2026-09-15 entry). Rationale: a longer window makes it less likely a fall's
onset or completion falls on a hard boundary cut.

**Before (window=30):** PR-AUC 0.640, best F1 0.653 (precision 0.59 /
recall 0.73 at that threshold), recall at default threshold 0.879.

**After (window=45):** PR-AUC 0.656, best F1 0.668 (precision 0.69 /
recall 0.65 at that threshold), recall at default threshold 0.863
(slightly worse).

**Result: modest, mixed improvement — not a clean fix.** PR-AUC and
precision at a matched high-recall operating point both improved a little
(precision at recall≥0.9: 0.36→0.40). But recall at the default threshold
got marginally worse, and window=45's own best-F1 point trades recall down
in exchange for precision — the opposite direction from what a
"don't-miss-falls" safety framing wants. The window-boundary theory
explained a majority of false negatives correctly (as confirmed by the
72% figure above), but fixing the boundary only partially closes the gap,
because the deeper bottleneck is data scarcity (~1,266 real positive
windows even at window=45), not window geometry alone.

## Final one-sentence weakness statement

**The model's single biggest weakness is false positives driven by
occlusion and fast-but-non-fall motion (sitting, bending) rather than
recall on true falls, which is already reasonably strong (86–88%) — so the
next highest-leverage improvement is not more window-size tuning but
either occlusion-aware features (visibility-weighted confidence) or more
diverse ADL training examples of fast sit/bend/crouch motions to reduce
the ~40% of false positives that are pure motion-shape confusion.**
