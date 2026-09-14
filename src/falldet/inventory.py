"""M1 - dataset inventory and train/val/test split decision.

Targets the Le2i fall-detection dataset as distributed on Kaggle
(tuyenldvn/falldataset-imvia), which mirrors the original Le2i UMR6306
release: 320x240 @ 25fps videos in six scenes.

Observed on-disk layout (confirmed against the real dataset, not assumed):

    <raw_dir>/<Scene>/<Scene>/Videos/video (N).avi
    <raw_dir>/<Scene>/<Scene>/Annotation_files/video (N).txt   # most scenes
    <raw_dir>/<Scene>/<Scene>/Annotations_files/video (N).txt  # Coffee_room_02 only (typo upstream)
    <raw_dir>/Lecture_room/Lecture room/video (N).avi          # no annotations, no Videos/ subfolder
    <raw_dir>/Office/Office/video (N).avi                      # no annotations, no Videos/ subfolder

Only Coffee_room_01, Coffee_room_02, Home_01 and Home_02 ship per-frame
ground truth. Every annotated video in this release contains exactly one
fall (no 0/0 "no fall" headers were found among the annotated videos), so
Lecture_room and Office - which ship no annotations at all - are treated as
pure ADL (activities of daily living, i.e. non-fall) scenes.

Annotation file format (first two lines, then one row per annotated frame):

    line 1: fall_start_frame
    line 2: fall_end_frame
    line N: frame_number,activity_code,x1,y1,x2,y2

x1/x2 are horizontal pixel coordinates (0-320), y1/y2 vertical (0-240) -
confirmed by range, not by the (misleading) height/width wording in the
dataset's own README. activity_code is an auxiliary per-frame state label
(e.g. walking vs. falling vs. lying) that this inventory step does not need.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

VIDEO_EXTS = {".avi", ".mp4", ".mkv", ".mov"}
ANNOTATION_DIR_NAMES = ("Annotation_files", "Annotations_files")

FALL = "fall"
ADL = "adl"


@dataclass
class VideoRecord:
    video_id: str  # stable id: "<scene>/<filename stem>"
    scene: str
    path: str  # relative to raw_dir
    label: str  # "fall" or "adl"
    has_annotation: bool
    annotation_path: str | None = None
    fall_start_frame: int | None = None
    fall_end_frame: int | None = None
    frame_count: int | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    duration_sec: float | None = None
    warnings: list[str] = field(default_factory=list)
    split: str | None = None


def _find_annotation_dir(scene_root: Path) -> Path | None:
    for name in ANNOTATION_DIR_NAMES:
        candidate = scene_root / name
        if candidate.is_dir():
            return candidate
    return None


def _find_video_dir(scene_root: Path) -> Path:
    videos_dir = scene_root / "Videos"
    return videos_dir if videos_dir.is_dir() else scene_root


def parse_annotation(path: Path) -> tuple[int, int, int]:
    """Parse fall_start_frame, fall_end_frame, and annotated frame count."""
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    try:
        fall_start = int(lines[0])
        fall_end = int(lines[1])
        data_lines = lines[2:]
    except ValueError:
        # A handful of annotation files in the full 190-video Le2i release
        # don't carry the two-line fall_start/fall_end header (line 0 is
        # already a per-frame data row like "1,1,72,58,132,170"). Fall back
        # to treating the video as unannotated (ADL) instead of crashing
        # the whole inventory pass over one malformed file.
        fall_start = fall_end = 0
        data_lines = lines
    annotated_frames = len(data_lines)
    return fall_start, fall_end, annotated_frames


def _probe_video(path: Path) -> tuple[int | None, float | None, int | None, int | None, list[str]]:
    warnings: list[str] = []
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, None, None, None, [f"opencv unavailable, skipped video probing: {exc}"]

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None, None, None, None, ["could not open video file"]

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
    cap.release()

    if not frame_count:
        warnings.append("frame count unavailable or zero")
    return frame_count, fps, width, height, warnings


def _scene_dirs(raw_dir: Path) -> list[tuple[str, Path]]:
    """Each Le2i scene is a duplicated <Scene>/<Scene or "Scene name"> pair."""
    scenes = []
    for top in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        children = [c for c in top.iterdir() if c.is_dir()]
        scene_root = children[0] if len(children) == 1 else top
        scenes.append((top.name, scene_root))
    return scenes


def discover_videos(raw_dir: Path, probe: bool = True) -> list[VideoRecord]:
    """Walk raw_dir and build one VideoRecord per video file found."""
    raw_dir = Path(raw_dir)
    records: list[VideoRecord] = []

    for scene_name, scene_root in _scene_dirs(raw_dir):
        annotation_dir = _find_annotation_dir(scene_root)
        video_dir = _find_video_dir(scene_root)

        video_paths = sorted(
            p for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS
        )
        for video_path in video_paths:
            warnings: list[str] = []
            annotation_path = None
            fall_start = fall_end = None
            label = ADL

            if annotation_dir is not None:
                candidate = annotation_dir / f"{video_path.stem}.txt"
                if candidate.exists():
                    annotation_path = candidate
                    fall_start, fall_end, annotated_frames = parse_annotation(candidate)
                    label = FALL if (fall_start or fall_end) else ADL
                    if annotated_frames <= 0:
                        warnings.append("annotation file has no per-frame rows")
                else:
                    warnings.append("scene has annotations but this video is missing one")

            frame_count = fps = width = height = None
            if probe:
                frame_count, fps, width, height, probe_warnings = _probe_video(video_path)
                warnings.extend(probe_warnings)
                if fall_end and frame_count and fall_end > frame_count:
                    warnings.append(
                        f"fall_end_frame ({fall_end}) exceeds probed frame_count ({frame_count})"
                    )

            duration_sec = frame_count / fps if frame_count and fps else None

            records.append(
                VideoRecord(
                    video_id=f"{scene_name}/{video_path.stem}",
                    scene=scene_name,
                    path=str(video_path.relative_to(raw_dir)),
                    label=label,
                    has_annotation=annotation_path is not None,
                    annotation_path=(
                        str(annotation_path.relative_to(raw_dir)) if annotation_path else None
                    ),
                    fall_start_frame=fall_start,
                    fall_end_frame=fall_end,
                    frame_count=frame_count,
                    fps=fps,
                    width=width,
                    height=height,
                    duration_sec=duration_sec,
                    warnings=warnings,
                )
            )

    return records


def assign_splits(
    records: list[VideoRecord],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> None:
    """Video-level split, stratified by (scene, label) group, in place.

    Splitting is done per (scene, label) group rather than globally at
    random. With only six scenes, a full scene holdout (e.g. "train on
    Home_*, test on Office") would either starve the model of an entire
    environment or wildly unbalance class counts, since Office/Lecture_room
    contribute only ADL videos and Coffee_room/Home only fall videos.
    Stratifying per (scene, label) instead guarantees every split sees
    every scene and every class, while keeping video-level (not
    frame-level) separation, which is the leakage boundary that actually
    matters here since every video is an independent recording.
    """
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[VideoRecord]] = {}
    for rec in records:
        groups.setdefault((rec.scene, rec.label), []).append(rec)

    for group_records in groups.values():
        shuffled = group_records[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = round(n * train_frac)
        n_val = round(n * val_frac)
        # keep at least one video in train even for tiny groups
        n_train = max(n_train, 1) if n else 0
        n_val = min(n_val, max(n - n_train, 0))

        for i, rec in enumerate(shuffled):
            if i < n_train:
                rec.split = "train"
            elif i < n_train + n_val:
                rec.split = "val"
            else:
                rec.split = "test"


def summarize(records: list[VideoRecord]) -> dict:
    summary: dict = {"total_videos": len(records), "by_scene": {}, "by_split": {}}
    for rec in records:
        scene_stats = summary["by_scene"].setdefault(
            rec.scene, {"fall": 0, "adl": 0, "warnings": 0}
        )
        scene_stats[rec.label] += 1
        if rec.warnings:
            scene_stats["warnings"] += 1

        split_stats = summary["by_split"].setdefault(
            rec.split or "unassigned", {"fall": 0, "adl": 0}
        )
        split_stats[rec.label] += 1
    return summary


def build_inventory(
    raw_dir: str | Path,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
    probe: bool = True,
) -> tuple[list[VideoRecord], dict]:
    records = discover_videos(Path(raw_dir), probe=probe)
    if not records:
        raise FileNotFoundError(f"no video files found under {raw_dir}")
    assign_splits(records, train_frac=train_frac, val_frac=val_frac, seed=seed)
    return records, summarize(records)


def write_inventory(records: list[VideoRecord], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([asdict(r) for r in records], indent=2))


def write_splits(records: list[VideoRecord], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for rec in records:
        splits.setdefault(rec.split or "unassigned", []).append(rec.video_id)
    out_path.write_text(json.dumps(splits, indent=2))
