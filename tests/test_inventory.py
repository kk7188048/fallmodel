"""M1 - dataset inventory and split logic, against a synthetic Le2i-shaped tree."""
from pathlib import Path

from falldet.inventory import ADL, FALL, assign_splits, build_inventory, discover_videos, parse_annotation


def _write_annotation(path: Path, fall_start: int, fall_end: int, n_frames: int) -> None:
    lines = [str(fall_start), str(fall_end)]
    for i in range(1, n_frames + 1):
        lines.append(f"{i},1,10,10,50,50")
    path.write_text("\n".join(lines))


def _make_le2i_tree(root: Path) -> None:
    # Coffee_room_01: annotated, "Annotation_files" + "Videos"
    scene = root / "Coffee_room_01" / "Coffee_room_01"
    (scene / "Videos").mkdir(parents=True)
    (scene / "Annotation_files").mkdir(parents=True)
    for i in range(1, 6):
        (scene / "Videos" / f"video ({i}).avi").write_bytes(b"fake")
        _write_annotation(scene / "Annotation_files" / f"video ({i}).txt", 10, 20, 30)

    # Coffee_room_02: annotated, but the upstream typo "Annotations_files"
    scene2 = root / "Coffee_room_02" / "Coffee_room_02"
    (scene2 / "Videos").mkdir(parents=True)
    (scene2 / "Annotations_files").mkdir(parents=True)
    for i in range(49, 53):
        (scene2 / "Videos" / f"video ({i}).avi").write_bytes(b"fake")
        _write_annotation(scene2 / "Annotations_files" / f"video ({i}).txt", 5, 15, 25)

    # Office: no annotations at all, videos directly in the scene dir (ADL)
    scene3 = root / "Office" / "Office"
    scene3.mkdir(parents=True)
    for i in range(1, 5):
        (scene3 / f"video ({i}).avi").write_bytes(b"fake")


def test_discover_videos_finds_all_scenes(tmp_path):
    _make_le2i_tree(tmp_path)
    records = discover_videos(tmp_path, probe=False)
    assert len(records) == 5 + 4 + 4

    by_scene = {}
    for r in records:
        by_scene.setdefault(r.scene, []).append(r)
    assert set(by_scene) == {"Coffee_room_01", "Coffee_room_02", "Office"}
    assert len(by_scene["Coffee_room_01"]) == 5
    assert len(by_scene["Coffee_room_02"]) == 4
    assert len(by_scene["Office"]) == 4


def test_annotation_typo_folder_is_found(tmp_path):
    _make_le2i_tree(tmp_path)
    records = discover_videos(tmp_path, probe=False)
    coffee2 = [r for r in records if r.scene == "Coffee_room_02"]
    assert all(r.has_annotation for r in coffee2)
    assert all(r.label == FALL for r in coffee2)
    assert all(r.fall_start_frame == 5 and r.fall_end_frame == 15 for r in coffee2)


def test_parse_annotation_missing_header_falls_back_to_adl(tmp_path):
    # Some real Le2i annotation files skip the fall_start/fall_end header
    # and start directly with a per-frame data row.
    path = tmp_path / "video (1).txt"
    path.write_text("1,1,72,58,132,170\n2,1,72,58,132,170\n3,1,72,58,132,170")
    fall_start, fall_end, annotated_frames = parse_annotation(path)
    assert fall_start == 0
    assert fall_end == 0
    assert annotated_frames == 3


def test_unannotated_scene_is_labeled_adl(tmp_path):
    _make_le2i_tree(tmp_path)
    records = discover_videos(tmp_path, probe=False)
    office = [r for r in records if r.scene == "Office"]
    assert all(r.label == ADL for r in office)
    assert all(not r.has_annotation for r in office)


def test_video_ids_are_unique(tmp_path):
    _make_le2i_tree(tmp_path)
    records = discover_videos(tmp_path, probe=False)
    ids = [r.video_id for r in records]
    assert len(ids) == len(set(ids))


def test_assign_splits_covers_every_record(tmp_path):
    _make_le2i_tree(tmp_path)
    records = discover_videos(tmp_path, probe=False)
    assign_splits(records, train_frac=0.6, val_frac=0.2, seed=0)
    assert all(r.split in {"train", "val", "test"} for r in records)


def test_assign_splits_every_scene_label_group_has_train(tmp_path):
    _make_le2i_tree(tmp_path)
    records = discover_videos(tmp_path, probe=False)
    assign_splits(records, train_frac=0.6, val_frac=0.2, seed=0)

    groups = {}
    for r in records:
        groups.setdefault((r.scene, r.label), []).append(r.split)
    for splits in groups.values():
        assert "train" in splits


def test_assign_splits_is_deterministic(tmp_path):
    _make_le2i_tree(tmp_path)
    records_a = discover_videos(tmp_path, probe=False)
    records_b = discover_videos(tmp_path, probe=False)
    assign_splits(records_a, seed=7)
    assign_splits(records_b, seed=7)
    a = {r.video_id: r.split for r in records_a}
    b = {r.video_id: r.split for r in records_b}
    assert a == b


def test_build_inventory_raises_on_empty_dir(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        build_inventory(tmp_path, probe=False)


def test_build_inventory_end_to_end(tmp_path):
    _make_le2i_tree(tmp_path)
    records, summary = build_inventory(tmp_path, probe=False, seed=1)
    assert summary["total_videos"] == len(records) == 13
    assert summary["by_scene"]["Coffee_room_01"]["fall"] == 5
    assert summary["by_scene"]["Office"]["adl"] == 4
    assert set(summary["by_split"]).issubset({"train", "val", "test", "unassigned"})
