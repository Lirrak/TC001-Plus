#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def box_iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix, iy = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix), max(0.0, iy2 - iy)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return 0.0 if union <= 0 else inter / union


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int((len(values) - 1) * p)))
    return float(values[idx])


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"WARNING: skipped invalid JSON line {line_no}: {exc}")
    return records


def analyze(path: Path, duplicate_iou: float, large_area_ratio: float) -> int:
    records = load_records(path)
    observations: list[dict[str, Any]] = []
    per_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    duplicate_pairs = []
    large_boxes = []
    frame_resets = []
    previous_frame = None
    configs = []
    detector_primary_candidates = []
    detector_final_after_nms = []
    reidentified_tracks = 0

    for record in records:
        frame_index = int(record.get("frame_index", -1))
        if previous_frame is not None and frame_index < previous_frame:
            frame_resets.append((previous_frame, frame_index))
        previous_frame = frame_index
        if record.get("config"):
            configs.append(record["config"])
        detector_debug = (record.get("summary") or {}).get("detector_debug") or {}
        reidentified_tracks += int((record.get("summary") or {}).get("reidentified_tracks") or 0)
        if isinstance(detector_debug.get("primary_candidates"), int):
            detector_primary_candidates.append(int(detector_debug["primary_candidates"]))
        if isinstance(detector_debug.get("final_after_nms"), int):
            detector_final_after_nms.append(int(detector_debug["final_after_nms"]))
        digital_size = record.get("digital_size") or [0, 0]
        frame_area = float(digital_size[0] * digital_size[1]) if len(digital_size) == 2 else 0.0
        for track in record.get("tracks", []):
            item = dict(track)
            item["frame_index"] = frame_index
            if not item.get("area_ratio") and frame_area > 0 and item.get("bbox"):
                item["area_ratio"] = float(item["bbox"][2] * item["bbox"][3]) / frame_area
            observations.append(item)
            per_track[int(track["track_id"])].append(item)

            area_ratio = float(item.get("area_ratio") or 0.0)
            if area_ratio >= large_area_ratio:
                large_boxes.append((frame_index, track.get("track_id"), track.get("source"), area_ratio, track.get("bbox")))

        tracks = record.get("tracks", [])
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                ov = box_iou(tracks[i]["bbox"], tracks[j]["bbox"])
                if ov >= duplicate_iou:
                    duplicate_pairs.append(
                        (
                            frame_index,
                            tracks[i].get("track_id"),
                            tracks[j].get("track_id"),
                            tracks[i].get("source"),
                            tracks[j].get("source"),
                            round(ov, 3),
                            tracks[i].get("bbox"),
                            tracks[j].get("bbox"),
                        )
                    )

    frames_with_tracks = sum(1 for record in records if record.get("tracks"))
    track_ids = sorted(per_track)
    track_lengths = {track_id: len(items) for track_id, items in per_track.items()}
    short_tracks = [track_id for track_id, length in track_lengths.items() if length <= 2]
    tracks_per_frame = [len(record.get("tracks", [])) for record in records]
    multi_track_frames = sum(1 for count in tracks_per_frame if count >= 2)
    sources = Counter(item.get("source") for item in observations)
    last_sources = Counter(item.get("last_detector_source") for item in observations)
    statuses = Counter(item.get("status") for item in observations)
    confirmed = Counter(bool(item.get("confirmed")) for item in observations)
    person_temps = [float(item["person_temp_c"]) for item in observations if isinstance(item.get("person_temp_c"), (int, float))]
    max_temps = [float(item["max_temp_c"]) for item in observations if isinstance(item.get("max_temp_c"), (int, float))]
    temps = person_temps or max_temps
    area_ratios = [float(item.get("area_ratio") or 0.0) for item in observations]
    contaminated = sum(1 for item in observations if item.get("roi_temp_contaminated"))

    print(f"File: {path}")
    print(f"Records: {len(records)}")
    if records:
        print(f"Frame range: {records[0].get('frame_index')} -> {records[-1].get('frame_index')}")
    if frame_resets:
        print(f"Frame resets detected: {len(frame_resets)}; file likely contains appended sessions. Sample: {frame_resets[:5]}")
    if configs:
        first_config = configs[0]
        interesting = {
            key: first_config.get(key)
            for key in (
                "digital_source",
                "face_model",
                "detector",
                "max_faces",
                "cascade_fallback",
                "head_fallback",
                "max_box_overlap",
            )
            if key in first_config
        }
        print(f"Config sample: {interesting}")
    print(f"Frames with tracks: {frames_with_tracks}; empty frames: {len(records) - frames_with_tracks}")
    print(f"Observations: {len(observations)}; unique track IDs: {len(track_ids)}")
    if track_ids:
        print(f"Track ID range: {track_ids[0]} -> {track_ids[-1]}")
    print(f"Tracks per frame: {dict(sorted(Counter(tracks_per_frame).items()))}")
    print(f"Sources: {dict(sources)}")
    print(f"Last detector sources: {dict(last_sources)}")
    print(f"Statuses: {dict(statuses)}")
    print(f"Confirmed: {dict(confirmed)}")
    print(f"Frames with >=2 tracks: {multi_track_frames}; max tracks/frame: {max(tracks_per_frame) if tracks_per_frame else 0}")
    print(f"Reidentified tracks: {reidentified_tracks}")
    print(f"ROI temp contaminated observations: {contaminated}")
    print(f"Short tracks <=2 observations: {len(short_tracks)} ({len(short_tracks) / max(len(track_ids), 1):.1%})")
    print(f"Duplicate pairs IoU>={duplicate_iou:.2f}: {len(duplicate_pairs)}")
    print(f"Large boxes area_ratio>={large_area_ratio:.2f}: {len(large_boxes)}")

    if temps:
        print(f"Person Temp C: min={min(temps):.2f}, median={statistics.median(temps):.2f}, max={max(temps):.2f}, n={len(temps)}")
    if max_temps:
        print(f"ROI Max Temp C: min={min(max_temps):.2f}, median={statistics.median(max_temps):.2f}, max={max(max_temps):.2f}, n={len(max_temps)}")
    if area_ratios:
        print(
            "Area ratio: "
            f"min={min(area_ratios):.3f}, p50={percentile(area_ratios, 0.50):.3f}, "
            f"p90={percentile(area_ratios, 0.90):.3f}, max={max(area_ratios):.3f}"
        )
    if detector_primary_candidates:
        print(
            "Detector primary candidates/frame: "
            f"max={max(detector_primary_candidates)}, "
            f"median={statistics.median(detector_primary_candidates):.1f}"
        )
    if detector_final_after_nms:
        print(
            "Detector final boxes after NMS/frame: "
            f"max={max(detector_final_after_nms)}, "
            f"median={statistics.median(detector_final_after_nms):.1f}"
        )

    long_tracks = sorted(track_lengths.items(), key=lambda item: item[1], reverse=True)[:10]
    print(f"Longest tracks: {long_tracks}")
    if duplicate_pairs:
        print("Duplicate sample:")
        for pair in duplicate_pairs[:10]:
            print(f"  {pair}")
    if large_boxes:
        print("Large box sample:")
        for box in large_boxes[:10]:
            print(f"  {box}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze TC001 Plus detection debug JSONL.")
    parser.add_argument("jsonl", type=Path, help="Path to detections_debug.jsonl")
    parser.add_argument("--duplicate-iou", type=float, default=0.30, help="IoU threshold for duplicate box report. Default: 0.30")
    parser.add_argument("--large-area-ratio", type=float, default=0.20, help="Area ratio threshold for large box report. Default: 0.20")
    args = parser.parse_args()

    if not args.jsonl.exists():
        print(f"ERROR: file not found: {args.jsonl}")
        return 2
    return analyze(args.jsonl, args.duplicate_iou, args.large_area_ratio)


if __name__ == "__main__":
    raise SystemExit(main())
