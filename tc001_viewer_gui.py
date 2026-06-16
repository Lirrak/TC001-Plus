from __future__ import annotations

import argparse
import cv2
import json
import numpy as np
import os
import time
from typing import Optional, Tuple, List, Any

from tc001_face import FaceBox, FaceTrack, FaceTracker, DigitalFaceDetector
from tc001_sdk import TC001SdkCamera, TC001SdkFrame
from tc001_viewer_types import (
    WINDOW_NAME,
    COLORMAPS,
    ThermalFrame,
    ViewerState,
    Recorder,
    OpenOptions,
    Alignment,
    RoiTemperatureStats,
)
from tc001_viewer_utils import (
    timestamp,
    backend_candidates,
    fmt_temp,
    put_text,
    draw_crosshair,
    safe_stats,
    resize_to_window,
)
from tc001_viewer_core import (
    open_camera,
    decode_frame,
    describe_frame,
    map_points_to_thermal,
    face_box_points,
    orient_points,
    inverse_orient_points,
    polygon_temperature_stats_for_state,
    save_alignment,
    apply_orientation,
    make_heatmap,
    split_frame_if_needed,
    extract_digital_frame,
    orient_digital_frame,
)


def draw_one_face_overlay(
    img: np.ndarray,
    raw_thermal: ThermalFrame,
    oriented_thermal: ThermalFrame,
    digital_shape: Tuple[int, int],
    track: FaceTrack,
    alignment: Alignment,
    state: ViewerState,
) -> None:
    if track.box is None:
        return

    raw_thermal_shape = raw_thermal.temp_c.shape[:2] if raw_thermal.temp_c is not None else raw_thermal.display_source.shape[:2]
    mapped = map_points_to_thermal(face_box_points(track.box), digital_shape, raw_thermal_shape, alignment)
    if mapped is None:
        return

    mapped_oriented = orient_points(mapped, raw_thermal_shape, state)
    src_h, src_w = oriented_thermal.display_source.shape[:2]
    dst_h, dst_w = img.shape[:2]
    sx = dst_w / max(src_w, 1)
    sy = dst_h / max(src_h, 1)
    draw_pts = mapped_oriented.copy()
    draw_pts[:, 0] *= sx
    draw_pts[:, 1] *= sy
    pts_i = np.round(draw_pts).astype(np.int32).reshape(-1, 1, 2)

    cv2.polylines(img, [pts_i], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    status = track.status if track.status and track.status != "NO FACE" else "FACE"
    label = f"P{track.track_id} {status} {track.box.confidence * 100:.0f}%"
    temp_stats = polygon_temperature_stats_for_state(raw_thermal.temp_c, mapped, state)
    if temp_stats.person_temp_c is not None:
        label += f" | temp {fmt_temp(temp_stats.person_temp_c, state.unit, raw_thermal.approx_temps)}"
        if temp_stats.roi_temp_contaminated:
            label += " !hot"
    x = int(np.min(draw_pts[:, 0]))
    y = int(np.min(draw_pts[:, 1]))
    put_text(img, label, (max(8, x), max(22, y - 8)), scale=0.45, color=(0, 255, 0))


def draw_face_overlay(
    img: np.ndarray,
    raw_thermal: ThermalFrame,
    oriented_thermal: ThermalFrame,
    digital_shape: Tuple[int, int],
    tracker: FaceTracker,
    alignment: Alignment,
    state: ViewerState,
) -> None:
    for track in tracker.active_tracks(confirmed_only=True):
        draw_one_face_overlay(img, raw_thermal, oriented_thermal, digital_shape, track, alignment, state)


def draw_digital_debug(frame: np.ndarray, tracker: FaceTracker, alignment: Alignment, source: str, rotate: int) -> np.ndarray:
    out = frame.copy()
    tracks = tracker.active_tracks()
    confirmed_count = len([track for track in tracks if track.confirmed])
    candidate_count = len(tracks) - confirmed_count
    head_count = len([track for track in tracks if track.detector_name == "head"])
    face_count = len([track for track in tracks if track.detector_name in ("tasks", "mediapipe", "haar", "profile")])
    if tracks:
        for track in tracks:
            if track.box is None:
                continue
            x0, y0 = int(track.box.x), int(track.box.y)
            x1, y1 = int(track.box.x + track.box.w), int(track.box.y + track.box.h)
            if not track.confirmed:
                color = (0, 255, 255)
            elif track.status == "HELD":
                color = (0, 200, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
            status = f"CANDIDATE {track.status}" if not track.confirmed else track.status
            label = f"P{track.track_id} {status} {track.box.confidence * 100:.0f}%"
            if track.detector_name and track.detector_name not in ("none", "held"):
                label += f" ({track.detector_name})"
            put_text(out, label, (x0, max(20, y0 - 8)), scale=0.5, color=color)
    else:
        put_text(out, "NO FACE", (8, 24), scale=0.55, color=(0, 255, 255))
    put_text(
        out,
        f"confirmed {confirmed_count} | candidate {candidate_count} | face {face_count} | head {head_count}",
        (8, 48 if not tracks else 24),
        scale=0.42,
        color=(0, 255, 255),
    )
    put_text(out, f"Digital source: {source} | rotate {rotate} | {out.shape[1]}x{out.shape[0]} | alignment: {alignment.mode}", (8, out.shape[0] - 12), scale=0.42)
    return out


def build_detection_debug_record(
    frame_index: int,
    digital_shape: Tuple[int, int],
    tracker: FaceTracker,
    raw_thermal: ThermalFrame,
    alignment: Alignment,
    config: Optional[dict] = None,
    state: Optional[ViewerState] = None,
) -> dict:
    raw_thermal_shape = raw_thermal.temp_c.shape[:2] if raw_thermal.temp_c is not None else raw_thermal.display_source.shape[:2]
    tracks = []
    for track in tracker.active_tracks():
        if track.box is None:
            continue
        mapped = map_points_to_thermal(face_box_points(track.box), digital_shape, raw_thermal_shape, alignment)
        temp_stats = polygon_temperature_stats_for_state(raw_thermal.temp_c, mapped, state) if mapped is not None else None
        tracks.append(
            {
                "track_id": int(track.track_id),
                "source": track.detector_name,
                "last_detector_source": track.last_detector_source,
                "status": track.status,
                "confirmed": bool(track.confirmed),
                "hits": int(track.hits),
                "missed_frames": int(track.missed_frames),
                "age_frames": int(frame_index - track.first_seen_frame) if track.first_seen_frame else 0,
                "confidence": float(track.box.confidence),
                "bbox": [
                    round(float(track.box.x), 2),
                    round(float(track.box.y), 2),
                    round(float(track.box.w), 2),
                    round(float(track.box.h), 2),
                ],
                "area_ratio": round(float((track.box.w * track.box.h) / max(float(digital_shape[0] * digital_shape[1]), 1.0)), 4),
                "iou_with_existing": round(float(track.last_iou_with_existing), 4),
                "merged_from": track.merged_from,
                "thermal_polygon": None
                if mapped is None
                else [[round(float(x), 2), round(float(y), 2)] for x, y in mapped.tolist()],
                "person_temp_c": None if temp_stats is None or temp_stats.person_temp_c is None else round(float(temp_stats.person_temp_c), 2),
                "max_temp_c": None if temp_stats is None or temp_stats.max_temp_c is None else round(float(temp_stats.max_temp_c), 2),
                "median_temp_c": None if temp_stats is None or temp_stats.median_temp_c is None else round(float(temp_stats.median_temp_c), 2),
                "percentile_temp_c": None if temp_stats is None or temp_stats.percentile_temp_c is None else round(float(temp_stats.percentile_temp_c), 2),
                "hot_outlier_delta_c": None
                if temp_stats is None or temp_stats.hot_outlier_delta_c is None
                else round(float(temp_stats.hot_outlier_delta_c), 2),
                "hot_outlier_pixels": 0 if temp_stats is None else int(temp_stats.hot_outlier_pixels),
                "roi_pixels": 0 if temp_stats is None else int(temp_stats.roi_pixels),
                "used_pixels": 0 if temp_stats is None else int(temp_stats.used_pixels),
                "roi_temp_contaminated": False if temp_stats is None else bool(temp_stats.roi_temp_contaminated),
                "roi_temp_mode": None if temp_stats is None else temp_stats.mode,
            }
        )
    return {
        "frame_index": int(frame_index),
        "digital_size": [int(digital_shape[1]), int(digital_shape[0])],
        "alignment": alignment.mode,
        "config": config or {},
        "summary": tracker.last_stats,
        "tracks": tracks,
    }


def emit_detection_debug(record: dict, debug_console: bool, debug_file: Optional[Any]) -> None:
    if debug_console:
        brief = [
            f"P{track['track_id']}:{track['source']}:{track['last_detector_source']}:{track['status']}:"
            f"{'ok' if track['confirmed'] else 'cand'}:"
            f"bbox={track['bbox']}:temp={track.get('person_temp_c')}:max={track.get('max_temp_c')}"
            for track in record.get("tracks", [])
        ]
        summary = record.get("summary", {}) or {}
        stats = (
            f"new={summary.get('new_tracks', 0)} "
            f"merge={summary.get('merged_detections', 0)} "
            f"suppress={summary.get('duplicate_suppressed', 0)} "
            f"reject={summary.get('rejected_detections', 0)}"
        )
        print(f"[detections frame={record['frame_index']} {stats}] " + (" | ".join(brief) if brief else "none"))
    if debug_file is not None:
        debug_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        debug_file.flush()


def collect_four_points(window: str, image: np.ndarray, title: str) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    display = image.copy()

    def redraw() -> None:
        view = display.copy()
        put_text(view, title, (10, 24), scale=0.52, color=(0, 255, 255))
        put_text(view, "Click 4 matching points. Press r to reset, ESC to cancel.", (10, 48), scale=0.42)
        for i, (x, y) in enumerate(points):
            cv2.circle(view, (int(x), int(y)), 5, (0, 255, 255), -1, cv2.LINE_AA)
            put_text(view, str(i + 1), (int(x) + 8, int(y) - 8), scale=0.45, color=(0, 255, 255))
        cv2.imshow(window, view)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((float(x), float(y)))
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    redraw()
    while len(points) < 4:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            points = []
            break
        if key == ord("r"):
            points.clear()
            redraw()
    cv2.setMouseCallback(window, lambda *_args: None)
    cv2.destroyWindow(window)
    return points


def read_opencv_digital_frame(args: argparse.Namespace, digital_cap: Optional[cv2.VideoCapture]) -> Optional[np.ndarray]:
    if digital_cap is None:
        return None
    ok, digital_raw = digital_cap.read()
    if not ok or digital_raw is None:
        return None
    return orient_digital_frame(
        extract_digital_frame(digital_raw, args.digital_split),
        args.digital_rotate,
        args.digital_flip_h,
        args.digital_flip_v,
    )


def open_digital_camera(index: int, fps: int, backend: str = "msmf") -> Tuple[cv2.VideoCapture, str]:
    backend_name, backend_value = backend_candidates(backend)[0]
    cap = cv2.VideoCapture(index, backend_value)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FPS, fps)
        for _ in range(8):
            ret, frame = cap.read()
            if ret and frame is not None:
                break
            time.sleep(0.03)
    return cap, backend_name


def select_digital_frame(args: argparse.Namespace, sdk_frame: Optional[TC001SdkFrame], digital_cap: Optional[cv2.VideoCapture]) -> Optional[np.ndarray]:
    if args.digital_source == "sdk-top":
        return orient_digital_frame(
            None if sdk_frame is None else sdk_frame.preview_bgr,
            args.digital_rotate,
            args.digital_flip_h,
            args.digital_flip_v,
        )
    return read_opencv_digital_frame(args, digital_cap)


def run_alignment_calibration(args: argparse.Namespace, sdk_cam: TC001SdkCamera, digital_cap: Optional[cv2.VideoCapture]) -> bool:
    print("Calibration: capturing one digital frame and one SDK thermal frame...")
    sdk_frame = sdk_cam.read_frame(timeout_s=2.0)
    digital = select_digital_frame(args, sdk_frame, digital_cap)
    if sdk_frame is None or digital is None:
        print("ERROR: Could not capture calibration frames from SDK and digital camera.")
        if args.digital_source == "sdk-top":
            print("The SDK top-half preview was empty. Retry with --digital-source opencv if the visual stream is exposed separately.")
        return False

    temp_c = sdk_frame.temp_c
    thermal_frame = ThermalFrame(display_source=temp_c, temp_c=temp_c, mode="sdk-radiometric")
    thermal_oriented = apply_orientation(thermal_frame, ViewerState(rotate=int(args.rotate), flip_h=bool(args.flip_h), flip_v=bool(args.flip_v)))
    thermal_preview, _ = make_heatmap(thermal_oriented.display_source, ViewerState())

    print("Calibration: select the same 4 points in DIGITAL first, then THERMAL.")
    digital_pts = collect_four_points("TC001 Digital Calibration", digital, "DIGITAL: click 4 points")
    if len(digital_pts) != 4:
        print("Calibration cancelled.")
        return False
    thermal_pts_oriented = collect_four_points("TC001 Thermal Calibration", thermal_preview, "THERMAL: click same 4 points")
    if len(thermal_pts_oriented) != 4:
        print("Calibration cancelled.")
        return False

    # The saved homography maps digital coordinates to the unrotated SDK matrix.
    cal_state = ViewerState(rotate=int(args.rotate), flip_h=bool(args.flip_h), flip_v=bool(args.flip_v))
    thermal_pts = inverse_orient_points(np.asarray(thermal_pts_oriented, dtype=np.float32), temp_c.shape[:2], cal_state)

    matrix, _mask = cv2.findHomography(np.asarray(digital_pts, dtype=np.float32), thermal_pts, method=0)
    if matrix is None:
        print("ERROR: Could not compute homography from the selected points.")
        return False
    save_alignment(args.alignment_file, matrix, (digital.shape[1], digital.shape[0]), (temp_c.shape[1], temp_c.shape[0]))
    print(f"Calibration saved: {os.path.abspath(args.alignment_file)}")
    return True


def probe_raw_streams(args: argparse.Namespace) -> int:
    print("Scanning for possible TC001 raw/radiometric streams...")
    print("This may briefly open/close each camera several times. Close TOPDON app, Windows Camera, OBS, Teams, etc. first.\n")

    configs: List[Tuple[str, OpenOptions]] = [
        ("normal visual", OpenOptions(request_raw=False)),
        ("raw bytes, no FOURCC", OpenOptions(request_raw=True, raw_format=True, fourcc=None, raw_height_factor=2)),
        ("Y16 raw bytes 256x384", OpenOptions(request_raw=True, raw_format=True, fourcc="Y16 ", raw_height_factor=2)),
        ("Y16 raw bytes 256x192", OpenOptions(request_raw=True, raw_format=True, fourcc="Y16 ", raw_height_factor=1)),
        ("YUY2 raw bytes 256x384", OpenOptions(request_raw=True, raw_format=True, fourcc="YUY2", raw_height_factor=2)),
        ("YUY2 no raw-format 256x384", OpenOptions(request_raw=True, raw_format=False, fourcc="YUY2", raw_height_factor=2)),
    ]

    any_radiometric = False

    for idx in range(args.scan_max + 1):
        for backend_name, _backend_value in backend_candidates(args.backend):
            for label, opt in configs:
                cap, used_backend = open_camera(
                    idx,
                    args.width,
                    args.height,
                    args.fps,
                    backend=backend_name,
                    options=opt,
                )
                if not cap.isOpened():
                    cap.release()
                    continue

                ret, frame = cap.read()
                if not ret or frame is None:
                    cap.release()
                    continue

                thermal = decode_frame(
                    frame,
                    args.width,
                    args.height,
                    args.fallback_min_c,
                    args.fallback_max_c,
                    False,
                    args.temp_scale,
                    args.temp_offset,
                )

                prefix = f"device={idx} backend={used_backend:<5} config={label:<26}"
                print(f"{prefix} -> {describe_frame(frame)} | decode={thermal.mode}")

                if thermal.temp_c is not None:
                    any_radiometric = True
                    print(
                        "    FOUND TEMPERATURE DATA: "
                        f"min={np.nanmin(thermal.temp_c):.2f}C, "
                        f"avg={np.nanmean(thermal.temp_c):.2f}C, "
                        f"max={np.nanmax(thermal.temp_c):.2f}C"
                    )
                    print(
                        "    Try running viewer with:\n"
                        f"    py tc001_thermal_viewer_v5.py --device {idx} --backend {used_backend} "
                        f"--raw-request {'--raw-format ' if opt.raw_format else ''}"
                        f"{'--fourcc ' + repr(opt.fourcc) + ' ' if opt.fourcc else ''}"
                        f"--raw-height-factor {opt.raw_height_factor} --probe\n"
                    )

                cap.release()
                time.sleep(0.05)

    if not any_radiometric:
        print("\nNo radiometric/raw temperature stream was detected through OpenCV.")
        print("Your current Windows driver is probably exposing only the visual/pseudo-color UVC stream.")
        print("Real TC001 temperatures require a raw/radiometric stream or the vendor SDK/app/driver path.")
        print("You can still display approximate values with --estimate-temps, but they are not calibrated.")
        return 1

    return 0


def draw_hud(
    img: np.ndarray,
    thermal: ThermalFrame,
    state: ViewerState,
    fps: float,
    contrast_range: Tuple[float, float],
    recorder: Recorder,
) -> None:
    h, w = img.shape[:2]

    src_h, src_w = thermal.display_source.shape[:2]
    scale_x = w / max(src_w, 1)
    scale_y = h / max(src_h, 1)
    marker_scale = max(1.0, min(scale_x, scale_y))
    center_x_src = src_w // 2
    center_y_src = src_h // 2
    center_x = int(center_x_src * scale_x)
    center_y = int(center_y_src * scale_y)

    approx = thermal.approx_temps
    temp = thermal.temp_c

    center_temp: Optional[float] = None
    avg_temp: Optional[float] = None
    min_temp: Optional[float] = None
    max_temp: Optional[float] = None
    min_xy_temp: Optional[Tuple[int, int]] = None
    max_xy_temp: Optional[Tuple[int, int]] = None

    if temp is not None and temp.size > 0:
        temp_h, temp_w = temp.shape[:2]
        safe_cx = min(max(center_x_src, 0), temp_w - 1)
        safe_cy = min(max(center_y_src, 0), temp_h - 1)
        center_temp = float(temp[safe_cy, safe_cx])
        avg_temp, min_temp, max_temp, min_xy_temp, max_xy_temp = safe_stats(temp)

    signal = thermal.display_source.astype(np.float32)
    safe_cx_signal = min(max(center_x_src, 0), signal.shape[1] - 1)
    safe_cy_signal = min(max(center_y_src, 0), signal.shape[0] - 1)
    center_signal = float(signal[safe_cy_signal, safe_cx_signal])
    avg_signal, min_signal, max_signal, min_xy_signal, max_xy_signal = safe_stats(signal)

    if temp is not None:
        center_label = fmt_temp(center_temp, state.unit, approx) if state.show_labels else None
    else:
        center_label = "Temp N/A" if state.show_labels else None

    draw_crosshair(img, center_x, center_y, center_label)

    if state.show_labels:
        if temp is not None and avg_temp is not None:
            if max_temp is not None and max_xy_temp is not None and max_temp >= avg_temp + state.threshold_c:
                x, y = int(max_xy_temp[0] * scale_x), int(max_xy_temp[1] * scale_y)
                radius = max(5, int(marker_scale * 1.6))
                cv2.circle(img, (x, y), radius, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(img, (x, y), max(4, radius - 1), (255, 255, 255), 1, cv2.LINE_AA)
                put_text(img, "HOT " + fmt_temp(max_temp, state.unit, approx), (x + 10, max(18, y - 8)), scale=0.42)

            if min_temp is not None and min_xy_temp is not None and min_temp <= avg_temp - state.threshold_c:
                x, y = int(min_xy_temp[0] * scale_x), int(min_xy_temp[1] * scale_y)
                radius = max(5, int(marker_scale * 1.6))
                cv2.circle(img, (x, y), radius, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(img, (x, y), max(4, radius - 1), (255, 255, 255), 1, cv2.LINE_AA)
                put_text(img, "COLD " + fmt_temp(min_temp, state.unit, approx), (x + 10, min(h - 12, y + 18)), scale=0.42)
        elif avg_signal is not None:
            if max_signal is not None and max_xy_signal is not None and max_signal >= avg_signal + state.signal_threshold:
                x, y = int(max_xy_signal[0] * scale_x), int(max_xy_signal[1] * scale_y)
                radius = max(5, int(marker_scale * 1.6))
                cv2.circle(img, (x, y), radius, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(img, (x, y), max(4, radius - 1), (255, 255, 255), 1, cv2.LINE_AA)
                put_text(img, f"HOT signal {max_signal:.0f}", (x + 10, max(18, y - 8)), scale=0.42)

            if min_signal is not None and min_xy_signal is not None and min_signal <= avg_signal - state.signal_threshold:
                x, y = int(min_xy_signal[0] * scale_x), int(min_xy_signal[1] * scale_y)
                radius = max(5, int(marker_scale * 1.6))
                cv2.circle(img, (x, y), radius, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(img, (x, y), max(4, radius - 1), (255, 255, 255), 1, cv2.LINE_AA)
                put_text(img, f"COLD signal {min_signal:.0f}", (x + 10, min(h - 12, y + 18)), scale=0.42)

    if not state.show_hud:
        return

    cmap_name, _ = COLORMAPS[state.cmap_index % len(COLORMAPS)]
    rec_text = "REC" if state.recording else "OFF"
    if state.recording and recorder.start_time:
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - recorder.start_time))
        rec_text += f" {elapsed}"

    if temp is not None:
        measurement_line = (
            f"Avg: {fmt_temp(avg_temp, state.unit, approx)} | "
            f"Center: {fmt_temp(center_temp, state.unit, approx)} | "
            f"Min/Max: {fmt_temp(min_temp, state.unit, approx)} / {fmt_temp(max_temp, state.unit, approx)}"
        )
        threshold_line = f"Temp threshold: {state.threshold_c:.1f}C"
    else:
        measurement_line = (
            f"Temp: N/A | Center signal: {center_signal:.0f} | "
            f"Avg signal: {avg_signal:.0f} | Min/Max signal: {min_signal:.0f}/{max_signal:.0f}"
            if avg_signal is not None and min_signal is not None and max_signal is not None
            else "Temp: N/A | Signal: N/A"
        )
        threshold_line = f"Signal threshold: {state.signal_threshold:.0f} brightness units"

    lines = [
        f"Mode: {thermal.mode} | FPS: {fps:.1f} | Source: {src_w}x{src_h}",
        measurement_line,
        threshold_line,
        f"Color: {cmap_name} | Contrast: {state.contrast:.1f} | Range: {contrast_range[0]:.1f}..{contrast_range[1]:.1f}",
        f"Blur: {state.blur} | Zoom: {state.zoom}x | Fit: {'ON' if state.fit_window else 'OFF'} | Rotate: {state.rotate}",
        f"Labels: {'ON' if state.show_labels else 'OFF'} | HUD: ON | Recording: {rec_text}",
    ]
    if state.show_help_overlay:
        lines.extend(
            [
                "Keys: q quit | c color | +/- contrast | z/x zoom | a fit | f fullscreen",
                "      h HUD | l labels | r record | s snapshot | t/g threshold | ? help",
            ]
        )

    if thermal.warning:
        lines.insert(1, thermal.warning)

    text_scale = 0.42 if w < 1200 else 0.46
    line_step = 17 if w < 1200 else 19
    x0, y0 = 10, 20
    for i, line in enumerate(lines):
        put_text(img, line, (x0, y0 + i * line_step), scale=text_scale)


def start_recording(recorder: Recorder, frame: np.ndarray, fps: int) -> bool:
    h, w = frame.shape[:2]
    filename = f"tc001_record_{timestamp()}.avi"

    for fourcc_name in ("MJPG", "XVID"):
        writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*fourcc_name), fps, (w, h))
        if writer.isOpened():
            recorder.writer = writer
            recorder.filename = filename
            recorder.frame_size = (w, h)
            recorder.start_time = time.time()
            print(f"Recording started: {os.path.abspath(filename)}")
            return True
        writer.release()

    print("ERROR: Could not start AVI recording. Try another OpenCV build or codec.")
    return False


def stop_recording(recorder: Recorder) -> None:
    if recorder.writer is not None:
        recorder.writer.release()
        print(f"Recording saved: {os.path.abspath(recorder.filename or '')}")

    recorder.writer = None
    recorder.filename = None
    recorder.frame_size = None
    recorder.start_time = 0.0


def save_snapshot(frame: np.ndarray) -> None:
    filename = f"tc001_snapshot_{timestamp()}.png"
    ok = cv2.imwrite(filename, frame)
    if ok:
        print(f"Snapshot saved: {os.path.abspath(filename)}")
    else:
        print("ERROR: Could not save snapshot PNG.")


def set_fullscreen(enabled: bool) -> None:
    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL,
    )


def print_controls() -> None:
    print(
        """
Keyboard controls
-----------------
q or ESC  : Quit
c / C     : Next / previous color map
+ / -     : Increase / decrease contrast
z / x     : Increase / decrease zoom. Default is 1x/no zoom.
a         : Toggle fit-to-window display. Default is ON.
b / n     : Increase / decrease blur/smoothing
f         : Toggle fullscreen
h         : Toggle HUD visibility
l         : Toggle labels and hot/cold point labels
t / g     : Increase / decrease threshold
u         : Toggle Celsius / Fahrenheit when temp data exists
i         : Invert heatmap
o / p     : Rotate 90 degrees clockwise / counter-clockwise
m         : Toggle horizontal mirror flip
v         : Toggle vertical flip
r         : Start/stop AVI recording in current folder
s         : Save PNG snapshot in current folder
1 / 2 / 3 : Split mode (1=Left/Thermal, 2=Right/Visual, 3=Both)
?         : Print this help again
""".strip()
    )


def handle_key(key: int, state: ViewerState, recorder: Recorder, current_frame: np.ndarray, fps: int) -> bool:
    if key < 0:
        return True

    key = key & 0xFF

    if key in (27, ord("q")):
        return False
    if key == ord("?"):
        state.show_help_overlay = not state.show_help_overlay
        print_controls()
    elif key == ord("c"):
        state.cmap_index = (state.cmap_index + 1) % len(COLORMAPS)
    elif key == ord("C"):
        state.cmap_index = (state.cmap_index - 1) % len(COLORMAPS)
    elif key in (ord("+"), ord("=")):
        state.contrast = min(10.0, round(state.contrast + 0.2, 1))
    elif key in (ord("-"), ord("_")):
        state.contrast = max(0.2, round(state.contrast - 0.2, 1))
    elif key == ord("z"):
        state.zoom = min(10, state.zoom + 1)
    elif key == ord("x"):
        state.zoom = max(1, state.zoom - 1)
    elif key == ord("a"):
        state.fit_window = not state.fit_window
    elif key == ord("b"):
        state.blur = min(31, state.blur + 2 if state.blur else 3)
        if state.blur % 2 == 0:
            state.blur += 1
    elif key == ord("n"):
        state.blur = max(0, state.blur - 2)
        if state.blur == 1:
            state.blur = 0
    elif key == ord("f"):
        state.fullscreen = not state.fullscreen
        set_fullscreen(state.fullscreen)
    elif key == ord("h"):
        state.show_hud = not state.show_hud
    elif key == ord("l"):
        state.show_labels = not state.show_labels
    elif key == ord("t"):
        state.threshold_c = min(100.0, round(state.threshold_c + 0.5, 1))
        state.signal_threshold = min(255.0, round(state.signal_threshold + 5.0, 1))
    elif key == ord("g"):
        state.threshold_c = max(0.0, round(state.threshold_c - 0.5, 1))
        state.signal_threshold = max(0.0, round(state.signal_threshold - 5.0, 1))
    elif key == ord("u"):
        state.unit = "F" if state.unit == "C" else "C"
    elif key == ord("i"):
        state.invert = not state.invert
    elif key == ord("o"):
        state.rotate = (state.rotate + 90) % 360
    elif key == ord("p"):
        state.rotate = (state.rotate - 90) % 360
    elif key == ord("m"):
        state.flip_h = not state.flip_h
    elif key == ord("v"):
        state.flip_v = not state.flip_v
    elif key == ord("s"):
        save_snapshot(current_frame)
    elif key == ord("r"):
        if state.recording:
            stop_recording(recorder)
            state.recording = False
        else:
            state.recording = start_recording(recorder, current_frame, fps)
    elif key == ord("1"):
        state.split_mode = 1
    elif key == ord("2"):
        state.split_mode = 2
    elif key == ord("3"):
        state.split_mode = 0

    return True
