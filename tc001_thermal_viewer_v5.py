#!/usr/bin/env python3
"""
TOPDON TC001 / TC001 Plus Thermal Viewer for Windows - v5
---------------------------------------------------------

This version is made for the situation where OpenCV only shows a nice-looking
640x480 BGR thermal video, but does NOT expose real radiometric temperature data.

Key points:
- If the camera feed is BGR/visual-only, real temperature is NOT available.
  The HUD will correctly show Temp=N/A instead of fake numbers.
- Use --find-raw to scan for a possible raw/radiometric stream.
- Use --raw-request --raw-format --backend dshow to try opening the TC001 raw stream.
- Use --estimate-temps only if you accept approximate, brightness-based temperatures.

Install:
    py -m pip install opencv-python numpy

List video devices:
    py tc001_thermal_viewer_v5.py --list

Normal visual viewer:
    py tc001_thermal_viewer_v5.py --device 1 --rotate 90

Scan for radiometric/raw mode:
    py tc001_thermal_viewer_v5.py --find-raw --scan-max 5

Try raw/radiometric mode directly:
    py tc001_thermal_viewer_v5.py --device 1 --backend dshow --raw-request --raw-format --probe

Approximate temperature from visual frame, NOT calibrated:
    py tc001_thermal_viewer_v5.py --device 1 --rotate 90 --estimate-temps --fallback-min-c 22 --fallback-max-c 45
"""

from __future__ import annotations

# Reduce noisy OpenCV native warnings while scanning camera indexes.
# This must be set before importing cv2.
import os
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import argparse
import sys
import time
from typing import Optional

import cv2
import numpy as np

from tc001_face import DigitalFaceDetector, FaceTracker, update_face_tracks
from tc001_sdk import TC001SdkCamera, TC001SdkFrame, TC001_FPS, TC001_HEIGHT, TC001_WIDTH

try:
    cv2.setLogLevel(0)
except Exception:
    pass

from tc001_viewer_types import (
    WINDOW_NAME,
    ViewerState,
    Recorder,
    OpenOptions,
    ThermalFrame,
)
from tc001_viewer_utils import (
    describe_fourcc,
    resize_to_window,
)
from tc001_viewer_core import (
    load_alignment,
    open_camera,
    decode_frame,
    describe_frame,
    list_video_devices,
    split_frame_if_needed,
    apply_orientation,
    make_heatmap,
)
from tc001_viewer_gui import (
    open_digital_camera,
    run_alignment_calibration,
    probe_raw_streams,
    select_digital_frame,
    build_detection_debug_record,
    emit_detection_debug,
    draw_face_overlay,
    draw_hud,
    draw_digital_debug,
    stop_recording,
    start_recording,
    print_controls,
    handle_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live false-color viewer for TOPDON TC001/TC001 Plus thermal camera on Windows."
    )
    parser.add_argument("-d", "--device", type=int, default=1, help="OpenCV video device number, e.g. 0, 1, 2. Default: 1.")
    parser.add_argument("--list", action="store_true", help="Scan and list readable OpenCV video device indexes.")
    parser.add_argument("--find-raw", action="store_true", help="Scan camera/backend/format combinations for radiometric temperature data.")
    parser.add_argument("--scan-max", type=int, default=3, help="Highest camera index to scan. Default: 3.")
    parser.add_argument("--backend", choices=["auto", "msmf", "dshow", "any"], default="auto", help="OpenCV backend. Default: auto.")
    parser.add_argument("--width", type=int, default=TC001_WIDTH, help="Raw thermal width. Default: 256.")
    parser.add_argument("--height", type=int, default=TC001_HEIGHT, help="Raw thermal height. Default: 192.")
    parser.add_argument("--fps", type=int, default=TC001_FPS, help="Requested FPS. Default: 25.")
    parser.add_argument("--zoom", type=int, default=1, help="Initial zoom factor. Default: 1/no zoom.")
    parser.add_argument("--window-width", type=int, default=1280, help="Initial viewer window width. Default: 1280.")
    parser.add_argument("--window-height", type=int, default=960, help="Initial viewer window height. Default: 960.")
    parser.add_argument("--no-fit-window", dest="fit_window", action="store_false", default=True, help="Do not resize the heatmap to fill the OpenCV window.")
    parser.add_argument("--contrast", type=float, default=1.4, help="Initial contrast. Default: 1.4.")
    parser.add_argument("--blur", type=int, default=0, help="Initial Gaussian blur kernel size. Default: 0/off.")
    parser.add_argument("--threshold", type=float, default=3.0, help="Hot/cold threshold from scene average in Celsius. Default: 3.0.")
    parser.add_argument("--signal-threshold", type=float, default=20.0, help="Hot/cold threshold in visual-only brightness units. Default: 20.")
    parser.add_argument("--fallback-min-c", type=float, default=20.0, help="Approx Celsius min for --estimate-temps mode.")
    parser.add_argument("--fallback-max-c", type=float, default=45.0, help="Approx Celsius max for --estimate-temps mode.")
    parser.add_argument("--estimate-temps", action="store_true", default=False, help="Estimate temperatures from 8-bit visual frames. Not calibrated. Default: disabled.")
    parser.add_argument("--no-estimate-temps", dest="estimate_temps", action="store_false", help="Disable temperature estimation.")
    parser.add_argument("--temp-scale", type=float, default=1.0, help="Calibration scale applied to decoded/estimated Celsius. Default: 1.0.")
    parser.add_argument("--temp-offset", type=float, default=0.0, help="Calibration offset in Celsius. Default: 0.0.")
    parser.add_argument("--roi-temp-mode", choices=["robust", "max"], default="robust", help="Face ROI temperature mode. robust ignores small hot outliers; max keeps old behavior. Default: robust.")
    parser.add_argument("--roi-temp-percentile", type=float, default=90.0, help="Percentile used for robust face ROI temperature. Default: 90.")
    parser.add_argument("--roi-hot-outlier-delta-c", type=float, default=2.5, help="Hot outlier delta above ROI median in Celsius. Default: 2.5.")
    parser.add_argument("--roi-mask-shrink", type=float, default=0.85, help="Shrink mapped ROI polygon before measuring temperature. Default: 0.85.")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=90, help="Initial clockwise rotation for the thermal viewer. Default: 90.")
    parser.add_argument("--flip-h", action="store_true", help="Initial horizontal mirror flip.")
    parser.add_argument("--flip-v", action="store_true", help="Initial vertical flip.")
    parser.add_argument("--raw-request", action="store_true", help="Try to request a raw TC001-sized stream.")
    parser.add_argument("--raw-format", action="store_true", help="Ask OpenCV for unconverted raw bytes using CAP_PROP_FORMAT=-1.")
    parser.add_argument("--raw-height-factor", type=int, choices=[1, 2], default=2, help="Raw stream height factor. 2 means 256x384 for TC001 stacked stream. Default: 2.")
    parser.add_argument("--fourcc", type=str, default=None, help="Optional FOURCC request, for example Y16, Y16<space>, YUY2, MJPG.")
    parser.add_argument("--probe", action="store_true", help="Print first-frame information after opening the device.")
    parser.add_argument("--sdk-raw", action="store_true", help="Use TOPDON libiruvc.dll to read real TC001 Plus radiometric frames from USB 0BDA:5830.")
    parser.add_argument(
        "--sdk-dll-dir",
        default=r"C:\Program Files\TOPDON\TopView\dll\dll_c001p",
        help="Directory containing TOPDON libiruvc.dll. Default: TopView dll_c001p folder.",
    )
    parser.add_argument("--face-detect", action="store_true", help="Detect faces on the TC001 Plus digital camera stream and map the box to SDK thermal frames.")
    parser.add_argument("--face-model", choices=["auto", "tasks", "mediapipe", "haar"], default="auto", help="Face detector backend. Default: auto.")
    parser.add_argument("--face-task-model", default=None, help="Optional MediaPipe Tasks face detector .tflite model path.")
    parser.add_argument("--face-min-confidence", type=float, default=0.55, help="Minimum MediaPipe face confidence. Default: 0.55.")
    parser.add_argument("--face-detect-interval", type=int, default=3, help="Run face detection every N frames, then track/smooth between detections. Default: 3.")
    parser.add_argument("--face-hold-frames", type=int, default=15, help="Keep the last face box for this many frames after missed detections. Default: 15.")
    parser.add_argument("--face-reid-frames", type=int, default=45, help="Keep recently lost confirmed tracks for re-identification without drawing them. Default: 45.")
    parser.add_argument("--face-smoothing", type=float, default=0.65, help="EMA smoothing for face box coordinates. Default: 0.65.")
    parser.add_argument("--min-face-hits", type=int, default=2, help="Detection hits required before Haar/profile tracks are shown. Default: 2.")
    parser.add_argument("--max-face-area-ratio", type=float, default=0.20, help="Reject face boxes larger than this frame area ratio. Default: 0.20.")
    parser.add_argument("--max-box-overlap", type=float, default=0.30, help="IoU threshold used to merge/suppress duplicate boxes. Default: 0.30.")
    parser.add_argument("--cascade-fallback", choices=["off", "auto", "always"], default="auto", help="Use Haar/profile cascade fallback. Default: auto.")
    parser.add_argument("--max-faces", type=int, default=5, help="Maximum number of people/face tracks to show. Default: 5.")
    parser.add_argument("--head-fallback", choices=["off", "auto", "always"], default="off", help="Use bright-region head fallback when face/profile detection misses. Default: off.")
    parser.add_argument("--head-confirm-frames", type=int, default=2, help="Detection frames required before showing a HEAD TRACK on thermal view. Default: 2.")
    parser.add_argument("--min-head-confidence", type=float, default=0.65, help="Assigned confidence for accepted head fallback boxes. Default: 0.65.")
    parser.add_argument("--debug-detections", action="store_true", help="Print face/ROI detection data every --debug-detections-every frames.")
    parser.add_argument("--debug-detections-every", type=int, default=15, help="Frame interval for --debug-detections and --save-detection-debug. Default: 15.")
    parser.add_argument("--save-detection-debug", default=None, help="Append face/ROI detection debug records as JSONL to this file.")
    parser.add_argument("--digital-source", choices=["sdk-top", "opencv"], default="sdk-top", help="Digital/visual source for AI. Default: sdk-top.")
    parser.add_argument("--digital-device", type=int, default=1, help="OpenCV digital/visual device for TC001 Plus AI. Default: 1.")
    parser.add_argument("--digital-backend", choices=["auto", "msmf", "dshow", "any"], default="msmf", help="OpenCV backend used only for --digital-source opencv. Default: msmf.")
    parser.add_argument("--digital-split", choices=["right", "left", "none", "both"], default="right", help="Which part of camera1 contains the digital image. Default: right.")
    parser.add_argument("--digital-rotate", type=int, choices=[0, 90, 180, 270], default=90, help="Clockwise rotation applied only to the digital AI/debug frame. Default: 90.")
    parser.add_argument("--digital-flip-h", action="store_true", help="Flip only the digital AI/debug frame horizontally.")
    parser.add_argument("--digital-flip-v", action="store_true", help="Flip only the digital AI/debug frame vertically.")
    parser.add_argument("--show-digital-debug", action="store_true", help="Show a separate window with digital camera face detection.")
    parser.add_argument("--alignment", choices=["homography", "simple-scale"], default="homography", help="Digital-to-thermal mapping mode. Default: homography with simple-scale fallback if file is missing.")
    parser.add_argument("--alignment-file", default="tc001_alignment.json", help="Path to digital-to-thermal homography JSON. Default: tc001_alignment.json.")
    parser.add_argument("--calibrate-alignment", action="store_true", help="Capture digital and SDK frames, click 4 matching points, and save --alignment-file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        list_video_devices(args.scan_max, args.width, args.height, args.fps, args.backend)
        return 0

    if args.find_raw:
        return probe_raw_streams(args)

    use_digital_ai = bool(args.face_detect or args.show_digital_debug or args.calibrate_alignment)
    if use_digital_ai and not args.sdk_raw:
        print("ERROR: AI/digital alignment modes require --sdk-raw.")
        print("Use: py .\\tc001_thermal_viewer_v5.py --sdk-raw --face-detect --digital-device 1 --show-digital-debug")
        return 2

    device = args.device
    if device is None:
        print("No --device was provided.")
        print("Use --list first, then run for example: py tc001_thermal_viewer_v5.py --device 1")
        try:
            typed = input("Enter video device number now, or press Enter to cancel: ").strip()
        except EOFError:
            typed = ""
        if not typed:
            return 2
        try:
            device = int(typed)
        except ValueError:
            print("ERROR: Device number must be an integer such as 0, 1, or 2.")
            return 2

    state = ViewerState(
        contrast=max(0.2, float(args.contrast)),
        blur=max(0, int(args.blur)),
        zoom=max(1, int(args.zoom)),
        threshold_c=max(0.0, float(args.threshold)),
        signal_threshold=max(0.0, float(args.signal_threshold)),
        roi_temp_mode=args.roi_temp_mode,
        roi_temp_percentile=float(args.roi_temp_percentile),
        roi_hot_outlier_delta_c=max(0.1, float(args.roi_hot_outlier_delta_c)),
        roi_mask_shrink=min(max(float(args.roi_mask_shrink), 0.1), 1.0),
        rotate=int(args.rotate),
        flip_h=bool(args.flip_h),
        flip_v=bool(args.flip_v),
        fit_window=bool(args.fit_window),
    )
    recorder = Recorder()

    fourcc = args.fourcc
    if fourcc == "Y16":
        fourcc = "Y16 "

    options = OpenOptions(
        request_raw=bool(args.raw_request),
        raw_format=bool(args.raw_format),
        fourcc=fourcc,
        raw_height_factor=int(args.raw_height_factor),
    )

    cap: Optional[cv2.VideoCapture] = None
    digital_cap: Optional[cv2.VideoCapture] = None
    sdk_cam: Optional[TC001SdkCamera] = None
    face_detector: Optional[DigitalFaceDetector] = None
    face_tracker = FaceTracker()
    detection_debug_file = None
    alignment = load_alignment(args.alignment_file, args.alignment)

    if args.sdk_raw:
        print("Opening TC001 Plus through TOPDON SDK/libiruvc real radiometric stream...")
        try:
            sdk_cam = TC001SdkCamera(args.sdk_dll_dir)
            sdk_cam.open()
        except Exception as exc:
            print("ERROR: Could not open TC001 Plus SDK raw stream.")
            print(f"SDK error: {exc}")
            print("Close TopView and other camera apps, then reconnect TC001 Plus.")
            return 1
        print("Opened SDK raw stream: 256x384 YUY2, bottom 256x192 decoded as real Celsius.")
    else:
        print(f"Opening video device {device} using backend={args.backend}...")
        if args.raw_request:
            print(
                "Raw request enabled: "
                f"size={args.width}x{args.height * args.raw_height_factor}, "
                f"raw_format={args.raw_format}, fourcc={repr(fourcc)}"
            )

        cap, used_backend = open_camera(
            device,
            args.width,
            args.height,
            args.fps,
            backend=args.backend,
            options=options,
        )

        if not cap.isOpened():
            print(f"ERROR: Could not open video device {device}.")
            print("Possible causes:")
            print("  - The TC001/TC001 Plus is not plugged in.")
            print("  - The selected --device number is wrong. Run: py tc001_thermal_viewer_v5.py --list")
            print("  - Another app is using the camera. Close TOPDON app, Windows Camera, OBS, Teams, etc.")
            print("  - Try a different backend: --backend msmf or --backend dshow")
            return 1

        print(f"Opened device {device} with backend={used_backend}.")

    if use_digital_ai and args.digital_source == "opencv":
        if int(args.digital_device) == 0:
            print("ERROR: --digital-device 0 is blocked for AI. TC001 Plus digital camera must use camera1.")
            return 2
        print(f"Opening TC001 Plus digital camera for AI on camera{args.digital_device} using backend={args.digital_backend}...")
        digital_cap, digital_backend = open_digital_camera(int(args.digital_device), args.fps, args.digital_backend)
        if not digital_cap.isOpened():
            print(f"ERROR: Could not open TC001 Plus digital camera device {args.digital_device}.")
            print("Run --list and confirm the TC001 Plus visual/digital stream index. Camera0 is intentionally not used for AI.")
            return 1
        print(f"Opened digital camera{args.digital_device} with backend={digital_backend}. Split={args.digital_split}.")
    elif use_digital_ai:
        print("Using SDK top-half preview as the AI digital source. OpenCV camera1 is not opened for AI.")

    if args.face_detect:
        face_detector = DigitalFaceDetector(args.face_model, args.face_min_confidence, args.face_task_model)
        print(
            f"Face detection enabled: detector={face_detector.name}, "
            f"digital_source={args.digital_source}, alignment={alignment.mode}, "
            f"max_faces={args.max_faces}, head_fallback={args.head_fallback}, "
            f"cascade_fallback={args.cascade_fallback}, min_face_hits={args.min_face_hits}."
        )

    if args.calibrate_alignment:
        if sdk_cam is None:
            print("ERROR: --calibrate-alignment requires --sdk-raw so the thermal target is real radiometric TC001 data.")
            if digital_cap is not None:
                digital_cap.release()
            return 2
        if args.digital_source == "opencv" and digital_cap is None:
            print("ERROR: --calibrate-alignment requires a readable --digital-device.")
            return 2
        ok = run_alignment_calibration(args, sdk_cam, digital_cap)
        if digital_cap is not None:
            digital_cap.release()
        if sdk_cam is not None:
            sdk_cam.close()
        cv2.destroyAllWindows()
        return 0 if ok else 1

    if args.probe:
        if sdk_cam is not None:
            temp_c = sdk_cam.read_temp_c(timeout_s=2.0)
            if temp_c is not None:
                print("First SDK raw frame: shape=(192, 256), dtype=float32, real radiometric Celsius")
                print(
                    "Temp stats C: "
                    f"min={np.nanmin(temp_c):.2f}, "
                    f"avg={np.nanmean(temp_c):.2f}, "
                    f"max={np.nanmax(temp_c):.2f}"
                )
            else:
                print("Probe warning: SDK stream opened, but no raw frame was received.")
        elif cap is not None:
            ret, frame = cap.read()
            if ret and frame is not None:
                thermal = decode_frame(
                    frame,
                    args.width,
                    args.height,
                    args.fallback_min_c,
                    args.fallback_max_c,
                    args.estimate_temps,
                    args.temp_scale,
                    args.temp_offset,
                )
                print(f"First frame: {describe_frame(frame)}")
                try:
                    print(
                        f"Capture properties: W={cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}, "
                        f"H={cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}, "
                        f"FPS={cap.get(cv2.CAP_PROP_FPS):.1f}, "
                        f"FOURCC={describe_fourcc(cap.get(cv2.CAP_PROP_FOURCC))}"
                    )
                except Exception:
                    pass
                print(f"Decode mode: {thermal.mode}")
                if thermal.warning:
                    print(f"Warning: {thermal.warning}")
                if thermal.temp_c is not None:
                    print(
                        "Temp stats C: "
                        f"min={np.nanmin(thermal.temp_c):.2f}, "
                        f"avg={np.nanmean(thermal.temp_c):.2f}, "
                        f"max={np.nanmax(thermal.temp_c):.2f}"
                    )
                else:
                    print("Temp stats C: N/A. This stream does not contain decoded temperature data.")
                    print("Run: py tc001_thermal_viewer_v5.py --find-raw --scan-max 5")
            else:
                print("Probe warning: camera opened, but first frame could not be read.")

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, max(320, int(args.window_width)), max(240, int(args.window_height)))
    except cv2.error as exc:
        print("ERROR: OpenCV GUI window could not be created.")
        print("This usually means opencv-python-headless is installed instead of opencv-python.")
        print("Fix:")
        print("  py -m pip uninstall opencv-python-headless -y")
        print("  py -m pip install --upgrade opencv-python")
        print(f"OpenCV error: {exc}")
        return 1

    print_controls()

    if args.save_detection_debug:
        detection_debug_file = open(args.save_detection_debug, "a", encoding="utf-8")
        print(f"Saving detection debug JSONL to: {os.path.abspath(args.save_detection_debug)}")

    fps_smooth = 0.0
    last_time = time.time()
    read_failures = 0
    frame_index = 0

    try:
        while True:
            frame = None
            digital_frame = None
            sdk_frame: Optional[TC001SdkFrame] = None
            if sdk_cam is not None:
                sdk_frame = sdk_cam.read_frame(timeout_s=1.0)
                temp_c = sdk_frame.temp_c if sdk_frame is not None else None
                ret = temp_c is not None
            elif cap is not None:
                ret, frame = cap.read()
                temp_c = None
            else:
                ret = False
                temp_c = None

            now = time.time()
            dt = max(now - last_time, 1e-6)
            inst_fps = 1.0 / dt
            fps_smooth = inst_fps if fps_smooth <= 0 else fps_smooth * 0.90 + inst_fps * 0.10
            last_time = now

            if not ret:
                read_failures += 1
                if read_failures > 30:
                    print("ERROR: Camera opened, but no frames were received.")
                    if sdk_cam is not None:
                        print("Close TopView/other camera apps, reconnect TC001 Plus, then retry --sdk-raw.")
                    else:
                        print("Try another --device number, unplug/replug camera, close other camera apps,")
                        print("or run with --backend msmf / --backend dshow.")
                    break
                time.sleep(0.03)
                continue

            read_failures = 0
            frame_index += 1

            if use_digital_ai:
                digital_frame = select_digital_frame(args, sdk_frame, digital_cap)
                if digital_frame is not None and face_detector is not None:
                    face_tracker = update_face_tracks(
                        face_detector,
                        face_tracker,
                        digital_frame,
                        frame_index,
                        args.face_detect_interval,
                        args.face_hold_frames,
                        args.face_smoothing,
                        args.max_faces,
                        args.head_fallback,
                        args.head_confirm_frames,
                        args.min_head_confidence,
                        args.min_face_hits,
                        args.max_face_area_ratio,
                        args.max_box_overlap,
                        args.cascade_fallback,
                        args.face_reid_frames,
                    )

            if sdk_cam is not None and temp_c is not None:
                temp_c = temp_c * args.temp_scale + args.temp_offset
                thermal = ThermalFrame(display_source=temp_c, temp_c=temp_c, mode="sdk-radiometric")
            else:
                # For TC001 Plus, split the side-by-side frame if requested.
                working_frame = split_frame_if_needed(frame, state.split_mode)
                thermal = decode_frame(
                    working_frame,
                    args.width,
                    args.height,
                    args.fallback_min_c,
                    args.fallback_max_c,
                    args.estimate_temps,
                    args.temp_scale,
                    args.temp_offset,
                )
            thermal_raw = thermal
            thermal = apply_orientation(thermal, state)
            heatmap, contrast_range = make_heatmap(thermal.display_source, state)
            heatmap = resize_to_window(heatmap, state.fit_window)
            if digital_frame is not None and args.face_detect:
                if args.debug_detections or detection_debug_file is not None:
                    every = max(1, int(args.debug_detections_every))
                    stats = face_tracker.last_stats or {}
                    should_emit_detection_debug = frame_index % every == 0 or bool(
                        stats.get("new_tracks")
                        or stats.get("reidentified_tracks")
                        or stats.get("merged_detections")
                        or stats.get("duplicate_suppressed")
                        or stats.get("rejected_detections")
                        or len(face_tracker.active_tracks(confirmed_only=True)) >= 2
                    )
                    if should_emit_detection_debug:
                        record = build_detection_debug_record(
                            frame_index,
                            digital_frame.shape[:2],
                            face_tracker,
                            thermal_raw,
                            alignment,
                            {
                                "digital_source": args.digital_source,
                                "digital_split": args.digital_split,
                                "digital_rotate": int(args.digital_rotate),
                                "face_model": args.face_model,
                                "detector": face_detector.name if face_detector is not None else "none",
                                "max_faces": int(args.max_faces),
                                "face_detect_interval": int(args.face_detect_interval),
                                "face_hold_frames": int(args.face_hold_frames),
                                "face_reid_frames": int(args.face_reid_frames),
                                "face_smoothing": float(args.face_smoothing),
                                "roi_temp_mode": state.roi_temp_mode,
                                "roi_temp_percentile": float(state.roi_temp_percentile),
                                "roi_hot_outlier_delta_c": float(state.roi_hot_outlier_delta_c),
                                "roi_mask_shrink": float(state.roi_mask_shrink),
                                "min_face_hits": int(args.min_face_hits),
                                "max_face_area_ratio": float(args.max_face_area_ratio),
                                "max_box_overlap": float(args.max_box_overlap),
                                "cascade_fallback": args.cascade_fallback,
                                "head_fallback": args.head_fallback,
                                "alignment": alignment.mode,
                            },
                            state=state,
                        )
                        emit_detection_debug(record, bool(args.debug_detections), detection_debug_file)
                draw_face_overlay(heatmap, thermal_raw, thermal, digital_frame.shape[:2], face_tracker, alignment, state)
            draw_hud(heatmap, thermal, state, fps_smooth, contrast_range, recorder)

            if digital_frame is not None and args.show_digital_debug:
                cv2.imshow("TC001 Plus Digital Face Debug", draw_digital_debug(digital_frame, face_tracker, alignment, args.digital_source, args.digital_rotate))

            if state.recording and recorder.writer is not None:
                out = heatmap
                if recorder.frame_size is not None:
                    rw, rh = recorder.frame_size
                    if (out.shape[1], out.shape[0]) != (rw, rh):
                        out = cv2.resize(out, (rw, rh), interpolation=cv2.INTER_AREA)
                recorder.writer.write(out)

            cv2.imshow(WINDOW_NAME, heatmap)
            key = cv2.waitKey(1)
            if not handle_key(key, state, recorder, heatmap, args.fps):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        if state.recording or recorder.writer is not None:
            stop_recording(recorder)
        if cap is not None:
            cap.release()
        if digital_cap is not None:
            digital_cap.release()
        if detection_debug_file is not None:
            detection_debug_file.close()
        if sdk_cam is not None:
            sdk_cam.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
