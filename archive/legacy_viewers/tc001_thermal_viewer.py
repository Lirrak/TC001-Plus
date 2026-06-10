#!/usr/bin/env python3
"""
TC001 Thermal Viewer for Windows
--------------------------------
A practical OpenCV viewer for the TOPDON TC001/TS001-style thermal cameras.

Main features:
- Pick a Windows video device number from the command line.
- Show a larger, false-color thermal image instead of the tiny raw feed.
- Cycle color maps, adjust contrast, blur/smoothing, zoom, labels, HUD, fullscreen.
- Show average scene temperature, center temperature with crosshair, hottest/coldest
  points when they exceed a threshold from the scene average.
- Save snapshots as PNG and recordings as AVI in the current folder.

Install:
    python -m pip install opencv-python numpy

List camera indexes:
    python tc001_thermal_viewer.py --list

Run:
    python tc001_thermal_viewer.py --device 1

Notes on temperature accuracy:
- Best case: the camera is exposed as a raw UVC stream with TC001-like 2-frame data.
  The script decodes the second half as 16-bit radiometric data using:
      temp_C = low_byte / 64 + high_byte * 4 - 273.15
- Fallback: if OpenCV only receives an 8-bit/color image, the program still displays a
  nice heatmap, but temperatures are approximate/relative. Use --fallback-min-c and
  --fallback-max-c if you want a rough mapping for display only.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


WINDOW_NAME = "TOPDON TC001 Thermal Viewer"

TC001_WIDTH = 256
TC001_HEIGHT = 192
TC001_FPS = 25

# OpenCV colormap names and constants. Some constants may not exist in older OpenCV.
COLORMAPS: Tuple[Tuple[str, int], ...] = tuple(
    (name, cmap)
    for name, cmap in [
        ("JET", cv2.COLORMAP_JET),
        ("INFERNO", getattr(cv2, "COLORMAP_INFERNO", cv2.COLORMAP_JET)),
        ("MAGMA", getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_JET)),
        ("PLASMA", getattr(cv2, "COLORMAP_PLASMA", cv2.COLORMAP_JET)),
        ("VIRIDIS", getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET)),
        ("TURBO", getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)),
        ("HOT", cv2.COLORMAP_HOT),
        ("BONE", cv2.COLORMAP_BONE),
        ("RAINBOW", cv2.COLORMAP_RAINBOW),
        ("OCEAN", cv2.COLORMAP_OCEAN),
    ]
)


@dataclass
class ThermalFrame:
    """Container for decoded camera data."""

    display_source: np.ndarray       # 2D float/uint8 source used for heatmap display
    temp_c: Optional[np.ndarray]     # 2D Celsius temperature matrix, if available
    mode: str                        # "raw16", "raw-yuyv", or "visual-approx"
    warning: Optional[str] = None


@dataclass
class ViewerState:
    """Runtime settings changed by keyboard controls."""

    cmap_index: int = 0
    contrast: float = 1.6
    blur: int = 0                    # Gaussian kernel radius-ish, forced odd kernel when used
    zoom: int = 4
    fullscreen: bool = False
    show_hud: bool = True
    show_labels: bool = True
    invert: bool = False
    unit: str = "C"                  # C or F
    threshold_c: float = 3.0         # hot/cold shown if far enough from scene average
    recording: bool = False
    snapshot_count: int = 0


@dataclass
class Recorder:
    writer: Optional[cv2.VideoWriter] = None
    filename: Optional[str] = None
    frame_size: Optional[Tuple[int, int]] = None
    start_time: float = 0.0


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def to_display_temp(value_c: float, unit: str) -> Tuple[float, str]:
    if unit.upper() == "F":
        return value_c * 9.0 / 5.0 + 32.0, "F"
    return value_c, "C"


def fmt_temp(value_c: Optional[float], unit: str, approx: bool = False) -> str:
    if value_c is None or not np.isfinite(value_c):
        return "N/A"
    value, suffix = to_display_temp(float(value_c), unit)
    mark = "~" if approx else ""
    return f"{mark}{value:.1f} deg{suffix}"


def open_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    """Open a camera, preferring DirectShow on Windows."""
    backends = []
    if platform.system().lower().startswith("win"):
        backends.append(cv2.CAP_DSHOW)
    backends.append(cv2.CAP_ANY)

    last_cap: Optional[cv2.VideoCapture] = None
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if last_cap is not None:
            last_cap.release()
        last_cap = cap

        # Try to request the raw TC001-like stream. Not all Windows drivers honor this.
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height * 2)  # TC001 raw stream often stacks two 192px frames.
        cap.set(cv2.CAP_PROP_FPS, fps)

        if cap.isOpened():
            return cap

    return last_cap if last_cap is not None else cv2.VideoCapture(index)


def list_video_devices(max_index: int = 10, width: int = TC001_WIDTH, height: int = TC001_HEIGHT) -> None:
    """Print camera indexes that can be opened by OpenCV."""
    print("Scanning video device indexes...")
    found = False
    for idx in range(max_index + 1):
        cap = open_camera(idx, width, height, TC001_FPS)
        ok = False
        shape = "unknown"
        if cap.isOpened():
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    ok = True
                    shape = str(frame.shape) + f", dtype={frame.dtype}"
                    break
        cap.release()
        if ok:
            found = True
            print(f"  [{idx}] OK  frame={shape}")
    if not found:
        print("No readable OpenCV video devices found.")
        print("Check USB connection, close other camera apps, or try another USB port/cable.")


def decode_frame(
    frame: np.ndarray,
    width: int,
    height: int,
    fallback_min_c: float,
    fallback_max_c: float,
) -> ThermalFrame:
    """
    Decode several likely OpenCV frame formats.

    Expected ideal TC001-like raw format:
        frame.reshape((height * 2, width, 2))
        first half: display YUYV image data
        second half: temperature bytes per pixel

    Fallback format:
        normal BGR/gray image, no reliable temperature data.
    """
    if frame is None:
        raise ValueError("Camera returned an empty frame.")

    arr = frame

    # Case 1: already a 16-bit single-channel thermal matrix.
    if arr.dtype == np.uint16:
        if arr.ndim == 3:
            arr16 = arr[:, :, 0]
        else:
            arr16 = arr
        arr16 = crop_or_resize_matrix(arr16, width, height)
        temp_c = arr16.astype(np.float32) / 64.0 - 273.15
        return ThermalFrame(display_source=temp_c, temp_c=temp_c, mode="raw16")

    # Case 2: raw byte stream from TC001/TS001 style UVC camera.
    # The known layout is (384, 256, 2) for 256x192 with two stacked frames.
    try:
        flat = arr.reshape(-1)
        needed = width * height * 2 * 2
        if flat.size >= needed:
            raw = flat[:needed].reshape((height * 2, width, 2))
            image_part = raw[:height, :, :]
            temp_part = raw[height : height * 2, :, :]

            # Convert display half from YUYV to BGR, then grayscale source for false color.
            try:
                bgr = cv2.cvtColor(image_part, cv2.COLOR_YUV2BGR_YUYV)
                display_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            except cv2.error:
                display_gray = image_part[:, :, 0].astype(np.float32)

            # TC001-like low/high byte formula. Channel 0 is low byte, channel 1 is high byte.
            low = temp_part[:, :, 0].astype(np.float32)
            high = temp_part[:, :, 1].astype(np.float32)
            temp_c = low / 64.0 + high * 4.0 - 273.15

            # Sanity check: reject impossible decoded temperatures and use visual fallback.
            finite = np.isfinite(temp_c)
            if finite.any():
                t_min = float(np.nanpercentile(temp_c, 1))
                t_max = float(np.nanpercentile(temp_c, 99))
                if -80.0 <= t_min <= 1000.0 and -80.0 <= t_max <= 1000.0 and t_max > t_min:
                    return ThermalFrame(display_source=temp_c, temp_c=temp_c, mode="raw-yuyv")
    except Exception:
        # Fall through to visual mode.
        pass

    # Case 3: normal color/gray image; display works, temps are approximate.
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            gray = arr[:, :, 0]
        else:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    else:
        gray = arr

    gray = crop_or_resize_matrix(gray, width, height).astype(np.float32)
    norm = normalize_linear(gray, float(np.nanmin(gray)), float(np.nanmax(gray)))
    temp_c = fallback_min_c + (fallback_max_c - fallback_min_c) * (norm.astype(np.float32) / 255.0)
    return ThermalFrame(
        display_source=gray,
        temp_c=temp_c,
        mode="visual-approx",
        warning="OpenCV did not expose raw radiometric data; temperatures are approximate.",
    )


def crop_or_resize_matrix(src: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return a height x width matrix by center cropping first, then resizing if needed."""
    mat = src
    if mat.ndim == 3:
        mat = mat[:, :, 0]
    h, w = mat.shape[:2]

    # If a stacked 2-frame image slips through as 384x256, use the first half for display fallback.
    if h >= height * 2 and w >= width:
        mat = mat[:height, :width]
        h, w = mat.shape[:2]

    if h >= height and w >= width:
        y0 = max(0, (h - height) // 2)
        x0 = max(0, (w - width) // 2)
        mat = mat[y0 : y0 + height, x0 : x0 + width]
    elif h != height or w != width:
        mat = cv2.resize(mat, (width, height), interpolation=cv2.INTER_AREA)
    return mat


def normalize_linear(src: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Normalize a matrix to uint8 0..255."""
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-6:
        return np.zeros(src.shape, dtype=np.uint8)
    out = (src.astype(np.float32) - vmin) * (255.0 / (vmax - vmin))
    return np.clip(out, 0, 255).astype(np.uint8)


def make_heatmap(src: np.ndarray, state: ViewerState) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Apply optional blur, adaptive contrast, false color, invert, and zoom."""
    work = src.astype(np.float32)

    if state.blur > 0:
        k = max(1, int(state.blur))
        if k % 2 == 0:
            k += 1
        work = cv2.GaussianBlur(work, (k, k), 0)

    finite = np.isfinite(work)
    if not finite.any():
        norm8 = np.zeros(work.shape, dtype=np.uint8)
        vmin, vmax = 0.0, 1.0
    else:
        lo, hi = np.nanpercentile(work, [1.0, 99.0])
        if abs(float(hi) - float(lo)) < 1e-6:
            lo, hi = float(np.nanmin(work)), float(np.nanmax(work))
        if abs(float(hi) - float(lo)) < 1e-6:
            hi = lo + 1.0

        mid = (float(lo) + float(hi)) / 2.0
        half = max((float(hi) - float(lo)) / 2.0, 1e-6)
        # Higher contrast means a narrower source range mapped into 0..255.
        half = half / max(state.contrast, 0.1)
        vmin, vmax = mid - half, mid + half
        norm8 = normalize_linear(work, vmin, vmax)

    if state.invert:
        norm8 = 255 - norm8

    cmap_name, cmap_value = COLORMAPS[state.cmap_index % len(COLORMAPS)]
    heatmap = cv2.applyColorMap(norm8, cmap_value)

    if state.zoom != 1:
        heatmap = cv2.resize(
            heatmap,
            (heatmap.shape[1] * state.zoom, heatmap.shape[0] * state.zoom),
            interpolation=cv2.INTER_CUBIC,
        )

    return heatmap, (vmin, vmax)


def draw_crosshair(img: np.ndarray, x: int, y: int, label: Optional[str] = None) -> None:
    """Draw a readable black/white crosshair."""
    h, w = img.shape[:2]
    length = max(12, min(w, h) // 25)
    thickness_outer = 3
    thickness_inner = 1

    for color, thick in [((0, 0, 0), thickness_outer), ((255, 255, 255), thickness_inner)]:
        cv2.line(img, (x - length, y), (x + length, y), color, thick, cv2.LINE_AA)
        cv2.line(img, (x, y - length), (x, y + length), color, thick, cv2.LINE_AA)

    if label:
        put_text(img, label, (x + length + 6, max(18, y - 8)), scale=0.55)


def put_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float = 0.55,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Draw text with a black outline for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, org, font, scale, bg, 3, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, 1, cv2.LINE_AA)


def draw_hud(
    img: np.ndarray,
    thermal: ThermalFrame,
    state: ViewerState,
    fps: float,
    contrast_range: Tuple[float, float],
    recorder: Recorder,
) -> None:
    """Draw HUD, center crosshair, hot/cold markers."""
    h, w = img.shape[:2]
    z = state.zoom
    temp = thermal.temp_c
    approx = thermal.mode == "visual-approx"

    center_x_src = TC001_WIDTH // 2
    center_y_src = TC001_HEIGHT // 2
    center_x = int(center_x_src * z)
    center_y = int(center_y_src * z)

    center_temp = None
    avg_temp = None
    min_temp = max_temp = None
    min_xy = max_xy = None

    if temp is not None and temp.size > 0:
        center_temp = float(temp[center_y_src, center_x_src])
        avg_temp = float(np.nanmean(temp))
        min_index = int(np.nanargmin(temp))
        max_index = int(np.nanargmax(temp))
        min_y, min_x = np.unravel_index(min_index, temp.shape)
        max_y, max_x = np.unravel_index(max_index, temp.shape)
        min_temp = float(temp[min_y, min_x])
        max_temp = float(temp[max_y, max_x])
        min_xy = (int(min_x), int(min_y))
        max_xy = (int(max_x), int(max_y))

    center_label = fmt_temp(center_temp, state.unit, approx) if state.show_labels else None
    draw_crosshair(img, center_x, center_y, center_label)

    if state.show_labels and avg_temp is not None:
        # Mark hottest/coldest only if they differ enough from the scene average.
        if max_temp is not None and max_xy is not None and max_temp >= avg_temp + state.threshold_c:
            x, y = int(max_xy[0] * z), int(max_xy[1] * z)
            cv2.circle(img, (x, y), max(5, z * 2), (0, 0, 0), 3, cv2.LINE_AA)
            cv2.circle(img, (x, y), max(4, z * 2 - 1), (255, 255, 255), 1, cv2.LINE_AA)
            put_text(img, "HOT " + fmt_temp(max_temp, state.unit, approx), (x + 10, max(18, y - 8)), scale=0.5)

        if min_temp is not None and min_xy is not None and min_temp <= avg_temp - state.threshold_c:
            x, y = int(min_xy[0] * z), int(min_xy[1] * z)
            cv2.circle(img, (x, y), max(5, z * 2), (0, 0, 0), 3, cv2.LINE_AA)
            cv2.circle(img, (x, y), max(4, z * 2 - 1), (255, 255, 255), 1, cv2.LINE_AA)
            put_text(img, "COLD " + fmt_temp(min_temp, state.unit, approx), (x + 10, min(h - 12, y + 18)), scale=0.5)

    if not state.show_hud:
        return

    cmap_name, _ = COLORMAPS[state.cmap_index % len(COLORMAPS)]
    rec_text = "REC" if state.recording else "OFF"
    if state.recording and recorder.start_time:
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - recorder.start_time))
        rec_text += f" {elapsed}"

    lines = [
        f"Mode: {thermal.mode} | FPS: {fps:.1f}",
        f"Avg: {fmt_temp(avg_temp, state.unit, approx)} | Center: {fmt_temp(center_temp, state.unit, approx)}",
        f"Min: {fmt_temp(min_temp, state.unit, approx)} | Max: {fmt_temp(max_temp, state.unit, approx)} | Threshold: {state.threshold_c:.1f}C",
        f"Color: {cmap_name} | Contrast: {state.contrast:.1f} | Range: {contrast_range[0]:.1f}..{contrast_range[1]:.1f}",
        f"Blur: {state.blur} | Zoom: {state.zoom}x | Labels: {'ON' if state.show_labels else 'OFF'} | HUD: ON",
        f"Recording: {rec_text}",
        "Keys: q/ESC quit | c colormap | +/- contrast | z/x zoom | b/n blur | f fullscreen",
        "      h HUD | l labels | r record | s snapshot | t/g threshold | u C/F | i invert",
    ]

    if thermal.warning:
        lines.insert(1, thermal.warning)

    x0, y0 = 10, 22
    for i, line in enumerate(lines):
        put_text(img, line, (x0, y0 + i * 20), scale=0.5)


def start_recording(recorder: Recorder, frame: np.ndarray) -> bool:
    """Create an AVI writer for the current frame size."""
    h, w = frame.shape[:2]
    filename = f"tc001_record_{timestamp()}.avi"

    # MJPG is broadly playable on Windows; XVID fallback is also common.
    for fourcc_name in ("MJPG", "XVID"):
        writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*fourcc_name), TC001_FPS, (w, h))
        if writer.isOpened():
            recorder.writer = writer
            recorder.filename = filename
            recorder.frame_size = (w, h)
            recorder.start_time = time.time()
            print(f"Recording started: {os.path.abspath(filename)}")
            return True
        writer.release()

    print("ERROR: Could not start AVI recording. Try installing a different OpenCV build/codecs.")
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
z / x     : Increase / decrease zoom
b / n     : Increase / decrease blur/smoothing
f         : Toggle fullscreen
h         : Toggle HUD visibility
l         : Toggle labels and hot/cold point labels
t / g     : Increase / decrease hot/cold threshold from scene average
u         : Toggle Celsius / Fahrenheit
i         : Invert heatmap
r         : Start/stop AVI recording in current folder
s         : Save PNG snapshot in current folder
?         : Print this help again
""".strip()
    )


def handle_key(key: int, state: ViewerState, recorder: Recorder, current_frame: np.ndarray) -> bool:
    """Handle keyboard input. Return False when the app should exit."""
    if key < 0:
        return True

    key = key & 0xFF

    if key in (27, ord("q")):
        return False
    if key == ord("?"):
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
    elif key == ord("g"):
        state.threshold_c = max(0.0, round(state.threshold_c - 0.5, 1))
    elif key == ord("u"):
        state.unit = "F" if state.unit == "C" else "C"
    elif key == ord("i"):
        state.invert = not state.invert
    elif key == ord("s"):
        save_snapshot(current_frame)
    elif key == ord("r"):
        if state.recording:
            stop_recording(recorder)
            state.recording = False
        else:
            state.recording = start_recording(recorder, current_frame)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live false-color viewer for TOPDON TC001 thermal camera on Windows."
    )
    parser.add_argument("-d", "--device", type=int, default=None, help="OpenCV video device number, e.g. 0, 1, 2.")
    parser.add_argument("--list", action="store_true", help="Scan and list readable OpenCV video device indexes.")
    parser.add_argument("--scan-max", type=int, default=10, help="Highest camera index to scan with --list.")
    parser.add_argument("--width", type=int, default=TC001_WIDTH, help="Thermal width. Default: 256.")
    parser.add_argument("--height", type=int, default=TC001_HEIGHT, help="Thermal height. Default: 192.")
    parser.add_argument("--fps", type=int, default=TC001_FPS, help="Requested FPS. Default: 25.")
    parser.add_argument("--zoom", type=int, default=4, help="Initial zoom factor. Default: 4.")
    parser.add_argument("--contrast", type=float, default=1.6, help="Initial contrast. Default: 1.6.")
    parser.add_argument("--blur", type=int, default=0, help="Initial Gaussian blur kernel size. Default: 0/off.")
    parser.add_argument("--threshold", type=float, default=3.0, help="Hot/cold point threshold from scene average in Celsius. Default: 3.0.")
    parser.add_argument("--fallback-min-c", type=float, default=15.0, help="Approximate Celsius minimum for non-radiometric fallback mode.")
    parser.add_argument("--fallback-max-c", type=float, default=45.0, help="Approximate Celsius maximum for non-radiometric fallback mode.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        list_video_devices(args.scan_max, args.width, args.height)
        return 0

    device = args.device
    if device is None:
        print("No --device was provided.")
        print("Use --list first, then run for example: python tc001_thermal_viewer.py --device 1")
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
    )
    recorder = Recorder()

    print(f"Opening video device {device}...")
    cap = open_camera(device, args.width, args.height, args.fps)
    if not cap.isOpened():
        print(f"ERROR: Could not open video device {device}.")
        print("Possible causes:")
        print("  - The TC001 is not plugged in or is not powered by the USB port.")
        print("  - The selected --device number is wrong. Run with --list.")
        print("  - Another app is using the camera. Close Topdon/Windows Camera/OBS/etc.")
        print("  - Windows driver exposes the TC001 in a non-UVC way; try reinstalling/removing that driver.")
        return 1

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print_controls()

    fps_smooth = 0.0
    last_time = time.time()
    read_failures = 0

    try:
        while True:
            ret, frame = cap.read()
            now = time.time()
            dt = max(now - last_time, 1e-6)
            inst_fps = 1.0 / dt
            fps_smooth = inst_fps if fps_smooth <= 0 else fps_smooth * 0.90 + inst_fps * 0.10
            last_time = now

            if not ret or frame is None:
                read_failures += 1
                if read_failures > 30:
                    print("ERROR: Camera opened, but no frames were received.")
                    print("Try a different --device number, unplug/replug the camera, or close other camera apps.")
                    break
                time.sleep(0.03)
                continue
            read_failures = 0

            thermal = decode_frame(frame, args.width, args.height, args.fallback_min_c, args.fallback_max_c)
            heatmap, contrast_range = make_heatmap(thermal.display_source, state)
            draw_hud(heatmap, thermal, state, fps_smooth, contrast_range, recorder)

            if state.recording and recorder.writer is not None:
                out = heatmap
                if recorder.frame_size is not None:
                    rw, rh = recorder.frame_size
                    if (out.shape[1], out.shape[0]) != (rw, rh):
                        out = cv2.resize(out, (rw, rh), interpolation=cv2.INTER_AREA)
                recorder.writer.write(out)

            cv2.imshow(WINDOW_NAME, heatmap)
            key = cv2.waitKey(1)
            if not handle_key(key, state, recorder, heatmap):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        if state.recording or recorder.writer is not None:
            stop_recording(recorder)
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
