#!/usr/bin/env python3
"""
TC001 / TC001 Plus Thermal Viewer for Windows - v3
---------------------------------------------------
A practical OpenCV viewer for TOPDON TC001 / TC001 Plus style USB thermal cameras.

What v3 fixes compared with the earlier version:
- Does NOT mis-detect normal 8-bit BGR frames as TC001 raw radiometric frames.
  This was the main reason for wrong temperatures such as nearly-fixed ~49 C.
- Uses the camera's received visual resolution directly in visual mode, for example
  640x480, instead of cropping it down to 256x192 and zooming it back up.
- Default zoom is now 1x. No artificial enlargement unless you press z or pass --zoom.
- Adds rotation and flip controls so you can correct the TC001 orientation.

Install:
    py -m pip install opencv-python numpy

List cameras:
    py tc001_thermal_viewer_v3.py --list

Run:
    py tc001_thermal_viewer_v3.py --device 1

Try rotation if the camera is sideways:
    py tc001_thermal_viewer_v3.py --device 1 --rotate 90
    py tc001_thermal_viewer_v3.py --device 1 --rotate 270

Important temperature note:
- If OpenCV exposes real raw/radiometric data, this program shows real-ish Celsius values.
- If OpenCV exposes only a normal 8-bit visual/BGR frame, real temperature data is not
  available through that frame. The program will show a nice heatmap and intensity HUD,
  but temperature fields will be N/A unless you explicitly enable --estimate-temps.
- --estimate-temps is only a visual approximation. It is not a calibrated measurement.
"""

from __future__ import annotations

# Reduce noisy OpenCV native warnings while scanning camera indexes.
# This must be set before importing cv2.
import os
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import argparse
import platform
import sys
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    cv2.setLogLevel(0)
except Exception:
    pass


WINDOW_NAME = "TOPDON TC001 Thermal Viewer v3"

# TC001 / TC001 Plus thermal sensor size commonly used by raw/radiometric streams.
TC001_WIDTH = 256
TC001_HEIGHT = 192
TC001_FPS = 25

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
    display_source: np.ndarray          # 2D source used to generate false-color heatmap
    temp_c: Optional[np.ndarray]        # 2D Celsius matrix when radiometric/estimated data exists
    mode: str                           # raw16, raw-tc001, visual-only, visual-estimate
    warning: Optional[str] = None
    approx_temps: bool = False


@dataclass
class ViewerState:
    cmap_index: int = 0
    contrast: float = 1.4
    blur: int = 0
    zoom: int = 1                       # v3 default: no zoom
    fullscreen: bool = False
    show_hud: bool = True
    show_labels: bool = True
    invert: bool = False
    unit: str = "C"
    threshold_c: float = 3.0            # used only when temp_c is available
    signal_threshold: float = 20.0      # used in visual-only mode, brightness units
    recording: bool = False
    rotate: int = 0                     # clockwise: 0, 90, 180, 270
    flip_h: bool = False
    flip_v: bool = False


@dataclass
class Recorder:
    writer: Optional[cv2.VideoWriter] = None
    filename: Optional[str] = None
    frame_size: Optional[Tuple[int, int]] = None
    start_time: float = 0.0


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def backend_name_to_cv2(name: str) -> int:
    name = name.lower().strip()
    if name == "msmf":
        return cv2.CAP_MSMF
    if name == "dshow":
        return cv2.CAP_DSHOW
    if name == "any":
        return cv2.CAP_ANY
    raise ValueError(f"Unknown backend: {name}")


def backend_candidates(name: str) -> Sequence[Tuple[str, int]]:
    name = name.lower().strip()
    if name != "auto":
        return [(name, backend_name_to_cv2(name))]

    if platform.system().lower().startswith("win"):
        # MSMF is usually stable for normal UVC frames.
        # DSHOW is sometimes better when trying to request raw formats.
        return [("msmf", cv2.CAP_MSMF), ("dshow", cv2.CAP_DSHOW), ("any", cv2.CAP_ANY)]

    return [("any", cv2.CAP_ANY)]


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


def normalize_linear(src: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-6:
        return np.zeros(src.shape, dtype=np.uint8)
    out = (src.astype(np.float32) - vmin) * (255.0 / (vmax - vmin))
    return np.clip(out, 0, 255).astype(np.uint8)


def robust_percentile_range(src: np.ndarray, low_p: float = 1.0, high_p: float = 99.0) -> Tuple[float, float]:
    finite = np.isfinite(src)
    if not finite.any():
        return 0.0, 1.0
    lo, hi = np.nanpercentile(src, [low_p, high_p])
    lo = float(lo)
    hi = float(hi)
    if abs(hi - lo) < 1e-6:
        lo = float(np.nanmin(src))
        hi = float(np.nanmax(src))
    if abs(hi - lo) < 1e-6:
        hi = lo + 1.0
    return lo, hi


def open_camera(
    index: int,
    width: int,
    height: int,
    fps: int,
    backend: str = "auto",
    request_raw: bool = True,
) -> Tuple[cv2.VideoCapture, str]:
    """Open a camera and return (capture, backend_name_used)."""
    last_cap: Optional[cv2.VideoCapture] = None
    last_backend_name = backend

    for backend_name, backend_value in backend_candidates(backend):
        if last_cap is not None:
            last_cap.release()

        cap = cv2.VideoCapture(index, backend_value)
        last_cap = cap
        last_backend_name = backend_name

        if not cap.isOpened():
            continue

        if request_raw:
            # Try to request a TC001-like raw stream. Drivers may ignore this.
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height * 2)
            cap.set(cv2.CAP_PROP_FPS, fps)
        else:
            # For listing, do not force raw sizes because it can make some UVC cameras fail.
            cap.set(cv2.CAP_PROP_FPS, fps)

        for _ in range(10):
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap, backend_name
            time.sleep(0.02)

    if last_cap is None:
        return cv2.VideoCapture(index), last_backend_name
    return last_cap, last_backend_name


def describe_frame(frame: np.ndarray) -> str:
    shape = tuple(int(v) for v in frame.shape)
    dtype = str(frame.dtype)
    hint = ""

    if frame.dtype == np.uint16:
        hint = " | possible raw16 thermal"
    elif len(shape) == 3 and shape[2] == 3:
        hint = " | normal BGR/visual frame, not raw radiometric"
    elif len(shape) == 3 and shape[2] == 2:
        hint = " | possible YUYV/raw two-channel frame"
    elif len(shape) == 2:
        hint = " | grayscale/single-channel frame"

    if shape[:2] == (TC001_HEIGHT * 2, TC001_WIDTH):
        hint += " | possible stacked TC001 raw frame"
    elif shape[:2] == (TC001_HEIGHT, TC001_WIDTH):
        hint += " | TC001 sensor-sized frame"

    return f"shape={shape}, dtype={dtype}{hint}"


def list_video_devices(
    max_index: int = 3,
    width: int = TC001_WIDTH,
    height: int = TC001_HEIGHT,
    fps: int = TC001_FPS,
    backend: str = "auto",
) -> None:
    print("Scanning video device indexes...")
    found = []

    for idx in range(max_index + 1):
        cap, used_backend = open_camera(
            idx,
            width,
            height,
            fps,
            backend=backend,
            request_raw=False,
        )

        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                found.append(idx)
                print(f"  [{idx}] OK  backend={used_backend}  {describe_frame(frame)}")
        cap.release()

    if not found:
        print("No readable OpenCV video devices found.")
        print("Check USB connection, close other camera apps, or try another USB port/cable.")
        return

    print("\nTip:")
    print("  - If your laptop has a built-in webcam, it is often index 0.")
    print("  - The TC001/TC001 Plus is often the next index, for example --device 1.")
    print("  - shape=(480, 640, 3) means OpenCV is receiving a normal 8-bit visual frame.")
    print("    That frame can be displayed nicely, but it does not contain reliable radiometric temperature data.")


def crop_or_resize_matrix(src: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return a height x width matrix by center-cropping first, then resizing if needed."""
    mat = src[:, :, 0] if src.ndim == 3 else src
    h, w = mat.shape[:2]

    if h >= height and w >= width:
        y0 = max(0, (h - height) // 2)
        x0 = max(0, (w - width) // 2)
        mat = mat[y0 : y0 + height, x0 : x0 + width]
    elif h != height or w != width:
        mat = cv2.resize(mat, (width, height), interpolation=cv2.INTER_AREA)

    return mat


def looks_like_tc001_raw_bytes(arr: np.ndarray, width: int, height: int) -> bool:
    """
    Return True only for shapes that plausibly represent TC001 raw bytes.

    Important: a normal BGR frame such as 640x480x3 has enough bytes to reshape into
    a fake raw buffer, but it is NOT raw radiometric data. v1 accepted that by mistake.
    """
    if arr.dtype != np.uint8:
        return False

    needed = width * height * 2 * 2

    if arr.ndim == 3 and arr.shape[2] == 2 and arr.shape[0] == height * 2 and arr.shape[1] == width:
        return True

    if arr.ndim == 2 and arr.shape == (height * 2, width * 2):
        return True

    if arr.ndim == 2 and arr.shape == (height * 2, width) and arr.size == needed:
        return True

    if arr.ndim == 3 and arr.shape[2] == 1 and arr.size == needed:
        return True

    return False


def decode_tc001_raw_bytes(arr: np.ndarray, width: int, height: int) -> Optional[np.ndarray]:
    try:
        raw = arr.reshape(-1)[: width * height * 2 * 2].reshape((height * 2, width, 2))
        temp_part = raw[height : height * 2, :, :]

        low = temp_part[:, :, 0].astype(np.float32)
        high = temp_part[:, :, 1].astype(np.float32)
        temp_c = low / 64.0 + high * 4.0 - 273.15

        if not np.isfinite(temp_c).any():
            return None

        t_min = float(np.nanpercentile(temp_c, 1))
        t_max = float(np.nanpercentile(temp_c, 99))

        # Conservative sanity check.
        if -80.0 <= t_min <= 1000.0 and -80.0 <= t_max <= 1000.0 and t_max > t_min:
            return temp_c
    except Exception:
        return None

    return None


def decode_frame(
    frame: np.ndarray,
    width: int,
    height: int,
    fallback_min_c: float,
    fallback_max_c: float,
    estimate_temps: bool,
    temp_scale: float,
    temp_offset: float,
) -> ThermalFrame:
    if frame is None:
        raise ValueError("Camera returned an empty frame.")

    arr = frame

    # Case 1: 16-bit single-channel thermal matrix.
    if arr.dtype == np.uint16:
        arr16 = arr[:, :, 0] if arr.ndim == 3 else arr
        arr16 = crop_or_resize_matrix(arr16, width, height)
        temp_c = arr16.astype(np.float32) / 64.0 - 273.15
        temp_c = temp_c * temp_scale + temp_offset
        return ThermalFrame(display_source=temp_c, temp_c=temp_c, mode="raw16")

    # Case 2: TC001-like stacked raw byte stream.
    if looks_like_tc001_raw_bytes(arr, width, height):
        temp_c = decode_tc001_raw_bytes(arr, width, height)
        if temp_c is not None:
            temp_c = temp_c * temp_scale + temp_offset
            return ThermalFrame(display_source=temp_c, temp_c=temp_c, mode="raw-tc001")

    # Case 3: normal visual/BGR/gray frame. Use full received resolution.
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            gray = arr[:, :, 0]
        elif arr.shape[2] == 2:
            # Non-stacked YUYV-ish visual frame.
            gray = arr[:, :, 0]
        else:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    else:
        gray = arr

    gray = gray.astype(np.float32)

    if estimate_temps:
        lo, hi = robust_percentile_range(gray, 1.0, 99.0)
        norm = normalize_linear(gray, lo, hi).astype(np.float32) / 255.0
        temp_c = fallback_min_c + (fallback_max_c - fallback_min_c) * norm
        temp_c = temp_c * temp_scale + temp_offset
        return ThermalFrame(
            display_source=gray,
            temp_c=temp_c,
            mode="visual-estimate",
            warning="Visual estimate mode: temperatures are approximate, not calibrated.",
            approx_temps=True,
        )

    return ThermalFrame(
        display_source=gray,
        temp_c=None,
        mode="visual-only",
        warning="Visual-only frame: real radiometric temperature is not available via OpenCV.",
    )


def apply_orientation_to_matrix(mat: Optional[np.ndarray], rotate: int, flip_h: bool, flip_v: bool) -> Optional[np.ndarray]:
    if mat is None:
        return None

    out = mat
    rotate = rotate % 360

    if rotate == 90:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 180:
        out = cv2.rotate(out, cv2.ROTATE_180)
    elif rotate == 270:
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if flip_h:
        out = cv2.flip(out, 1)
    if flip_v:
        out = cv2.flip(out, 0)

    return out


def apply_orientation(thermal: ThermalFrame, state: ViewerState) -> ThermalFrame:
    display_source = apply_orientation_to_matrix(thermal.display_source, state.rotate, state.flip_h, state.flip_v)
    temp_c = apply_orientation_to_matrix(thermal.temp_c, state.rotate, state.flip_h, state.flip_v)
    assert display_source is not None
    return ThermalFrame(
        display_source=display_source,
        temp_c=temp_c,
        mode=thermal.mode,
        warning=thermal.warning,
        approx_temps=thermal.approx_temps,
    )


def make_heatmap(src: np.ndarray, state: ViewerState) -> Tuple[np.ndarray, Tuple[float, float]]:
    work = src.astype(np.float32)

    if state.blur > 0:
        k = max(1, int(state.blur))
        if k % 2 == 0:
            k += 1
        work = cv2.GaussianBlur(work, (k, k), 0)

    lo, hi = robust_percentile_range(work, 1.0, 99.0)

    mid = (lo + hi) / 2.0
    half = max((hi - lo) / 2.0, 1e-6)
    half = half / max(state.contrast, 0.1)
    vmin, vmax = mid - half, mid + half

    norm8 = normalize_linear(work, vmin, vmax)

    if state.invert:
        norm8 = 255 - norm8

    _, cmap_value = COLORMAPS[state.cmap_index % len(COLORMAPS)]
    heatmap = cv2.applyColorMap(norm8, cmap_value)

    # v3 default is 1x. Only resize when explicitly requested.
    if state.zoom != 1:
        interpolation = cv2.INTER_NEAREST if state.zoom >= 2 else cv2.INTER_AREA
        heatmap = cv2.resize(
            heatmap,
            (heatmap.shape[1] * state.zoom, heatmap.shape[0] * state.zoom),
            interpolation=interpolation,
        )

    return heatmap, (vmin, vmax)


def put_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float = 0.55,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, org, font, scale, bg, 3, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, 1, cv2.LINE_AA)


def draw_crosshair(img: np.ndarray, x: int, y: int, label: Optional[str] = None) -> None:
    h, w = img.shape[:2]
    length = max(10, min(w, h) // 28)

    for color, thick in [((0, 0, 0), 3), ((255, 255, 255), 1)]:
        cv2.line(img, (x - length, y), (x + length, y), color, thick, cv2.LINE_AA)
        cv2.line(img, (x, y - length), (x, y + length), color, thick, cv2.LINE_AA)

    if label:
        put_text(img, label, (x + length + 6, max(18, y - 8)), scale=0.52)


def safe_stats(mat: Optional[np.ndarray]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    if mat is None or mat.size == 0:
        return None, None, None, None, None
    try:
        avg = float(np.nanmean(mat))
        min_index = int(np.nanargmin(mat))
        max_index = int(np.nanargmax(mat))
        min_y, min_x = np.unravel_index(min_index, mat.shape)
        max_y, max_x = np.unravel_index(max_index, mat.shape)
        min_val = float(mat[min_y, min_x])
        max_val = float(mat[max_y, max_x])
        return avg, min_val, max_val, (int(min_x), int(min_y)), (int(max_x), int(max_y))
    except Exception:
        return None, None, None, None, None


def draw_hud(
    img: np.ndarray,
    thermal: ThermalFrame,
    state: ViewerState,
    fps: float,
    contrast_range: Tuple[float, float],
    recorder: Recorder,
) -> None:
    h, w = img.shape[:2]
    z = max(1, state.zoom)

    src_h, src_w = thermal.display_source.shape[:2]
    center_x_src = src_w // 2
    center_y_src = src_h // 2
    center_x = int(center_x_src * z)
    center_y = int(center_y_src * z)

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
        center_label = f"Signal {center_signal:.0f}" if state.show_labels else None

    draw_crosshair(img, center_x, center_y, center_label)

    if state.show_labels:
        if temp is not None and avg_temp is not None:
            if max_temp is not None and max_xy_temp is not None and max_temp >= avg_temp + state.threshold_c:
                x, y = int(max_xy_temp[0] * z), int(max_xy_temp[1] * z)
                cv2.circle(img, (x, y), max(5, z * 2), (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(img, (x, y), max(4, z * 2 - 1), (255, 255, 255), 1, cv2.LINE_AA)
                put_text(img, "HOT " + fmt_temp(max_temp, state.unit, approx), (x + 10, max(18, y - 8)), scale=0.5)

            if min_temp is not None and min_xy_temp is not None and min_temp <= avg_temp - state.threshold_c:
                x, y = int(min_xy_temp[0] * z), int(min_xy_temp[1] * z)
                cv2.circle(img, (x, y), max(5, z * 2), (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(img, (x, y), max(4, z * 2 - 1), (255, 255, 255), 1, cv2.LINE_AA)
                put_text(img, "COLD " + fmt_temp(min_temp, state.unit, approx), (x + 10, min(h - 12, y + 18)), scale=0.5)
        elif avg_signal is not None:
            if max_signal is not None and max_xy_signal is not None and max_signal >= avg_signal + state.signal_threshold:
                x, y = int(max_xy_signal[0] * z), int(max_xy_signal[1] * z)
                cv2.circle(img, (x, y), max(5, z * 2), (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(img, (x, y), max(4, z * 2 - 1), (255, 255, 255), 1, cv2.LINE_AA)
                put_text(img, f"HOT signal {max_signal:.0f}", (x + 10, max(18, y - 8)), scale=0.5)

            if min_signal is not None and min_xy_signal is not None and min_signal <= avg_signal - state.signal_threshold:
                x, y = int(min_xy_signal[0] * z), int(min_xy_signal[1] * z)
                cv2.circle(img, (x, y), max(5, z * 2), (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(img, (x, y), max(4, z * 2 - 1), (255, 255, 255), 1, cv2.LINE_AA)
                put_text(img, f"COLD signal {min_signal:.0f}", (x + 10, min(h - 12, y + 18)), scale=0.5)

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
        f"Blur: {state.blur} | Zoom: {state.zoom}x | Rotate: {state.rotate} | FlipH/V: {state.flip_h}/{state.flip_v}",
        f"Labels: {'ON' if state.show_labels else 'OFF'} | HUD: ON | Recording: {rec_text}",
        "Keys: q quit | c color | +/- contrast | z/x zoom | b/n blur | o/p rotate | f fullscreen",
        "      h HUD | l labels | r record | s snapshot | t/g threshold | u C/F | i invert | m/v flip",
    ]

    if thermal.warning:
        lines.insert(1, thermal.warning)

    x0, y0 = 10, 22
    for i, line in enumerate(lines):
        put_text(img, line, (x0, y0 + i * 20), scale=0.5)


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

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live false-color viewer for TOPDON TC001/TC001 Plus thermal camera on Windows."
    )
    parser.add_argument("-d", "--device", type=int, default=None, help="OpenCV video device number, e.g. 0, 1, 2.")
    parser.add_argument("--list", action="store_true", help="Scan and list readable OpenCV video device indexes.")
    parser.add_argument("--scan-max", type=int, default=3, help="Highest camera index to scan with --list. Default: 3.")
    parser.add_argument("--backend", choices=["auto", "msmf", "dshow", "any"], default="auto", help="OpenCV backend. Default: auto.")
    parser.add_argument("--width", type=int, default=TC001_WIDTH, help="Raw thermal width. Default: 256.")
    parser.add_argument("--height", type=int, default=TC001_HEIGHT, help="Raw thermal height. Default: 192.")
    parser.add_argument("--fps", type=int, default=TC001_FPS, help="Requested FPS. Default: 25.")
    parser.add_argument("--zoom", type=int, default=1, help="Initial zoom factor. Default: 1/no zoom.")
    parser.add_argument("--contrast", type=float, default=1.4, help="Initial contrast. Default: 1.4.")
    parser.add_argument("--blur", type=int, default=0, help="Initial Gaussian blur kernel size. Default: 0/off.")
    parser.add_argument("--threshold", type=float, default=3.0, help="Hot/cold threshold from scene average in Celsius. Default: 3.0.")
    parser.add_argument("--signal-threshold", type=float, default=20.0, help="Hot/cold threshold in visual-only brightness units. Default: 20.")
    parser.add_argument("--fallback-min-c", type=float, default=20.0, help="Approx Celsius min for --estimate-temps mode.")
    parser.add_argument("--fallback-max-c", type=float, default=45.0, help="Approx Celsius max for --estimate-temps mode.")
    parser.add_argument("--estimate-temps", action="store_true", help="Estimate temperatures from 8-bit visual frames. Not calibrated.")
    parser.add_argument("--temp-scale", type=float, default=1.0, help="Calibration scale applied to decoded/estimated Celsius. Default: 1.0.")
    parser.add_argument("--temp-offset", type=float, default=0.0, help="Calibration offset in Celsius. Default: 0.0.")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0, help="Initial clockwise rotation. Default: 0.")
    parser.add_argument("--flip-h", action="store_true", help="Initial horizontal mirror flip.")
    parser.add_argument("--flip-v", action="store_true", help="Initial vertical flip.")
    parser.add_argument("--no-raw-request", action="store_true", help="Do not request raw TC001-sized stream; use normal camera mode.")
    parser.add_argument("--probe", action="store_true", help="Print first-frame information after opening the device.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        list_video_devices(args.scan_max, args.width, args.height, args.fps, args.backend)
        return 0

    device = args.device
    if device is None:
        print("No --device was provided.")
        print("Use --list first, then run for example: py tc001_thermal_viewer_v3.py --device 1")
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
        rotate=int(args.rotate),
        flip_h=bool(args.flip_h),
        flip_v=bool(args.flip_v),
    )
    recorder = Recorder()

    print(f"Opening video device {device} using backend={args.backend}...")
    cap, used_backend = open_camera(
        device,
        args.width,
        args.height,
        args.fps,
        backend=args.backend,
        request_raw=not args.no_raw_request,
    )

    if not cap.isOpened():
        print(f"ERROR: Could not open video device {device}.")
        print("Possible causes:")
        print("  - The TC001/TC001 Plus is not plugged in.")
        print("  - The selected --device number is wrong. Run: py tc001_thermal_viewer_v3.py --list")
        print("  - Another app is using the camera. Close TOPDON app, Windows Camera, OBS, Teams, etc.")
        print("  - Try a different backend: --backend msmf or --backend dshow")
        return 1

    print(f"Opened device {device} with backend={used_backend}.")

    if args.probe:
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
            print("Probe warning: camera opened, but first frame could not be read.")

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
                    print("Try another --device number, unplug/replug camera, close other camera apps,")
                    print("or run with --backend msmf / --backend dshow.")
                    break
                time.sleep(0.03)
                continue

            read_failures = 0

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
            thermal = apply_orientation(thermal, state)
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
            if not handle_key(key, state, recorder, heatmap, args.fps):
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
