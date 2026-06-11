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
import json
import platform
import sys
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List, Any

import cv2
import numpy as np

from tc001_face import DigitalFaceDetector, FaceBox, FaceTrack, FaceTracker, update_face_tracks
from tc001_sdk import TC001SdkCamera, TC001SdkFrame, TC001_FPS, TC001_HEIGHT, TC001_WIDTH

try:
    cv2.setLogLevel(0)
except Exception:
    pass


WINDOW_NAME = "TOPDON TC001 Thermal Viewer v5"

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
    display_source: np.ndarray
    temp_c: Optional[np.ndarray]
    mode: str
    warning: Optional[str] = None
    approx_temps: bool = False


@dataclass
class ViewerState:
    cmap_index: int = 0
    contrast: float = 1.4
    blur: int = 0
    zoom: int = 1
    fullscreen: bool = False
    show_hud: bool = True
    show_labels: bool = True
    invert: bool = False
    unit: str = "C"
    threshold_c: float = 3.0
    signal_threshold: float = 20.0
    recording: bool = False
    rotate: int = 0
    flip_h: bool = False
    flip_v: bool = False
    split_mode: int = 0  # 0=both, 1=left, 2=right
    fit_window: bool = True
    show_help_overlay: bool = False


@dataclass
class Recorder:
    writer: Optional[cv2.VideoWriter] = None
    filename: Optional[str] = None
    frame_size: Optional[Tuple[int, int]] = None
    start_time: float = 0.0


@dataclass
class OpenOptions:
    request_raw: bool = False
    raw_format: bool = False
    fourcc: Optional[str] = None
    raw_height_factor: int = 2


@dataclass
class Alignment:
    mode: str
    matrix: Optional[np.ndarray] = None
    digital_size: Optional[Tuple[int, int]] = None
    thermal_size: Optional[Tuple[int, int]] = None


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
        # DSHOW is often better when requesting raw formats.
        # MSMF is often better for normal visual frames.
        return [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF), ("any", cv2.CAP_ANY)]

    return [("any", cv2.CAP_ANY)]


def fourcc_to_int(code: str) -> int:
    code = (code + "    ")[:4]
    return cv2.VideoWriter_fourcc(code[0], code[1], code[2], code[3])


def describe_fourcc(value: float) -> str:
    try:
        v = int(value)
        chars = [chr((v >> 8 * i) & 0xFF) for i in range(4)]
        text = "".join(c if c.isprintable() else "." for c in chars)
        return text
    except Exception:
        return "????"


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


def is_plausible_temperature_matrix(temp_c: np.ndarray) -> bool:
    if temp_c is None or temp_c.size == 0:
        return False
    if not np.isfinite(temp_c).any():
        return False

    p01, p50, p99 = np.nanpercentile(temp_c, [1, 50, 99])
    p01 = float(p01)
    p50 = float(p50)
    p99 = float(p99)

    # A practical human-room range. Wider than normal to allow hot/cold objects,
    # but narrow enough to reject random BGR bytes interpreted as temperature.
    if not (-60.0 <= p01 <= 200.0 and -60.0 <= p50 <= 200.0 and -60.0 <= p99 <= 300.0):
        return False
    if p99 <= p01:
        return False
    if (p99 - p01) > 220.0:
        return False
    return True


def uint16_le_to_temp_c(values: np.ndarray) -> np.ndarray:
    # TOPDON-style raw conversion: raw_uint16 / 64 - 273.15
    return values.astype(np.float32) / 64.0 - 273.15


def decode_raw_uint8_candidates(arr: np.ndarray, width: int, height: int) -> Optional[Tuple[np.ndarray, str]]:
    """
    Decode raw/radiometric bytes if OpenCV returns an unconverted stream.

    Supported candidate layouts:
    1. TC001 stacked stream: 256 x 384 x 2 bytes, bottom 256 x 192 contains temperature.
    2. Single Y16 stream: 256 x 192 x 2 bytes contains temperature.
    3. Stacked Y16 stream: 256 x 384 x 2 bytes interpreted as uint16, bottom half contains temperature.

    This function deliberately rejects ordinary BGR visual frames.
    """
    if arr.dtype != np.uint8:
        return None

    # Do not reinterpret normal 3-channel BGR visual frames as raw bytes.
    if arr.ndim == 3 and arr.shape[2] == 3:
        return None

    data = np.ascontiguousarray(arr).reshape(-1)
    single_bytes = width * height * 2
    stacked_bytes = width * height * 2 * 2

    candidates: List[Tuple[str, np.ndarray]] = []

    # Candidate A: TC001 stacked byte stream, bottom half as low/high bytes.
    if data.size >= stacked_bytes:
        raw = data[:stacked_bytes].reshape((height * 2, width, 2))
        temp_part = raw[height : height * 2, :, :]
        low = temp_part[:, :, 0].astype(np.float32)
        high = temp_part[:, :, 1].astype(np.float32)
        temp_c = low / 64.0 + high * 4.0 - 273.15
        candidates.append(("raw-tc001-stacked-u8", temp_c))

    # Candidate B: Single-frame Y16 little-endian radiometric image.
    if data.size >= single_bytes:
        values = np.frombuffer(data[:single_bytes].tobytes(), dtype="<u2").reshape((height, width))
        candidates.append(("raw-y16-single", uint16_le_to_temp_c(values)))

    # Candidate C: Stacked Y16, bottom half.
    if data.size >= stacked_bytes:
        values = np.frombuffer(data[:stacked_bytes].tobytes(), dtype="<u2").reshape((height * 2, width))
        candidates.append(("raw-y16-stacked-bottom", uint16_le_to_temp_c(values[height : height * 2, :])))

    for name, temp_c in candidates:
        if is_plausible_temperature_matrix(temp_c):
            return temp_c, name

    return None


def decode_raw_uint16_candidate(arr: np.ndarray, width: int, height: int) -> Optional[Tuple[np.ndarray, str]]:
    if arr.dtype != np.uint16:
        return None

    mat = arr[:, :, 0] if arr.ndim == 3 else arr

    if mat.shape[:2] == (height, width):
        temp_c = uint16_le_to_temp_c(mat)
        if is_plausible_temperature_matrix(temp_c):
            return temp_c, "raw16-single"

    if mat.shape[:2] == (height * 2, width):
        temp_c = uint16_le_to_temp_c(mat[height : height * 2, :])
        if is_plausible_temperature_matrix(temp_c):
            return temp_c, "raw16-stacked-bottom"

    # Last resort: resize/crop only if the data is already uint16.
    cropped = crop_or_resize_matrix(mat, width, height)
    temp_c = uint16_le_to_temp_c(cropped)
    if is_plausible_temperature_matrix(temp_c):
        return temp_c, "raw16-resized"

    return None


def open_camera(
    index: int,
    width: int,
    height: int,
    fps: int,
    backend: str = "auto",
    options: Optional[OpenOptions] = None,
) -> Tuple[cv2.VideoCapture, str]:
    options = options or OpenOptions()
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

        if options.request_raw:
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height * int(options.raw_height_factor))
            cap.set(cv2.CAP_PROP_FPS, fps)

            if options.fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, fourcc_to_int(options.fourcc))

            # CAP_PROP_FORMAT = -1 asks OpenCV for undecoded raw bytes when supported.
            if options.raw_format:
                cap.set(cv2.CAP_PROP_FORMAT, -1)
        else:
            cap.set(cv2.CAP_PROP_FPS, fps)

        for _ in range(12):
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap, backend_name
            time.sleep(0.03)

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
        hint = " | possible 2-byte raw/YUYV frame"
    elif len(shape) == 2:
        hint = " | grayscale/single-channel/raw-byte frame"

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
            options=OpenOptions(request_raw=False),
        )

        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                found.append(idx)
                print(f"  [{idx}] OK  backend={used_backend}  {describe_frame(frame)}")
                try:
                    print(
                        f"      properties: W={cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}, "
                        f"H={cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}, "
                        f"FPS={cap.get(cv2.CAP_PROP_FPS):.1f}, "
                        f"FOURCC={describe_fourcc(cap.get(cv2.CAP_PROP_FOURCC))}"
                    )
                except Exception:
                    pass
        cap.release()

    if not found:
        print("No readable OpenCV video devices found.")
        print("Check USB connection, close other camera apps, or try another USB port/cable.")
        return

    print("\nTip:")
    print("  - shape=(480, 640, 3) means OpenCV is receiving a normal 8-bit visual frame.")
    print("  - A visual frame can be displayed as a heatmap, but it does not contain reliable real temperature data.")
    print("  - Run --find-raw to check whether any raw/radiometric stream is exposed.")


def crop_or_resize_matrix(src: np.ndarray, width: int, height: int) -> np.ndarray:
    mat = src[:, :, 0] if src.ndim == 3 else src
    h, w = mat.shape[:2]

    if h >= height and w >= width:
        y0 = max(0, (h - height) // 2)
        x0 = max(0, (w - width) // 2)
        mat = mat[y0 : y0 + height, x0 : x0 + width]
    elif h != height or w != width:
        mat = cv2.resize(mat, (width, height), interpolation=cv2.INTER_AREA)

    return mat


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

    raw16 = decode_raw_uint16_candidate(arr, width, height)
    if raw16 is not None:
        temp_c, mode = raw16
        temp_c = temp_c * temp_scale + temp_offset
        return ThermalFrame(display_source=temp_c, temp_c=temp_c, mode=mode)

    raw8 = decode_raw_uint8_candidates(arr, width, height)
    if raw8 is not None:
        temp_c, mode = raw8
        temp_c = temp_c * temp_scale + temp_offset
        return ThermalFrame(display_source=temp_c, temp_c=temp_c, mode=mode)

    # Normal visual/BGR/gray frame. Use the full received resolution.
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            gray = arr[:, :, 0]
        elif arr.shape[2] == 2:
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
            warning="Visual estimate: approximate only. Not real radiometric TC001 temperature.",
            approx_temps=True,
        )

    return ThermalFrame(
        display_source=gray,
        temp_c=None,
        mode="visual-only",
        warning="Visual-only feed: OpenCV is not receiving real radiometric temperature data.",
    )

def split_frame_if_needed(frame: np.ndarray, split_mode: int) -> np.ndarray:
    if split_mode == 0 or frame is None or frame.ndim < 2:
        return frame
    h, w = frame.shape[:2]
    half_w = w // 2
    if split_mode == 1:
        # Left half (Thermal on TC001 Plus)
        cropped = frame[:, :half_w]
        return cv2.resize(cropped, (half_w, h // 2)) if h == 480 else cropped
    elif split_mode == 2:
        # Right half (Visual on TC001 Plus)
        cropped = frame[:, half_w:]
        return cv2.resize(cropped, (half_w, h // 2)) if h == 480 else cropped
    return frame


def extract_digital_frame(frame: np.ndarray, split: str) -> np.ndarray:
    """Return the RGB/visual part of the TC001 Plus OpenCV feed."""
    if frame is None or frame.ndim < 2:
        return frame
    split = (split or "right").lower()
    h, w = frame.shape[:2]
    half_w = w // 2

    if split == "none":
        return frame
    if split == "left":
        cropped = frame[:, :half_w]
    elif split == "right":
        cropped = frame[:, half_w:]
    else:
        cropped = frame

    # TC001 Plus side-by-side preview is commonly 640x480 while each half is
    # effectively 320x240. Keep detector input in the natural aspect ratio.
    if h == 480 and split in ("left", "right"):
        return cv2.resize(cropped, (half_w, h // 2), interpolation=cv2.INTER_AREA)
    return cropped


def orient_digital_frame(frame: Optional[np.ndarray], rotate: int, flip_h: bool, flip_v: bool) -> Optional[np.ndarray]:
    if frame is None:
        return None
    out = frame
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


def load_alignment(path: str, mode: str) -> Alignment:
    if mode == "simple-scale":
        return Alignment(mode="simple-scale")
    if not path or not os.path.exists(path):
        return Alignment(mode="missing")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        matrix = np.asarray(data.get("homography"), dtype=np.float32)
        if matrix.shape != (3, 3):
            raise ValueError("homography must be a 3x3 matrix")
        digital_size = tuple(data.get("digital_size", []))
        thermal_size = tuple(data.get("thermal_size", []))
        return Alignment(
            mode="homography",
            matrix=matrix,
            digital_size=(int(digital_size[0]), int(digital_size[1])) if len(digital_size) == 2 else None,
            thermal_size=(int(thermal_size[0]), int(thermal_size[1])) if len(thermal_size) == 2 else None,
        )
    except Exception as exc:
        print(f"WARNING: Could not load alignment file {path!r}: {exc}")
        return Alignment(mode="invalid")


def save_alignment(path: str, matrix: np.ndarray, digital_size: Tuple[int, int], thermal_size: Tuple[int, int]) -> None:
    data = {
        "version": 1,
        "type": "digital_to_thermal_homography",
        "digital_size": [int(digital_size[0]), int(digital_size[1])],
        "thermal_size": [int(thermal_size[0]), int(thermal_size[1])],
        "homography": matrix.astype(float).tolist(),
        "created_at": timestamp(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def map_points_to_thermal(points: np.ndarray, digital_shape: Tuple[int, int], thermal_shape: Tuple[int, int], alignment: Alignment) -> Optional[np.ndarray]:
    if points.size == 0:
        return None
    src_h, src_w = digital_shape[:2]
    dst_h, dst_w = thermal_shape[:2]
    pts = points.astype(np.float32).reshape(-1, 2)

    if alignment.mode == "homography" and alignment.matrix is not None:
        mapped = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), alignment.matrix).reshape(-1, 2)
    elif alignment.mode in ("simple-scale", "missing", "invalid"):
        sx = dst_w / max(src_w, 1)
        sy = dst_h / max(src_h, 1)
        mapped = pts.copy()
        mapped[:, 0] *= sx
        mapped[:, 1] *= sy
    else:
        return None

    mapped[:, 0] = np.clip(mapped[:, 0], 0, max(dst_w - 1, 0))
    mapped[:, 1] = np.clip(mapped[:, 1], 0, max(dst_h - 1, 0))
    return mapped


def face_box_points(box: FaceBox) -> np.ndarray:
    x0, y0 = box.x, box.y
    x1, y1 = box.x + box.w, box.y + box.h
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def orient_points(points: np.ndarray, source_shape: Tuple[int, int], state: ViewerState) -> np.ndarray:
    h, w = source_shape[:2]
    out = points.astype(np.float32).copy()
    rotate = state.rotate % 360

    if rotate == 90:
        x = out[:, 0].copy()
        y = out[:, 1].copy()
        out[:, 0] = h - 1 - y
        out[:, 1] = x
        w, h = h, w
    elif rotate == 180:
        out[:, 0] = w - 1 - out[:, 0]
        out[:, 1] = h - 1 - out[:, 1]
    elif rotate == 270:
        x = out[:, 0].copy()
        y = out[:, 1].copy()
        out[:, 0] = y
        out[:, 1] = w - 1 - x
        w, h = h, w

    if state.flip_h:
        out[:, 0] = w - 1 - out[:, 0]
    if state.flip_v:
        out[:, 1] = h - 1 - out[:, 1]
    return out


def inverse_orient_points(points: np.ndarray, raw_shape: Tuple[int, int], state: ViewerState) -> np.ndarray:
    raw_h, raw_w = raw_shape[:2]
    rotate = state.rotate % 360
    if rotate in (90, 270):
        oriented_w, oriented_h = raw_h, raw_w
    else:
        oriented_w, oriented_h = raw_w, raw_h

    out = points.astype(np.float32).copy()
    if state.flip_h:
        out[:, 0] = oriented_w - 1 - out[:, 0]
    if state.flip_v:
        out[:, 1] = oriented_h - 1 - out[:, 1]

    if rotate == 90:
        x = out[:, 0].copy()
        y = out[:, 1].copy()
        out[:, 0] = y
        out[:, 1] = raw_h - 1 - x
    elif rotate == 180:
        out[:, 0] = raw_w - 1 - out[:, 0]
        out[:, 1] = raw_h - 1 - out[:, 1]
    elif rotate == 270:
        x = out[:, 0].copy()
        y = out[:, 1].copy()
        out[:, 0] = raw_w - 1 - y
        out[:, 1] = x

    out[:, 0] = np.clip(out[:, 0], 0, max(raw_w - 1, 0))
    out[:, 1] = np.clip(out[:, 1], 0, max(raw_h - 1, 0))
    return out


def polygon_max_temperature_c(temp_c: Optional[np.ndarray], polygon: np.ndarray) -> Optional[float]:
    if temp_c is None or temp_c.size == 0 or polygon.size == 0:
        return None
    mask = np.zeros(temp_c.shape[:2], dtype=np.uint8)
    pts = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    values = temp_c[mask > 0]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.nanmax(values))


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
    face_temp_c = polygon_max_temperature_c(raw_thermal.temp_c, mapped)
    if face_temp_c is not None:
        label += f" | max {fmt_temp(face_temp_c, state.unit, raw_thermal.approx_temps)}"
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
    put_text(out, f"Digital source: {source} | rotate {rotate} | {out.shape[1]}x{out.shape[0]} | alignment: {alignment.mode}", (8, out.shape[0] - 12), scale=0.42)
    return out


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

    if state.zoom != 1 and not state.fit_window:
        interpolation = cv2.INTER_NEAREST if state.zoom >= 2 else cv2.INTER_AREA
        heatmap = cv2.resize(
            heatmap,
            (heatmap.shape[1] * state.zoom, heatmap.shape[0] * state.zoom),
            interpolation=interpolation,
        )

    return heatmap, (vmin, vmax)


def resize_to_window(img: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return img
    try:
        _x, _y, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
    except cv2.error:
        return img
    if win_w <= 1 or win_h <= 1:
        return img
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img
    scale = max(win_w / w, win_h / h)
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(img, (win_w, win_h), interpolation=interpolation)


def put_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float = 0.55,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness_bg = max(2, int(round(scale * 4.0)))
    cv2.putText(img, text, org, font, scale, bg, thickness_bg, cv2.LINE_AA)
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
    parser.add_argument("--face-model", choices=["auto", "mediapipe", "haar"], default="auto", help="Face detector backend. Default: auto.")
    parser.add_argument("--face-min-confidence", type=float, default=0.55, help="Minimum MediaPipe face confidence. Default: 0.55.")
    parser.add_argument("--face-detect-interval", type=int, default=3, help="Run face detection every N frames, then track/smooth between detections. Default: 3.")
    parser.add_argument("--face-hold-frames", type=int, default=15, help="Keep the last face box for this many frames after missed detections. Default: 15.")
    parser.add_argument("--face-smoothing", type=float, default=0.65, help="EMA smoothing for face box coordinates. Default: 0.65.")
    parser.add_argument("--max-faces", type=int, default=5, help="Maximum number of people/face tracks to show. Default: 5.")
    parser.add_argument("--head-fallback", choices=["off", "auto", "always"], default="auto", help="Use thermal/bright-region head fallback. Default: auto.")
    parser.add_argument("--head-confirm-frames", type=int, default=2, help="Detection frames required before showing a HEAD TRACK on thermal view. Default: 2.")
    parser.add_argument("--min-head-confidence", type=float, default=0.65, help="Assigned confidence for accepted head fallback boxes. Default: 0.65.")
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
        face_detector = DigitalFaceDetector(args.face_model, args.face_min_confidence)
        print(
            f"Face detection enabled: detector={face_detector.name}, "
            f"digital_source={args.digital_source}, alignment={alignment.mode}, "
            f"max_faces={args.max_faces}, head_fallback={args.head_fallback}."
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
        if sdk_cam is not None:
            sdk_cam.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
