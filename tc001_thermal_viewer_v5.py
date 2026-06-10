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
    py tc001_thermal_viewer_v5.py --find-raw --scan-min 1 --scan-max 5

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
import ctypes
import json
import platform
import sys
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List, Dict, Any

import cv2
import numpy as np

try:
    cv2.setLogLevel(0)
except Exception:
    pass


WINDOW_NAME = "TOPDON TC001 Thermal Viewer v5"

# TC001 / TC001 Plus commonly uses a 256 x 192 thermal sensor.
TC001_WIDTH = 256
TC001_HEIGHT = 192
TC001_FPS = 25
DEFAULT_OPENCV_CAMERA_INDEX = 1

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


class TC001SdkCamera:
    """Read real TC001 Plus radiometric frames through TOPDON's libiruvc.dll."""

    WIDTH = TC001_WIDTH
    HEIGHT = TC001_HEIGHT * 2
    FRAME_SIZE = WIDTH * HEIGHT * 2
    FPS = TC001_FPS

    def __init__(self, dll_dir: str) -> None:
        self.dll_dir = dll_dir
        self.app_dir = os.path.dirname(os.path.dirname(dll_dir))
        self.dll: Optional[ctypes.CDLL] = None
        self.dev = ctypes.create_string_buffer(8192)
        self.info = ctypes.create_string_buffer(8192)
        self.stream = ctypes.create_string_buffer(0x40)
        self.format_name = ctypes.create_string_buffer(b"YUY2")
        self.frame_buf: Optional[int] = None
        self.raw_buf = ctypes.create_string_buffer(self.FRAME_SIZE)

    def _check(self, code: int, name: str) -> None:
        if code < 0:
            raise RuntimeError(f"{name} failed: {code}")

    def open(self) -> None:
        if not os.path.exists(os.path.join(self.dll_dir, "libiruvc.dll")):
            raise RuntimeError(f"libiruvc.dll was not found in: {self.dll_dir}")

        os.add_dll_directory(self.dll_dir)
        if os.path.isdir(self.app_dir):
            os.add_dll_directory(self.app_dir)

        dll = ctypes.CDLL(os.path.join(self.dll_dir, "libiruvc.dll"))
        self.dll = dll

        dll.uvc_camera_init.restype = ctypes.c_int
        dll.uvc_camera_list.argtypes = [ctypes.c_void_p]
        dll.uvc_camera_list.restype = ctypes.c_int
        dll.uvc_camera_info_get.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        dll.uvc_camera_info_get.restype = ctypes.c_int
        dll.uvc_camera_open.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        dll.uvc_camera_open.restype = ctypes.c_int
        dll.uvc_camera_stream_start.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        dll.uvc_camera_stream_start.restype = ctypes.c_int
        dll.uvc_frame_buf_create.argtypes = [ctypes.c_void_p]
        dll.uvc_frame_buf_create.restype = ctypes.c_void_p
        dll.uvc_frame_get.argtypes = [ctypes.c_void_p]
        dll.uvc_frame_get.restype = ctypes.c_int
        dll.uvc_camera_stream_close.restype = ctypes.c_int
        dll.uvc_frame_buf_release.argtypes = [ctypes.c_void_p]
        dll.uvc_frame_buf_release.restype = ctypes.c_int
        dll.uvc_camera_close.restype = ctypes.c_int
        dll.uvc_camera_release.restype = ctypes.c_int

        ctypes.c_void_p.from_buffer(self.stream, 0x10).value = ctypes.addressof(self.format_name)
        ctypes.c_int.from_buffer(self.stream, 0x18).value = self.WIDTH
        ctypes.c_int.from_buffer(self.stream, 0x1C).value = self.HEIGHT
        ctypes.c_int.from_buffer(self.stream, 0x20).value = self.FRAME_SIZE
        ctypes.c_int.from_buffer(self.stream, 0x24).value = self.FPS
        ctypes.c_int.from_buffer(self.stream, 0x28).value = self.FRAME_SIZE

        self._check(dll.uvc_camera_init(), "uvc_camera_init")
        self._check(dll.uvc_camera_list(ctypes.byref(self.dev)), "uvc_camera_list")
        self._check(dll.uvc_camera_info_get(ctypes.byref(self.dev), ctypes.byref(self.info)), "uvc_camera_info_get")

        # Offset 160 is the SDK's 256x384@25 mode in the info table.
        self._check(dll.uvc_camera_open(ctypes.byref(self.dev), ctypes.byref(self.info, 160)), "uvc_camera_open")

        self.frame_buf = dll.uvc_frame_buf_create(ctypes.byref(self.stream))
        if not self.frame_buf:
            raise RuntimeError("uvc_frame_buf_create failed")

        self._check(dll.uvc_camera_stream_start(ctypes.byref(self.stream), None), "uvc_camera_stream_start")

    def read_temp_c(self, timeout_s: float = 1.0) -> Optional[np.ndarray]:
        if self.dll is None:
            raise RuntimeError("SDK camera is not open")
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            ret = self.dll.uvc_frame_get(ctypes.byref(self.raw_buf))
            if ret == 0:
                raw = np.frombuffer(bytes(self.raw_buf), dtype="<u2").reshape((self.HEIGHT, self.WIDTH))
                temp_raw = raw[TC001_HEIGHT : self.HEIGHT, :]
                return temp_raw.astype(np.float32) / 64.0 - 273.15
            time.sleep(0.005)
        return None

    def close(self) -> None:
        if self.dll is None:
            return
        try:
            self.dll.uvc_camera_stream_close()
        except Exception:
            pass
        if self.frame_buf:
            try:
                self.dll.uvc_frame_buf_release(ctypes.c_void_p(self.frame_buf))
            except Exception:
                pass
        try:
            self.dll.uvc_camera_close()
        except Exception:
            pass
        try:
            self.dll.uvc_camera_release()
        except Exception:
            pass
        self.dll = None


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
    min_index: int = DEFAULT_OPENCV_CAMERA_INDEX,
    max_index: int = 3,
    width: int = TC001_WIDTH,
    height: int = TC001_HEIGHT,
    fps: int = TC001_FPS,
    backend: str = "auto",
) -> None:
    print("Scanning video device indexes...")
    found = []

    for idx in range(max(1, int(min_index)), max_index + 1):
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


@dataclass
class ForeheadMeasurement:
    rgb_poly: np.ndarray
    thermal_poly: np.ndarray
    temp_c: Optional[float]
    status: str
    calibrated: bool


@dataclass
class HotObjectMeasurement:
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    max_c: float
    avg_c: float
    area: int
    persistent_s: float
    label: Optional[str]
    confidence: Optional[float]
    alert: bool


@dataclass
class AIOverlay:
    forehead: Optional[ForeheadMeasurement]
    hot_objects: List[HotObjectMeasurement]
    status_lines: List[str]


class AlignmentMap:
    def __init__(
        self,
        matrix: Optional[np.ndarray],
        rgb_size: Optional[Tuple[int, int]],
        thermal_size: Optional[Tuple[int, int]],
        source: str,
    ) -> None:
        self.matrix = matrix
        self.rgb_size = rgb_size
        self.thermal_size = thermal_size
        self.source = source

    @property
    def calibrated(self) -> bool:
        return self.matrix is not None

    @classmethod
    def load(cls, path: str) -> "AlignmentMap":
        if not os.path.exists(path):
            return cls(None, None, None, "rough-scale")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        matrix = np.asarray(data.get("homography"), dtype=np.float32)
        if matrix.shape != (3, 3):
            raise ValueError(f"Invalid homography in {path}")
        rgb_size_data = data.get("rgb_size")
        thermal_size_data = data.get("thermal_size")
        rgb_size = tuple(int(v) for v in rgb_size_data) if rgb_size_data else None
        thermal_size = tuple(int(v) for v in thermal_size_data) if thermal_size_data else None
        return cls(matrix, rgb_size, thermal_size, path)

    def map_rgb_to_thermal(
        self,
        points: np.ndarray,
        rgb_shape: Tuple[int, int],
        thermal_shape: Tuple[int, int],
    ) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))
        th, tw = thermal_shape[:2]
        rh, rw = rgb_shape[:2]

        if self.matrix is not None:
            mapped = cv2.perspectiveTransform(pts, self.matrix).reshape((-1, 2))
        else:
            sx = tw / max(rw, 1)
            sy = th / max(rh, 1)
            mapped = points.astype(np.float32).copy()
            mapped[:, 0] *= sx
            mapped[:, 1] *= sy

        mapped[:, 0] = np.clip(mapped[:, 0], 0, max(0, tw - 1))
        mapped[:, 1] = np.clip(mapped[:, 1], 0, max(0, th - 1))
        return mapped

    def map_thermal_to_rgb(
        self,
        points: np.ndarray,
        rgb_shape: Tuple[int, int],
        thermal_shape: Tuple[int, int],
    ) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))
        th, tw = thermal_shape[:2]
        rh, rw = rgb_shape[:2]

        if self.matrix is not None:
            inv = np.linalg.inv(self.matrix)
            mapped = cv2.perspectiveTransform(pts, inv.astype(np.float32)).reshape((-1, 2))
        else:
            sx = rw / max(tw, 1)
            sy = rh / max(th, 1)
            mapped = points.astype(np.float32).copy()
            mapped[:, 0] *= sx
            mapped[:, 1] *= sy

        mapped[:, 0] = np.clip(mapped[:, 0], 0, max(0, rw - 1))
        mapped[:, 1] = np.clip(mapped[:, 1], 0, max(0, rh - 1))
        return mapped


class FaceForeheadDetector:
    def __init__(self, max_faces: int = 2) -> None:
        self.error: Optional[str] = None
        self.face_mesh: Optional[Any] = None
        try:
            import mediapipe as mp  # type: ignore

            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_faces,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as exc:
            self.error = f"MediaPipe unavailable: {exc}"

    def detect_forehead_poly(self, bgr_frame: np.ndarray) -> Optional[np.ndarray]:
        if self.face_mesh is None:
            return None

        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        faces = getattr(results, "multi_face_landmarks", None)
        if not faces:
            return None

        best_points: Optional[np.ndarray] = None
        best_area = -1.0
        for face in faces:
            pts = np.asarray([(lm.x * w, lm.y * h) for lm in face.landmark], dtype=np.float32)
            if pts.size == 0:
                continue
            x0, y0 = np.nanmin(pts, axis=0)
            x1, y1 = np.nanmax(pts, axis=0)
            bw = max(1.0, float(x1 - x0))
            bh = max(1.0, float(y1 - y0))
            area = bw * bh
            if area <= best_area:
                continue

            cx = float((x0 + x1) / 2.0)
            top = float(y0 + 0.10 * bh)
            bottom = float(y0 + 0.30 * bh)
            left_top = float(cx - 0.18 * bw)
            right_top = float(cx + 0.18 * bw)
            left_bottom = float(cx - 0.16 * bw)
            right_bottom = float(cx + 0.16 * bw)
            poly = np.asarray(
                [
                    [left_top, top],
                    [right_top, top],
                    [right_bottom, bottom],
                    [left_bottom, bottom],
                ],
                dtype=np.float32,
            )
            poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
            poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
            best_points = poly
            best_area = area

        return best_points


class YoloObjectDetector:
    def __init__(self, model_name: str, enabled: bool) -> None:
        self.model: Optional[Any] = None
        self.names: Dict[int, str] = {}
        self.error: Optional[str] = None
        if not enabled:
            return
        try:
            yolo_config_dir = os.path.abspath(".ultralytics")
            os.makedirs(yolo_config_dir, exist_ok=True)
            os.environ.setdefault("YOLO_CONFIG_DIR", yolo_config_dir)
            from ultralytics import YOLO  # type: ignore

            self.model = YOLO(model_name)
            names = getattr(self.model, "names", {}) or {}
            self.names = {int(k): str(v) for k, v in names.items()} if isinstance(names, dict) else {}
        except Exception as exc:
            self.error = f"YOLO unavailable: {exc}"

    def detect(self, bgr_frame: np.ndarray, conf: float) -> List[Dict[str, Any]]:
        if self.model is None:
            return []
        try:
            results = self.model.predict(bgr_frame, conf=conf, verbose=False)
        except Exception as exc:
            self.error = f"YOLO predict failed: {exc}"
            return []

        detections: List[Dict[str, Any]] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].detach().cpu().numpy().astype(float)
                cls_id = int(box.cls[0].detach().cpu().item())
                score = float(box.conf[0].detach().cpu().item())
                detections.append(
                    {
                        "bbox": (xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        "label": self.names.get(cls_id, str(cls_id)),
                        "confidence": score,
                    }
                )
        return detections


class HotSpotTracker:
    def __init__(self) -> None:
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def update(self, spots: List[Dict[str, Any]], now: float, max_distance: float = 18.0) -> List[Dict[str, Any]]:
        assigned: set[int] = set()

        for spot in spots:
            cx, cy = spot["centroid"]
            best_id: Optional[int] = None
            best_dist = max_distance
            for track_id, track in self.tracks.items():
                if track_id in assigned:
                    continue
                tx, ty = track["centroid"]
                dist = float(np.hypot(cx - tx, cy - ty))
                if dist <= best_dist:
                    best_id = track_id
                    best_dist = dist

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = {"first_seen": now, "last_seen": now, "centroid": (cx, cy)}
            else:
                self.tracks[best_id]["last_seen"] = now
                self.tracks[best_id]["centroid"] = (cx, cy)

            assigned.add(best_id)
            spot["track_id"] = best_id
            spot["persistent_s"] = now - float(self.tracks[best_id]["first_seen"])

        stale = [track_id for track_id, track in self.tracks.items() if now - float(track["last_seen"]) > 2.0]
        for track_id in stale:
            self.tracks.pop(track_id, None)

        return spots


class ThermalAI:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.alignment = AlignmentMap.load(args.alignment_file)
        self.face = FaceForeheadDetector() if args.ai_forehead else None
        self.yolo = YoloObjectDetector(args.ai_object_model, args.hot_object_watch and args.ai_use_yolo)
        self.hot_tracker = HotSpotTracker()
        self.frame_index = 0
        self.last_forehead_poly: Optional[np.ndarray] = None
        self.last_yolo_detections: List[Dict[str, Any]] = []
        self.risk_classes = [
            item.strip().lower()
            for item in str(args.risk_classes).split(",")
            if item.strip()
        ]

    def needs_rgb(self) -> bool:
        face_needs_rgb = self.face is not None and self.face.face_mesh is not None
        yolo_needs_rgb = self.yolo is not None and self.yolo.model is not None
        return bool(face_needs_rgb or yolo_needs_rgb)

    def status_lines(self) -> List[str]:
        lines: List[str] = []
        if self.alignment.calibrated:
            lines.append(f"AI align: {self.alignment.source}")
        else:
            if self.args.thermal_forehead_fallback:
                lines.append("AI align: thermal fallback active")
            else:
                lines.append("AI align: rough scale only; run --calibrate-ai")
        if self.face is not None and self.face.error:
            if self.args.thermal_forehead_fallback:
                lines.append("MediaPipe unavailable; using thermal forehead estimate")
            else:
                lines.append(self.face.error)
        if self.yolo is not None and self.yolo.error:
            lines.append(self.yolo.error)
        return lines

    def analyze(self, rgb_frame: Optional[np.ndarray], temp_c: Optional[np.ndarray]) -> AIOverlay:
        if temp_c is None:
            return AIOverlay(None, [], ["AI needs real temperature matrix"])

        self.frame_index += 1
        forehead = self._measure_forehead(rgb_frame, temp_c) if self.args.ai_forehead else None
        hot_objects = self._measure_hot_objects(rgb_frame, temp_c, forehead) if self.args.hot_object_watch else []
        return AIOverlay(forehead, hot_objects, self.status_lines())

    def _measure_forehead(self, rgb_frame: Optional[np.ndarray], temp_c: np.ndarray) -> Optional[ForeheadMeasurement]:
        thermal_shape = temp_c.shape[:2]

        if self.face is not None and self.face.face_mesh is not None and rgb_frame is not None:
            rgb_shape = rgb_frame.shape[:2]
            if self.frame_index % max(1, int(self.args.ai_every_n)) == 1 or self.last_forehead_poly is None:
                poly = self.face.detect_forehead_poly(rgb_frame)
                if poly is not None:
                    self.last_forehead_poly = poly

            if self.last_forehead_poly is None:
                return ForeheadMeasurement(
                    rgb_poly=np.zeros((0, 2), dtype=np.float32),
                    thermal_poly=np.zeros((0, 2), dtype=np.float32),
                    temp_c=None,
                    status="no-face",
                    calibrated=self.alignment.calibrated,
                )

            thermal_poly = self.alignment.map_rgb_to_thermal(self.last_forehead_poly, rgb_shape, thermal_shape)
            return self._measure_forehead_poly(
                self.last_forehead_poly.copy(),
                thermal_poly,
                temp_c,
                self.alignment.calibrated,
            )

        if self.args.thermal_forehead_fallback:
            return self._measure_forehead_from_thermal(temp_c)

        return None

    def _measure_forehead_from_thermal(self, temp_c: np.ndarray) -> ForeheadMeasurement:
        thermal_shape = temp_c.shape[:2]
        finite = np.isfinite(temp_c)
        mask = np.zeros(thermal_shape, dtype=np.uint8)
        mask[
            finite
            & (temp_c >= float(self.args.thermal_face_min_c))
            & (temp_c <= float(self.args.thermal_face_max_c))
        ] = 255

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        best_idx = -1
        best_score = -1.0
        min_area = int(self.args.thermal_face_min_area)
        for idx in range(1, count):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            if w < 8 or h < 8:
                continue
            values = temp_c[labels == idx]
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            mean_c = float(np.nanmean(values))
            top_bonus = 1.0 + (thermal_shape[0] - y) / max(thermal_shape[0], 1)
            score = area * top_bonus + mean_c * 4.0
            if score > best_score:
                best_idx = idx
                best_score = score

        if best_idx < 0:
            return ForeheadMeasurement(
                rgb_poly=np.zeros((0, 2), dtype=np.float32),
                thermal_poly=np.zeros((0, 2), dtype=np.float32),
                temp_c=None,
                status="thermal-no-face",
                calibrated=False,
            )

        x = int(stats[best_idx, cv2.CC_STAT_LEFT])
        y = int(stats[best_idx, cv2.CC_STAT_TOP])
        w = int(stats[best_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[best_idx, cv2.CC_STAT_HEIGHT])

        head_h = max(10.0, min(float(h), float(w) * 0.90))
        cx = x + w / 2.0
        forehead_top = y + head_h * float(self.args.thermal_forehead_top_ratio)
        forehead_bottom = y + head_h * float(self.args.thermal_forehead_bottom_ratio)
        half_w = max(4.0, w * float(self.args.thermal_forehead_width_ratio) / 2.0)
        thermal_poly = np.asarray(
            [
                [cx - half_w, forehead_top],
                [cx + half_w, forehead_top],
                [cx + half_w * 0.88, forehead_bottom],
                [cx - half_w * 0.88, forehead_bottom],
            ],
            dtype=np.float32,
        )
        thermal_poly[:, 0] = np.clip(thermal_poly[:, 0], 0, thermal_shape[1] - 1)
        thermal_poly[:, 1] = np.clip(thermal_poly[:, 1], 0, thermal_shape[0] - 1)

        return self._measure_forehead_poly(
            np.zeros((0, 2), dtype=np.float32),
            thermal_poly,
            temp_c,
            False,
        )

    def _measure_forehead_poly(
        self,
        rgb_poly: np.ndarray,
        thermal_poly: np.ndarray,
        temp_c: np.ndarray,
        calibrated: bool,
    ) -> ForeheadMeasurement:
        thermal_shape = temp_c.shape[:2]
        mask = np.zeros(thermal_shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(thermal_poly).astype(np.int32), 255)
        values = temp_c[mask > 0]
        values = values[np.isfinite(values)]
        values = values[
            (values >= float(self.args.forehead_min_c))
            & (values <= float(self.args.forehead_max_c))
        ]

        if values.size == 0:
            measured = None
            status = "forehead-no-temp"
        else:
            measured = float(np.nanpercentile(values, float(self.args.forehead_percentile)))
            status = "forehead-alert" if measured >= float(self.args.forehead_threshold_c) else "forehead-ok"

        return ForeheadMeasurement(
            rgb_poly=rgb_poly,
            thermal_poly=thermal_poly,
            temp_c=measured,
            status=status,
            calibrated=calibrated,
        )

    def _measure_hot_objects(
        self,
        rgb_frame: Optional[np.ndarray],
        temp_c: np.ndarray,
        forehead: Optional[ForeheadMeasurement],
    ) -> List[HotObjectMeasurement]:
        threshold = float(self.args.hot_threshold_c)
        mask = np.zeros(temp_c.shape[:2], dtype=np.uint8)
        mask[np.isfinite(temp_c) & (temp_c >= threshold)] = 255

        if forehead is not None and forehead.thermal_poly.size:
            cv2.fillConvexPoly(mask, np.round(forehead.thermal_poly).astype(np.int32), 0)

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        spots: List[Dict[str, Any]] = []
        for idx in range(1, count):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < int(self.args.hot_min_area):
                continue
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            component_values = temp_c[labels == idx]
            component_values = component_values[np.isfinite(component_values)]
            if component_values.size == 0:
                continue
            spots.append(
                {
                    "bbox": (x, y, x + w, y + h),
                    "centroid": (float(centroids[idx][0]), float(centroids[idx][1])),
                    "max_c": float(np.nanmax(component_values)),
                    "avg_c": float(np.nanmean(component_values)),
                    "area": area,
                }
            )

        spots.sort(key=lambda item: item["max_c"], reverse=True)
        spots = spots[: max(1, int(self.args.hot_max_alerts))]
        spots = self.hot_tracker.update(spots, time.time())

        if self.yolo is not None and self.yolo.model is not None and rgb_frame is not None:
            if self.frame_index % max(1, int(self.args.yolo_every_n)) == 1 or not self.last_yolo_detections:
                self.last_yolo_detections = self.yolo.detect(rgb_frame, float(self.args.yolo_conf))

        out: List[HotObjectMeasurement] = []
        for spot in spots:
            label, conf = (None, None)
            if rgb_frame is not None:
                label, conf = self._label_for_hot_spot(spot, rgb_frame.shape[:2], temp_c.shape[:2])
            risk_label = label is not None and any(token in label.lower() for token in self.risk_classes)
            no_yolo_label = self.yolo is None or self.yolo.model is None or label is None
            persistent_s = float(spot.get("persistent_s", 0.0))
            alert = persistent_s >= float(self.args.hot_persist_s) and (risk_label or no_yolo_label)
            out.append(
                HotObjectMeasurement(
                    bbox=spot["bbox"],
                    centroid=spot["centroid"],
                    max_c=float(spot["max_c"]),
                    avg_c=float(spot["avg_c"]),
                    area=int(spot["area"]),
                    persistent_s=persistent_s,
                    label=label,
                    confidence=conf,
                    alert=alert,
                )
            )
        return out

    def _label_for_hot_spot(
        self,
        spot: Dict[str, Any],
        rgb_shape: Tuple[int, int],
        thermal_shape: Tuple[int, int],
    ) -> Tuple[Optional[str], Optional[float]]:
        if not self.last_yolo_detections:
            return None, None

        cx, cy = spot["centroid"]
        rgb_pt = self.alignment.map_thermal_to_rgb(
            np.asarray([[cx, cy]], dtype=np.float32),
            rgb_shape,
            thermal_shape,
        )[0]
        px, py = float(rgb_pt[0]), float(rgb_pt[1])

        best: Optional[Dict[str, Any]] = None
        best_area = float("inf")
        for det in self.last_yolo_detections:
            x0, y0, x1, y1 = det["bbox"]
            if x0 <= px <= x1 and y0 <= py <= y1:
                area = max(1.0, float((x1 - x0) * (y1 - y0)))
                if area < best_area:
                    best = det
                    best_area = area

        if best is None:
            return None, None
        return str(best["label"]), float(best["confidence"])


def draw_poly_on_heatmap(
    img: np.ndarray,
    poly: np.ndarray,
    thermal_shape: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    if poly.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    h, w = img.shape[:2]
    th, tw = thermal_shape[:2]
    display_poly = poly.astype(np.float32).copy()
    display_poly[:, 0] *= w / max(tw, 1)
    display_poly[:, 1] *= h / max(th, 1)
    display_poly_i = np.round(display_poly).astype(np.int32)
    cv2.polylines(img, [display_poly_i], True, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.polylines(img, [display_poly_i], True, color, thickness, cv2.LINE_AA)
    return display_poly_i


def draw_ai_overlay(img: np.ndarray, overlay: AIOverlay, thermal_shape: Tuple[int, int], unit: str) -> None:
    h, w = img.shape[:2]

    if overlay.forehead is not None:
        color = (0, 0, 255) if overlay.forehead.status == "forehead-alert" else (0, 255, 128)
        display_poly = draw_poly_on_heatmap(img, overlay.forehead.thermal_poly, thermal_shape, color, 2)
        if display_poly.size:
            x = int(np.nanmin(display_poly[:, 0]))
            y = int(np.nanmin(display_poly[:, 1]))
            prefix = "Forehead"
            if not overlay.forehead.calibrated:
                prefix += " ~"
            label = f"{prefix}: {fmt_temp(overlay.forehead.temp_c, unit)}"
            put_text(img, label, (max(8, x), max(22, y - 8)), scale=0.50, color=color)
        elif overlay.forehead.status in ("no-face", "thermal-no-face"):
            put_text(img, "AI forehead: no thermal face", (10, h - 18), scale=0.46, color=(0, 255, 255))

    for item in overlay.hot_objects:
        x0, y0, x1, y1 = item.bbox
        poly = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        color = (0, 0, 255) if item.alert else (0, 255, 255)
        display_poly = draw_poly_on_heatmap(img, poly, thermal_shape, color, 2)
        if display_poly.size:
            dx0 = int(np.nanmin(display_poly[:, 0]))
            dy0 = int(np.nanmin(display_poly[:, 1]))
            label = "ALERT" if item.alert else "HOT"
            label += f" {fmt_temp(item.max_c, unit)}"
            if item.label:
                label += f" {item.label}"
            put_text(img, label, (max(8, dx0), max(22, dy0 - 8)), scale=0.46, color=color)

    if overlay.status_lines:
        y = h - 18
        for line in reversed(overlay.status_lines[-3:]):
            put_text(img, line, (10, y), scale=0.38, color=(200, 255, 255))
            y -= 15


def run_ai_calibration(
    rgb_frame: np.ndarray,
    thermal_display_source: np.ndarray,
    output_path: str,
) -> bool:
    rgb_h, rgb_w = rgb_frame.shape[:2]
    th, tw = thermal_display_source.shape[:2]
    thermal_norm = normalize_linear(
        thermal_display_source.astype(np.float32),
        *robust_percentile_range(thermal_display_source.astype(np.float32), 1.0, 99.0),
    )
    thermal_color = cv2.applyColorMap(thermal_norm, cv2.COLORMAP_JET)
    thermal_view = cv2.resize(thermal_color, (rgb_w, rgb_h), interpolation=cv2.INTER_CUBIC)
    rgb_points: List[Tuple[float, float]] = []
    thermal_points: List[Tuple[float, float]] = []
    window = "TC001 AI Calibration"

    def redraw() -> np.ndarray:
        left = rgb_frame.copy()
        right = thermal_view.copy()
        for idx, (x, y) in enumerate(rgb_points):
            cv2.circle(left, (int(x), int(y)), 6, (0, 255, 0), -1, cv2.LINE_AA)
            put_text(left, str(idx + 1), (int(x) + 8, int(y) - 8), scale=0.5, color=(0, 255, 0))
        for idx, (x, y) in enumerate(thermal_points):
            sx = x * rgb_w / max(tw, 1)
            sy = y * rgb_h / max(th, 1)
            cv2.circle(right, (int(sx), int(sy)), 6, (0, 255, 255), -1, cv2.LINE_AA)
            put_text(right, str(idx + 1), (int(sx) + 8, int(sy) - 8), scale=0.5, color=(0, 255, 255))
        canvas = np.hstack([left, right])
        put_text(canvas, "Click 4 RGB points on left, then same 4 thermal points on right", (10, 24), scale=0.55)
        put_text(canvas, "r reset | q/ESC cancel | saves automatically after 8 clicks", (10, 48), scale=0.48)
        return canvas

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(rgb_points) < 4:
            if x < rgb_w and y < rgb_h:
                rgb_points.append((float(x), float(y)))
        elif len(thermal_points) < 4:
            if rgb_w <= x < rgb_w * 2 and y < rgb_h:
                tx = (x - rgb_w) * tw / max(rgb_w, 1)
                ty = y * th / max(rgb_h, 1)
                thermal_points.append((float(tx), float(ty)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    saved = False

    try:
        while True:
            cv2.imshow(window, redraw())
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                rgb_points.clear()
                thermal_points.clear()
            if len(rgb_points) == 4 and len(thermal_points) == 4:
                src = np.asarray(rgb_points, dtype=np.float32)
                dst = np.asarray(thermal_points, dtype=np.float32)
                matrix = cv2.getPerspectiveTransform(src, dst)
                data = {
                    "created_at": timestamp(),
                    "rgb_size": [rgb_w, rgb_h],
                    "thermal_size": [tw, th],
                    "rgb_points": [[float(x), float(y)] for x, y in rgb_points],
                    "thermal_points": [[float(x), float(y)] for x, y in thermal_points],
                    "homography": matrix.astype(float).tolist(),
                }
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                print(f"AI calibration saved: {os.path.abspath(output_path)}")
                saved = True
                break
    finally:
        cv2.destroyWindow(window)

    return saved


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

    for idx in range(max(1, int(args.scan_min)), int(args.scan_max) + 1):
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
    interpolation = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
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
    parser.add_argument("-d", "--device", type=int, default=DEFAULT_OPENCV_CAMERA_INDEX, help="OpenCV video device number. Camera 0 is intentionally blocked. Default: 1.")
    parser.add_argument("--list", action="store_true", help="Scan and list readable OpenCV video device indexes.")
    parser.add_argument("--find-raw", action="store_true", help="Scan camera/backend/format combinations for radiometric temperature data.")
    parser.add_argument("--scan-min", type=int, default=DEFAULT_OPENCV_CAMERA_INDEX, help="Lowest camera index to scan. Camera 0 is intentionally skipped. Default: 1.")
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
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0, help="Initial clockwise rotation. Default: 0.")
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
    parser.add_argument("--ai-forehead", action="store_true", help="Enable RGB AI face detection and forehead temperature ROI.")
    parser.add_argument("--hot-object-watch", action="store_true", help="Enable thermal hot-object detection and optional YOLO labels.")
    parser.add_argument("--calibrate-ai", action="store_true", help="Open a 4-point RGB-to-thermal calibration window and save --alignment-file.")
    parser.add_argument("--alignment-file", default="tc001_alignment.json", help="AI calibration file. Default: tc001_alignment.json.")
    parser.add_argument("--rgb-device", type=int, default=DEFAULT_OPENCV_CAMERA_INDEX, help="OpenCV RGB/visual camera index for AI. Camera 0 is intentionally blocked. Default: 1.")
    parser.add_argument("--rgb-backend", choices=["auto", "msmf", "dshow", "any"], default="auto", help="OpenCV backend for AI RGB camera. Default: auto.")
    parser.add_argument("--rgb-width", type=int, default=640, help="Requested RGB camera width for AI. Default: 640.")
    parser.add_argument("--rgb-height", type=int, default=480, help="Requested RGB camera height for AI. Default: 480.")
    parser.add_argument("--rgb-fps", type=int, default=25, help="Requested RGB camera FPS for AI. Default: 25.")
    parser.add_argument("--ai-every-n", type=int, default=3, help="Run face detection every N frames. Default: 3.")
    parser.add_argument("--forehead-threshold-c", type=float, default=37.5, help="Forehead alert threshold in Celsius. Default: 37.5.")
    parser.add_argument("--forehead-percentile", type=float, default=90.0, help="Percentile used inside forehead ROI. Default: 90.")
    parser.add_argument("--forehead-min-c", type=float, default=20.0, help="Minimum plausible forehead ROI temp. Default: 20.")
    parser.add_argument("--forehead-max-c", type=float, default=45.0, help="Maximum plausible forehead ROI temp. Default: 45.")
    parser.add_argument("--no-thermal-forehead-fallback", dest="thermal_forehead_fallback", action="store_false", default=True, help="Disable thermal-only forehead estimate when MediaPipe/RGB is unavailable.")
    parser.add_argument("--thermal-face-min-c", type=float, default=30.0, help="Thermal-only face lower threshold. Default: 30.")
    parser.add_argument("--thermal-face-max-c", type=float, default=43.0, help="Thermal-only face upper threshold. Default: 43.")
    parser.add_argument("--thermal-face-min-area", type=int, default=80, help="Minimum warm connected-component area for thermal forehead fallback. Default: 80.")
    parser.add_argument("--thermal-forehead-top-ratio", type=float, default=0.18, help="Thermal fallback forehead top ratio inside detected head area. Default: 0.18.")
    parser.add_argument("--thermal-forehead-bottom-ratio", type=float, default=0.36, help="Thermal fallback forehead bottom ratio inside detected head area. Default: 0.36.")
    parser.add_argument("--thermal-forehead-width-ratio", type=float, default=0.34, help="Thermal fallback forehead width ratio inside detected person area. Default: 0.34.")
    parser.add_argument("--hot-threshold-c", type=float, default=60.0, help="Hot object threshold in Celsius. Default: 60.")
    parser.add_argument("--hot-min-area", type=int, default=8, help="Minimum thermal pixels for a hot object. Default: 8.")
    parser.add_argument("--hot-persist-s", type=float, default=2.0, help="Seconds a hot object must persist before alert. Default: 2.")
    parser.add_argument("--hot-max-alerts", type=int, default=3, help="Maximum hot objects shown per frame. Default: 3.")
    parser.add_argument("--ai-use-yolo", action="store_true", help="Use Ultralytics YOLO for hot object labels if installed.")
    parser.add_argument("--ai-object-model", default="yolo11n.pt", help="YOLO model path/name. Default: yolo11n.pt.")
    parser.add_argument("--yolo-conf", type=float, default=0.35, help="YOLO confidence threshold. Default: 0.35.")
    parser.add_argument("--yolo-every-n", type=int, default=10, help="Run YOLO every N frames. Default: 10.")
    parser.add_argument(
        "--risk-classes",
        default="socket,outlet,power strip,plug,charger,adapter,battery,laptop,cell phone,tv,oven,microwave,toaster,hair drier",
        help="Comma-separated YOLO labels treated as fire/electrical risk classes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        list_video_devices(args.scan_min, args.scan_max, args.width, args.height, args.fps, args.backend)
        return 0

    if args.find_raw:
        return probe_raw_streams(args)

    device = args.device
    ai_requested = bool(args.ai_forehead or args.hot_object_watch or args.calibrate_ai)

    if device == 0:
        print("ERROR: Camera index 0 is blocked by design. Use --device 1 for TC001 Plus/OpenCV visual mode.")
        return 2
    if ai_requested and int(args.rgb_device) == 0:
        print("ERROR: AI RGB camera index 0 is blocked by design. Use --rgb-device 1.")
        return 2

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
        if device == 0:
            print("ERROR: Camera index 0 is blocked by design. Use camera index 1.")
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
    rgb_cap: Optional[cv2.VideoCapture] = None
    sdk_cam: Optional[TC001SdkCamera] = None
    ai: Optional[ThermalAI] = None

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

    if ai_requested:
        try:
            ai = ThermalAI(args)
            for line in ai.status_lines():
                print(line)
        except Exception as exc:
            print(f"WARNING: AI setup failed: {exc}")
            ai = None
            if args.calibrate_ai:
                return 1

        needs_rgb = bool(args.calibrate_ai or (ai is not None and ai.needs_rgb()))
        if needs_rgb:
            if cap is not None and sdk_cam is None and int(args.rgb_device) == int(device):
                print(f"AI RGB source: reusing video device {device}.")
            else:
                print(f"Opening AI RGB source device {args.rgb_device} using backend={args.rgb_backend}...")
                rgb_cap, rgb_backend = open_camera(
                    int(args.rgb_device),
                    int(args.rgb_width),
                    int(args.rgb_height),
                    int(args.rgb_fps),
                    backend=args.rgb_backend,
                    options=OpenOptions(request_raw=False),
                )
                if not rgb_cap.isOpened():
                    print(f"WARNING: Could not open AI RGB device {args.rgb_device}. AI will use thermal-only features.")
                    rgb_cap.release()
                    rgb_cap = None
                    if args.calibrate_ai:
                        return 1
                else:
                    print(f"Opened AI RGB device {args.rgb_device} with backend={rgb_backend}.")
        elif ai_requested:
            print("AI RGB source not opened: using thermal-only forehead/hot-object logic.")

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

    if args.calibrate_ai:
        print("Starting 4-point AI calibration...")
        calibration_rgb: Optional[np.ndarray] = None
        calibration_thermal: Optional[ThermalFrame] = None

        if rgb_cap is not None:
            ret_rgb, calibration_rgb = rgb_cap.read()
            if not ret_rgb:
                calibration_rgb = None
        elif cap is not None:
            ret_rgb, calibration_rgb = cap.read()
            if not ret_rgb:
                calibration_rgb = None

        if sdk_cam is not None:
            temp_c = sdk_cam.read_temp_c(timeout_s=2.0)
            if temp_c is not None:
                temp_c = temp_c * args.temp_scale + args.temp_offset
                calibration_thermal = ThermalFrame(display_source=temp_c, temp_c=temp_c, mode="sdk-radiometric")
        elif cap is not None:
            ret_frame, frame_for_cal = cap.read()
            if ret_frame and frame_for_cal is not None:
                working_frame = split_frame_if_needed(frame_for_cal, state.split_mode)
                calibration_thermal = decode_frame(
                    working_frame,
                    args.width,
                    args.height,
                    args.fallback_min_c,
                    args.fallback_max_c,
                    args.estimate_temps,
                    args.temp_scale,
                    args.temp_offset,
                )

        if calibration_rgb is None or calibration_thermal is None:
            print("ERROR: Calibration needs both an RGB frame and a thermal frame.")
            return 1

        calibration_thermal = apply_orientation(calibration_thermal, state)
        if run_ai_calibration(calibration_rgb, calibration_thermal.display_source, args.alignment_file):
            if ai is not None:
                ai.alignment = AlignmentMap.load(args.alignment_file)
        else:
            print("AI calibration cancelled.")

    print_controls()

    fps_smooth = 0.0
    last_time = time.time()
    read_failures = 0

    try:
        while True:
            frame = None
            rgb_frame_for_ai: Optional[np.ndarray] = None
            if sdk_cam is not None:
                temp_c = sdk_cam.read_temp_c(timeout_s=1.0)
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

            if ai is not None and (args.ai_forehead or args.hot_object_watch):
                if rgb_cap is not None:
                    ret_rgb, rgb_candidate = rgb_cap.read()
                    if ret_rgb and rgb_candidate is not None:
                        rgb_frame_for_ai = rgb_candidate
                elif frame is not None:
                    rgb_frame_for_ai = frame

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
            thermal = apply_orientation(thermal, state)
            heatmap, contrast_range = make_heatmap(thermal.display_source, state)
            heatmap = resize_to_window(heatmap, state.fit_window)
            draw_hud(heatmap, thermal, state, fps_smooth, contrast_range, recorder)
            if ai is not None and (args.ai_forehead or args.hot_object_watch):
                ai_overlay = ai.analyze(rgb_frame_for_ai, thermal.temp_c)
                draw_ai_overlay(heatmap, ai_overlay, thermal.display_source.shape[:2], state.unit)

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
        if rgb_cap is not None:
            rgb_cap.release()
        if sdk_cam is not None:
            sdk_cam.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
