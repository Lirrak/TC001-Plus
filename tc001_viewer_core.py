from __future__ import annotations

import cv2
import json
import numpy as np
import os
import time
from typing import Optional, Sequence, Tuple, List

from tc001_face import FaceBox
from tc001_sdk import TC001_WIDTH, TC001_HEIGHT, TC001_FPS
from tc001_viewer_types import (
    ThermalFrame,
    ViewerState,
    OpenOptions,
    Alignment,
    RoiTemperatureStats,
    COLORMAPS,
)
from tc001_viewer_utils import (
    timestamp,
    backend_candidates,
    fourcc_to_int,
    describe_fourcc,
    robust_percentile_range,
    normalize_linear,
    is_plausible_temperature_matrix,
    uint16_le_to_temp_c,
    crop_or_resize_matrix,
)


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

    # Normalise dimensions
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


def shrink_polygon(points: np.ndarray, scale: float) -> np.ndarray:
    scale = min(max(float(scale), 0.1), 1.0)
    pts = points.astype(np.float32).reshape(-1, 2)
    if pts.size == 0 or scale >= 0.999:
        return pts
    center = np.mean(pts, axis=0, keepdims=True)
    return center + (pts - center) * scale


def polygon_temperature_values(
    temp_c: Optional[np.ndarray],
    polygon: np.ndarray,
    mask_shrink: float = 1.0,
) -> np.ndarray:
    if temp_c is None or temp_c.size == 0 or polygon.size == 0:
        return np.asarray([], dtype=np.float32)
    mask = np.zeros(temp_c.shape[:2], dtype=np.uint8)
    pts = np.round(shrink_polygon(polygon, mask_shrink)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    values = temp_c[mask > 0]
    values = values[np.isfinite(values)]
    if values.size == 0 and mask_shrink < 0.999:
        mask.fill(0)
        pts = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
        values = temp_c[mask > 0]
        values = values[np.isfinite(values)]
    return values.astype(np.float32, copy=False)


def polygon_temperature_stats(
    temp_c: Optional[np.ndarray],
    polygon: np.ndarray,
    mode: str = "robust",
    percentile: float = 90.0,
    hot_outlier_delta_c: float = 2.5,
    mask_shrink: float = 0.85,
) -> RoiTemperatureStats:
    values = polygon_temperature_values(temp_c, polygon, mask_shrink)
    mode = (mode or "robust").lower()
    if values.size == 0:
        return RoiTemperatureStats(None, None, None, None, None, 0, 0, 0, False, mode)

    percentile = min(max(float(percentile), 50.0), 99.0)
    hot_outlier_delta_c = max(0.1, float(hot_outlier_delta_c))
    max_temp = float(np.nanmax(values))
    median_temp = float(np.nanmedian(values))

    if mode == "max":
        return RoiTemperatureStats(max_temp, max_temp, median_temp, max_temp, 0.0, 0, int(values.size), int(values.size), False, mode)

    q25, q75 = np.nanpercentile(values, [25.0, 75.0])
    iqr = max(float(q75 - q25), 0.0)
    iqr_cutoff = float(q75) + max(1.0, 1.5 * iqr)
    delta_cutoff = median_temp + hot_outlier_delta_c
    cutoff = max(float(q75), min(delta_cutoff, iqr_cutoff))

    filtered = values[values <= cutoff]
    min_used = max(8, int(values.size * 0.25))
    if filtered.size < min_used:
        fallback_cutoff = median_temp + hot_outlier_delta_c
        fallback = values[values <= fallback_cutoff]
        filtered = fallback if fallback.size >= min_used else values
        cutoff = fallback_cutoff if fallback.size >= min_used else max_temp

    percentile_temp = float(np.nanpercentile(filtered, percentile))
    person_temp = percentile_temp
    hot_pixels = int(np.count_nonzero(values > cutoff))
    hot_delta = max_temp - person_temp
    contaminated = bool(hot_pixels > 0 and hot_delta >= hot_outlier_delta_c)
    return RoiTemperatureStats(
        person_temp_c=person_temp,
        max_temp_c=max_temp,
        median_temp_c=median_temp,
        percentile_temp_c=percentile_temp,
        hot_outlier_delta_c=float(hot_delta),
        hot_outlier_pixels=hot_pixels,
        roi_pixels=int(values.size),
        used_pixels=int(filtered.size),
        roi_temp_contaminated=contaminated,
        mode=mode,
    )


def polygon_max_temperature_c(temp_c: Optional[np.ndarray], polygon: np.ndarray) -> Optional[float]:
    return polygon_temperature_stats(temp_c, polygon, mode="max", mask_shrink=1.0).max_temp_c


def polygon_temperature_stats_for_state(
    temp_c: Optional[np.ndarray],
    polygon: np.ndarray,
    state: Optional[ViewerState],
) -> RoiTemperatureStats:
    if state is None:
        return polygon_temperature_stats(temp_c, polygon)
    return polygon_temperature_stats(
        temp_c,
        polygon,
        mode=state.roi_temp_mode,
        percentile=state.roi_temp_percentile,
        hot_outlier_delta_c=state.roi_hot_outlier_delta_c,
        mask_shrink=state.roi_mask_shrink,
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

    if state.zoom != 1 and not state.fit_window:
        interpolation = cv2.INTER_NEAREST if state.zoom >= 2 else cv2.INTER_AREA
        heatmap = cv2.resize(
            heatmap,
            (heatmap.shape[1] * state.zoom, heatmap.shape[0] * state.zoom),
            interpolation=interpolation,
        )

    return heatmap, (vmin, vmax)
