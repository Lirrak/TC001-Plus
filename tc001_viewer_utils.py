from __future__ import annotations

import cv2
import numpy as np
import platform
import time
from typing import Optional, Sequence, Tuple

from tc001_viewer_types import WINDOW_NAME


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
