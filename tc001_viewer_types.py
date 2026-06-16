from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any

import cv2
import numpy as np

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
    roi_temp_mode: str = "robust"
    roi_temp_percentile: float = 90.0
    roi_hot_outlier_delta_c: float = 2.5
    roi_mask_shrink: float = 0.85
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


@dataclass
class RoiTemperatureStats:
    person_temp_c: Optional[float]
    max_temp_c: Optional[float]
    median_temp_c: Optional[float]
    percentile_temp_c: Optional[float]
    hot_outlier_delta_c: Optional[float]
    hot_outlier_pixels: int
    roi_pixels: int
    used_pixels: int
    roi_temp_contaminated: bool
    mode: str
