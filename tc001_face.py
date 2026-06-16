from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

SAME_TRACK_THRESHOLD = 0.18
REID_TRACK_THRESHOLD = 0.16


@dataclass
class FaceBox:
    x: float
    y: float
    w: float
    h: float
    confidence: float
    source: str


@dataclass
class FaceTrack:
    track_id: int = 0
    box: Optional[FaceBox] = None
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    detector_name: str = "none"
    last_detector_source: str = "none"
    status: str = "NO FACE"
    missed_frames: int = 0
    hits: int = 0
    confirmed: bool = False
    merged_from: Optional[int] = None
    last_iou_with_existing: float = 0.0


@dataclass
class FaceTracker:
    tracks: list[FaceTrack] = field(default_factory=list)
    retired_tracks: list[FaceTrack] = field(default_factory=list)
    next_track_id: int = 1
    last_stats: dict = field(default_factory=dict)

    def active_tracks(
        self,
        hold_frames: Optional[int] = None,
        frame_index: Optional[int] = None,
        confirmed_only: bool = False,
    ) -> list[FaceTrack]:
        tracks = [track for track in self.tracks if track.box is not None]
        if confirmed_only:
            tracks = [track for track in tracks if track.confirmed]
        if hold_frames is not None and frame_index is not None:
            tracks = [track for track in tracks if frame_index - track.last_seen_frame <= hold_frames]
        return sorted(tracks, key=lambda track: track.track_id)


class DigitalFaceDetector:
    def __init__(
        self,
        model: str = "auto",
        min_confidence: float = 0.55,
        task_model_path: Optional[str] = None,
    ) -> None:
        self.requested_model = model
        self.min_confidence = float(min_confidence)
        self.name = "none"
        self._mp_task_detector = None
        self._mp_image_cls = None
        self._mp_image_format = None
        self._task_timestamp_ms = 0
        self._mp_detector = None
        self._haar_frontal = None
        self._haar_profile = None
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.last_debug: dict = {}

        if model in ("auto", "tasks") and task_model_path:
            try:
                import mediapipe as mp  # type: ignore
                from mediapipe.tasks import python as mp_tasks_python  # type: ignore
                from mediapipe.tasks.python import vision as mp_tasks_vision  # type: ignore

                base_options = mp_tasks_python.BaseOptions(model_asset_path=task_model_path)
                options = mp_tasks_vision.FaceDetectorOptions(
                    base_options=base_options,
                    running_mode=mp_tasks_vision.RunningMode.VIDEO,
                    min_detection_confidence=self.min_confidence,
                )
                self._mp_task_detector = mp_tasks_vision.FaceDetector.create_from_options(options)
                self._mp_image_cls = mp.Image
                self._mp_image_format = mp.ImageFormat.SRGB
            except Exception as exc:
                if model == "tasks":
                    print(f"WARNING: MediaPipe Tasks face detector unavailable: {exc}")
        elif model == "tasks" and not task_model_path:
            print("WARNING: --face-model tasks requires --face-task-model pointing to a .tflite face detector model.")

        if self._mp_task_detector is None and model in ("auto", "mediapipe"):
            try:
                import mediapipe as mp  # type: ignore

                self._mp_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=self.min_confidence,
                )
            except Exception as exc:
                if model == "mediapipe":
                    print(f"WARNING: MediaPipe face detector unavailable: {exc}")

        if model in ("auto", "tasks", "mediapipe", "haar"):
            frontal_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            profile_path = os.path.join(cv2.data.haarcascades, "haarcascade_profileface.xml")
            self._haar_frontal = self._load_cascade(frontal_path, "frontal Haar")
            self._haar_profile = self._load_cascade(profile_path, "profile Haar")

        available = []
        if self._mp_task_detector is not None:
            available.append("tasks")
        if self._mp_detector is not None:
            available.append("mediapipe")
        if self._haar_frontal is not None:
            available.append("frontal")
        if self._haar_profile is not None:
            available.append("profile")
        available.append("head")
        if available:
            self.name = "+".join(available)

        if self.name == "none":
            print("WARNING: No face detector is available. Install mediapipe or use a full opencv-python build.")

    def _load_cascade(self, path: str, label: str) -> Optional[cv2.CascadeClassifier]:
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            print(f"WARNING: Could not load {label} cascade: {path}")
            return None
        return cascade

    def detect(self, frame_bgr: np.ndarray) -> Optional[FaceBox]:
        boxes = self.detect_all(frame_bgr, max_faces=1)
        return boxes[0] if boxes else None

    def detect_all(
        self,
        frame_bgr: np.ndarray,
        max_faces: int = 5,
        head_fallback: str = "auto",
        min_head_confidence: float = 0.65,
        cascade_fallback: str = "auto",
    ) -> list[FaceBox]:
        if frame_bgr is None or frame_bgr.size == 0:
            self.last_debug = {"empty_frame": True}
            return []

        prepared_bgr, gray = self._preprocess(frame_bgr)
        frame_h, frame_w = gray.shape[:2]
        max_faces = max(1, int(max_faces))
        primary_candidates: list[FaceBox] = []

        model_candidates: list[FaceBox] = []
        if self._mp_task_detector is not None:
            model_candidates.extend(self._detect_mediapipe_tasks_all(frame_bgr if frame_bgr.ndim == 3 else prepared_bgr))
            if not model_candidates:
                model_candidates.extend(self._detect_mediapipe_tasks_all(prepared_bgr))

        if self._mp_detector is not None:
            model_candidates.extend(self._detect_mediapipe_all(frame_bgr if frame_bgr.ndim == 3 else prepared_bgr))
            model_candidates.extend(self._detect_mediapipe_all(prepared_bgr))

        primary_candidates.extend(model_candidates)
        cascade_fallback = (cascade_fallback or "auto").lower()
        use_cascade = cascade_fallback == "always" or (cascade_fallback == "auto" and len(model_candidates) < max_faces)

        if use_cascade and self._haar_frontal is not None:
            primary_candidates.extend(self._detect_haar_all(gray, self._haar_frontal, "haar", 0.80, min_neighbors=4))

        if use_cascade and self._haar_profile is not None:
            primary_candidates.extend(self._detect_profile_all(gray))

        primary, primary_nms = self._nms_with_debug(primary_candidates, iou_threshold=0.35, max_boxes=max_faces)
        head_fallback = (head_fallback or "auto").lower()
        candidates = list(primary)
        head_candidates: list[FaceBox] = []

        head_target = min(max_faces, 2)
        use_head = head_fallback == "always" or (head_fallback == "auto" and len(primary) < head_target)
        if use_head and head_fallback != "off":
            head_candidates = self._detect_head_all(gray, float(min_head_confidence))
            heads = [head for head in head_candidates if not self._is_near_primary(head, primary)]
            candidates.extend(heads)

        final, final_nms = self._nms_with_debug(candidates, iou_threshold=0.35, max_boxes=max_faces)
        self.last_debug = {
            "empty_frame": False,
            "frame_size": [int(frame_w), int(frame_h)],
            "max_faces": int(max_faces),
            "cascade_fallback": cascade_fallback,
            "cascade_used": bool(use_cascade),
            "head_fallback": head_fallback,
            "head_used": bool(use_head and head_fallback != "off"),
            "model_candidates": int(len(model_candidates)),
            "primary_candidates": int(len(primary_candidates)),
            "primary_candidates_by_source": count_boxes_by_source(primary_candidates),
            "primary_after_nms": int(len(primary)),
            "primary_nms_suppressed": primary_nms,
            "head_candidates": int(len(head_candidates)),
            "final_candidates": int(len(candidates)),
            "final_candidates_by_source": count_boxes_by_source(candidates),
            "final_after_nms": int(len(final)),
            "final_nms_suppressed": final_nms,
            "final_boxes": [box_to_debug_dict(box, (frame_h, frame_w)) for box in final],
        }
        return final

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if frame_bgr.ndim == 2:
            gray = frame_bgr
        elif frame_bgr.shape[2] == 1:
            gray = frame_bgr[:, :, 0]
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        gray = np.ascontiguousarray(gray)
        lo, hi = np.percentile(gray, [2.0, 98.0])
        if hi > lo:
            gray = np.clip((gray.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
        else:
            gray = gray.astype(np.uint8, copy=False)
        gray = self._clahe.apply(gray)
        gray = cv2.medianBlur(gray, 3)
        prepared_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return prepared_bgr, gray

    def _detect_mediapipe_tasks_all(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        if self._mp_task_detector is None or self._mp_image_cls is None or self._mp_image_format is None:
            return []
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp_image_cls(image_format=self._mp_image_format, data=np.ascontiguousarray(rgb))
        self._task_timestamp_ms += 40
        result = self._mp_task_detector.detect_for_video(image, self._task_timestamp_ms)
        detections = result.detections if result and result.detections else []
        boxes: list[FaceBox] = []
        for det in detections:
            score = 0.0
            categories = getattr(det, "categories", None)
            if categories:
                score = float(categories[0].score)
            if score < self.min_confidence:
                continue
            bbox = det.bounding_box
            box = FaceBox(
                x=float(bbox.origin_x),
                y=float(bbox.origin_y),
                w=max(1.0, float(bbox.width)),
                h=max(1.0, float(bbox.height)),
                confidence=score,
                source="tasks",
            )
            boxes.append(self._clip_box(box, w, h))
        return boxes

    def _detect_mediapipe_all(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._mp_detector.process(rgb)
        detections = result.detections if result and result.detections else []
        boxes: list[FaceBox] = []
        for det in detections:
            score = float(det.score[0]) if det.score else 0.0
            if score < self.min_confidence:
                continue
            rel = det.location_data.relative_bounding_box
            box = FaceBox(
                x=max(0.0, rel.xmin * w),
                y=max(0.0, rel.ymin * h),
                w=max(1.0, rel.width * w),
                h=max(1.0, rel.height * h),
                confidence=score,
                source="mediapipe",
            )
            boxes.append(self._clip_box(box, w, h))
        return boxes

    def _detect_haar_all(
        self,
        gray: np.ndarray,
        cascade: cv2.CascadeClassifier,
        source: str,
        confidence: float,
        min_neighbors: int = 5,
    ) -> list[FaceBox]:
        frame_h, frame_w = gray.shape[:2]
        min_side = max(28, int(min(frame_w, frame_h) * 0.14))
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=min_neighbors,
            minSize=(min_side, min_side),
        )
        return [
            self._clip_box(FaceBox(float(x), float(y), float(w), float(h), confidence, source), frame_w, frame_h)
            for x, y, w, h in faces
        ]

    def _detect_profile_all(self, gray: np.ndarray) -> list[FaceBox]:
        assert self._haar_profile is not None
        frame_h, frame_w = gray.shape[:2]
        boxes = self._detect_haar_all(gray, self._haar_profile, "profile", 0.72, min_neighbors=3)

        flipped = cv2.flip(gray, 1)
        for box in self._detect_haar_all(flipped, self._haar_profile, "profile", 0.72, min_neighbors=3):
            mirrored = FaceBox(
                x=float(frame_w - box.x - box.w),
                y=box.y,
                w=box.w,
                h=box.h,
                confidence=box.confidence,
                source="profile",
            )
            boxes.append(self._clip_box(mirrored, frame_w, frame_h))
        return boxes

    def _detect_head_all(self, gray: np.ndarray, confidence: float) -> list[FaceBox]:
        frame_h, frame_w = gray.shape[:2]
        if frame_h < 40 or frame_w < 40:
            return []

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh_value = max(64.0, float(np.percentile(blurred, 70.0)))
        _, mask = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = frame_w * frame_h * 0.035
        max_area = frame_w * frame_h * 0.28
        boxes: list[FaceBox] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if y > frame_h * 0.72:
                continue
            if area > max_area or w > frame_w * 0.55 or h > frame_h * 0.58:
                boxes.extend(self._detect_head_from_large_contour(mask, contour, confidence))
                continue
            if w < frame_w * 0.14 or h < frame_h * 0.18:
                continue
            if w > frame_w * 0.48 or h > frame_h * 0.55:
                continue
            aspect = w / max(h, 1)
            if not 0.45 <= aspect <= 1.35:
                continue
            fill_ratio = area / max(float(w * h), 1.0)
            if fill_ratio < 0.32:
                continue
            head_h = min(h, max(int(w * 1.15), int(frame_h * 0.18)))
            boxes.append(self._clip_box(FaceBox(float(x), float(y), float(w), float(head_h), confidence, "head"), frame_w, frame_h))
        return boxes

    def _detect_head_from_large_contour(
        self,
        mask: np.ndarray,
        contour: np.ndarray,
        confidence: float,
    ) -> list[FaceBox]:
        frame_h, frame_w = mask.shape[:2]
        x, y, w, h = cv2.boundingRect(contour)
        if h < frame_h * 0.24 or w < frame_w * 0.16:
            return []

        component = np.zeros_like(mask)
        cv2.drawContours(component, [contour], -1, 255, thickness=cv2.FILLED)
        upper_h = min(h, max(int(h * 0.52), int(frame_h * 0.26)))
        upper = component[y : y + upper_h, x : x + w]
        if upper.size == 0:
            return []

        upper_contours, _hierarchy = cv2.findContours(upper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[FaceBox] = []
        for upper_contour in upper_contours:
            area = float(cv2.contourArea(upper_contour))
            if area < frame_w * frame_h * 0.018:
                continue
            ux, uy, uw, uh = cv2.boundingRect(upper_contour)
            if uw < frame_w * 0.11 or uh < frame_h * 0.12:
                continue
            if y + uy > frame_h * 0.68:
                continue

            box_w = min(max(float(uw) * 1.18, frame_w * 0.16), frame_w * 0.44)
            box_h = min(max(float(uh) * 1.10, box_w * 1.05, frame_h * 0.20), frame_h * 0.48)
            cx = x + ux + uw * 0.5
            by = y + uy - box_h * 0.08
            bx = cx - box_w * 0.5
            box = self._clip_box(FaceBox(float(bx), float(by), float(box_w), float(box_h), confidence, "head"), frame_w, frame_h)
            ratio = area_ratio(box, (frame_h, frame_w))
            aspect = box.w / max(box.h, 1.0)
            contour_fill = area / max(float(uw * uh), 1.0)
            box_fill = area / max(float(box.w * box.h), 1.0)
            if 0.035 <= ratio <= 0.20 and 0.45 <= aspect <= 1.45 and contour_fill >= 0.35 and box_fill >= 0.20:
                candidates.append(box)

        return self._nms(candidates, iou_threshold=0.30, max_boxes=2)

    def _is_near_primary(self, head: FaceBox, primary_boxes: list[FaceBox]) -> bool:
        for primary in primary_boxes:
            if box_iou(head, primary) >= 0.15:
                return True
            distance = center_distance(head, primary)
            scale = max(head.w, head.h, primary.w, primary.h, 1.0)
            if distance <= scale * 0.65:
                return True
        return False

    def _nms(self, boxes: list[FaceBox], iou_threshold: float, max_boxes: int) -> list[FaceBox]:
        selected, _suppressed = self._nms_with_debug(boxes, iou_threshold, max_boxes)
        return selected

    def _nms_with_debug(
        self,
        boxes: list[FaceBox],
        iou_threshold: float,
        max_boxes: int,
    ) -> tuple[list[FaceBox], list[dict]]:
        if not boxes:
            return [], []

        def score(box: FaceBox) -> tuple[float, float, float]:
            priority = {"tasks": 3.5, "mediapipe": 3.0, "haar": 2.0, "profile": 1.5, "head": 1.0}.get(box.source, 0.0)
            area = box.w * box.h
            return priority, box.confidence, area

        selected: list[FaceBox] = []
        suppressed: list[dict] = []
        ordered = sorted(boxes, key=score, reverse=True)
        for index, box in enumerate(ordered):
            if len(selected) >= max_boxes:
                suppressed.append({"reason": "max_faces", **box_to_debug_dict(box)})
                continue
            best_iou = 0.0
            best_source = None
            for chosen in selected:
                overlap = box_iou(box, chosen)
                if overlap > best_iou:
                    best_iou = overlap
                    best_source = chosen.source
            if best_iou < iou_threshold:
                selected.append(box)
            else:
                suppressed.append(
                    {
                        "reason": "nms",
                        "iou": round(float(best_iou), 4),
                        "matched_source": best_source,
                        **box_to_debug_dict(box),
                    }
                )
        return sorted(selected, key=lambda box: box.x), suppressed

    def _clip_box(self, box: FaceBox, frame_w: int, frame_h: int) -> FaceBox:
        x0 = min(max(float(box.x), 0.0), max(float(frame_w - 1), 0.0))
        y0 = min(max(float(box.y), 0.0), max(float(frame_h - 1), 0.0))
        x1 = min(max(float(box.x + box.w), x0 + 1.0), float(frame_w))
        y1 = min(max(float(box.y + box.h), y0 + 1.0), float(frame_h))
        return FaceBox(x0, y0, x1 - x0, y1 - y0, box.confidence, box.source)


def face_status_from_source(source: str, held: bool = False) -> str:
    if held:
        return "HELD"
    if source == "profile":
        return "PROFILE"
    if source == "head":
        return "HEAD TRACK"
    if source in ("tasks", "mediapipe", "haar"):
        return "FACE"
    return source.upper() if source else "NO FACE"


def required_hits_for_source(source: str, head_confirm_frames: int) -> int:
    if source == "head":
        return max(1, int(head_confirm_frames))
    if source in ("haar", "profile"):
        return 2
    return 1


def required_hits(source: str, min_face_hits: int, head_confirm_frames: int) -> int:
    if source == "head":
        return max(1, int(head_confirm_frames))
    if source in ("haar", "profile"):
        return max(1, int(min_face_hits))
    return 1


def area_ratio(box: FaceBox, frame_shape: tuple[int, int]) -> float:
    frame_h, frame_w = frame_shape[:2]
    return float((box.w * box.h) / max(float(frame_w * frame_h), 1.0))


def count_boxes_by_source(boxes: list[FaceBox]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for box in boxes:
        counts[box.source] = counts.get(box.source, 0) + 1
    return counts


def box_to_debug_dict(box: FaceBox, frame_shape: Optional[tuple[int, int]] = None) -> dict:
    data = {
        "source": box.source,
        "confidence": round(float(box.confidence), 4),
        "bbox": [round(float(box.x), 2), round(float(box.y), 2), round(float(box.w), 2), round(float(box.h), 2)],
    }
    if frame_shape is not None:
        data["area_ratio"] = round(area_ratio(box, frame_shape), 4)
    return data


def filter_detection_boxes(
    boxes: list[FaceBox],
    frame_shape: tuple[int, int],
    max_face_area_ratio: float,
) -> tuple[list[FaceBox], list[dict]]:
    frame_h, frame_w = frame_shape[:2]
    max_face_area_ratio = max(0.01, float(max_face_area_ratio))
    kept: list[FaceBox] = []
    rejected: list[dict] = []
    for box in boxes:
        reasons = []
        ratio = area_ratio(box, frame_shape)
        if ratio > max_face_area_ratio:
            reasons.append("area")
        if box.w > frame_w * 0.55 or box.h > frame_h * 0.55:
            reasons.append("size")
        if box.w < frame_w * 0.08 or box.h < frame_h * 0.08:
            reasons.append("small")
        aspect = box.w / max(box.h, 1.0)
        if not 0.45 <= aspect <= 1.55:
            reasons.append("aspect")
        touches_edge = box.x <= 1 or box.y <= 1 or box.x + box.w >= frame_w - 1 or box.y + box.h >= frame_h - 1
        if touches_edge and box.confidence < 0.85:
            reasons.append("edge")
        if reasons:
            rejected.append({"reasons": reasons, **box_to_debug_dict(box, frame_shape)})
        else:
            kept.append(box)
    return kept, rejected


def box_iou(a: FaceBox, b: FaceBox) -> float:
    ax0, ay0, ax1, ay1 = a.x, a.y, a.x + a.w, a.y + a.h
    bx0, by0, bx1, by1 = b.x, b.y, b.x + b.w, b.y + b.h
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    intersection = iw * ih
    union = a.w * a.h + b.w * b.h - intersection
    return 0.0 if union <= 0 else intersection / union


def center_distance(a: FaceBox, b: FaceBox) -> float:
    acx, acy = a.x + a.w * 0.5, a.y + a.h * 0.5
    bcx, bcy = b.x + b.w * 0.5, b.y + b.h * 0.5
    return float(np.hypot(acx - bcx, acy - bcy))


def smooth_box(old_box: FaceBox, new_box: FaceBox, smoothing: float) -> FaceBox:
    a = min(max(float(smoothing), 0.0), 0.95)
    return FaceBox(
        x=old_box.x * a + new_box.x * (1.0 - a),
        y=old_box.y * a + new_box.y * (1.0 - a),
        w=old_box.w * a + new_box.w * (1.0 - a),
        h=old_box.h * a + new_box.h * (1.0 - a),
        confidence=new_box.confidence,
        source=new_box.source,
    )


def match_score(track: FaceTrack, box: FaceBox) -> float:
    if track.box is None:
        return -1.0
    iou = box_iou(track.box, box)
    distance = center_distance(track.box, box)
    scale = max(track.box.w, track.box.h, box.w, box.h, 1.0)
    distance_score = max(0.0, 1.0 - distance / (scale * 1.4))
    return max(iou, distance_score * 0.75)


def update_face_tracks(
    detector: DigitalFaceDetector,
    tracker: FaceTracker,
    digital_frame: np.ndarray,
    frame_index: int,
    detect_interval: int,
    hold_frames: int,
    smoothing: float,
    max_faces: int,
    head_fallback: str = "auto",
    head_confirm_frames: int = 2,
    min_head_confidence: float = 0.65,
    min_face_hits: int = 2,
    max_face_area_ratio: float = 0.20,
    max_box_overlap: float = 0.30,
    cascade_fallback: str = "auto",
    face_reid_frames: int = 45,
) -> FaceTracker:
    max_faces = max(1, int(max_faces))
    face_reid_frames = max(int(hold_frames), int(face_reid_frames))
    should_detect = not tracker.tracks or frame_index % max(1, detect_interval) == 0
    active_before = len([track for track in tracker.tracks if track.box is not None])
    retired_before = len([track for track in tracker.retired_tracks if track.box is not None])
    detections = (
        detector.detect_all(
            digital_frame,
            max_faces=max_faces,
            head_fallback=head_fallback,
            min_head_confidence=min_head_confidence,
            cascade_fallback=cascade_fallback,
        )
        if should_detect
        else []
    )
    detector_debug = dict(getattr(detector, "last_debug", {}) or {}) if should_detect else {}
    rejected: list[dict] = []
    if should_detect:
        detections, rejected = filter_detection_boxes(detections, digital_frame.shape[:2], max_face_area_ratio)
    stats = {
        "frame_index": int(frame_index),
        "should_detect": bool(should_detect),
        "active_tracks_before": int(active_before),
        "detector": getattr(detector, "name", "none"),
        "detector_requested_model": getattr(detector, "requested_model", "unknown"),
        "detect_interval": int(max(1, detect_interval)),
        "hold_frames": int(hold_frames),
        "face_reid_frames": int(face_reid_frames),
        "max_faces": int(max_faces),
        "head_fallback": head_fallback,
        "cascade_fallback": cascade_fallback,
        "same_track_threshold": float(SAME_TRACK_THRESHOLD),
        "reid_track_threshold": float(REID_TRACK_THRESHOLD),
        "duplicate_iou_threshold": float(max_box_overlap),
        "max_face_area_ratio": float(max_face_area_ratio),
        "detector_debug": detector_debug,
        "raw_detections": int(len(detections) + len(rejected)),
        "kept_detections": int(len(detections)),
        "kept_by_source": count_boxes_by_source(detections),
        "rejected_detections": int(len(rejected)),
        "rejected": rejected,
        "merged_detections": 0,
        "duplicate_suppressed": 0,
        "new_tracks": 0,
        "matches": [],
        "held_tracks": 0,
        "expired_tracks": 0,
        "active_retired_tracks_before": int(retired_before),
        "retired_tracks_after": 0,
        "reidentified_tracks": 0,
        "reidentified": [],
    }
    unmatched_detection_indexes = set(range(len(detections)))
    matched_track_ids: set[int] = set()

    if should_detect:
        match_candidates: list[tuple[float, int, int, FaceTrack]] = []
        for track in tracker.tracks:
            if track.box is None:
                continue
            for index in unmatched_detection_indexes:
                score = match_score(track, detections[index])
                if score >= SAME_TRACK_THRESHOLD:
                    match_candidates.append((float(score), int(track.track_id), int(index), track))

        for best_score, _track_id, best_index, track in sorted(match_candidates, key=lambda item: item[0], reverse=True):
            if track.track_id in matched_track_ids or best_index not in unmatched_detection_indexes:
                continue
            new_box = detections[best_index]
            track.box = smooth_box(track.box, new_box, smoothing)
            track.last_seen_frame = frame_index
            track.detector_name = new_box.source
            track.last_detector_source = new_box.source
            track.status = face_status_from_source(new_box.source)
            track.hits += 1
            track.confirmed = track.confirmed or track.hits >= required_hits(new_box.source, min_face_hits, head_confirm_frames)
            track.missed_frames = 0
            track.last_iou_with_existing = float(best_score)
            matched_track_ids.add(track.track_id)
            unmatched_detection_indexes.remove(best_index)
            stats["matches"].append(
                {
                    "track_id": int(track.track_id),
                    "detection_index": int(best_index),
                    "score": round(float(best_score), 4),
                    "source": new_box.source,
                }
            )

    for track in tracker.tracks:
        if track.box is None:
            continue
        if not should_detect or track.track_id in matched_track_ids:
            continue
        if track.confirmed and frame_index - track.last_seen_frame <= hold_frames:
            track.missed_frames += 1
            track.detector_name = "held"
            track.status = "HELD"
            stats["held_tracks"] += 1
        else:
            track.missed_frames += 1
            stats["expired_tracks"] += 1

    kept_tracks: list[FaceTrack] = []
    newly_retired: list[FaceTrack] = []
    for track in tracker.tracks:
        if track.box is None:
            continue
        age_since_seen = frame_index - track.last_seen_frame
        if (track.confirmed and age_since_seen <= hold_frames) or (not track.confirmed and track.missed_frames <= 0):
            kept_tracks.append(track)
        elif track.confirmed and age_since_seen <= face_reid_frames:
            track.detector_name = "lost"
            track.status = "LOST"
            newly_retired.append(track)
    tracker.tracks = kept_tracks

    retained_retired: dict[int, FaceTrack] = {}
    for track in list(tracker.retired_tracks) + newly_retired:
        if track.box is None or not track.confirmed:
            continue
        if frame_index - track.last_seen_frame <= face_reid_frames:
            retained_retired[int(track.track_id)] = track
    tracker.retired_tracks = sorted(retained_retired.values(), key=lambda track: track.track_id)

    if should_detect:
        current_count = len([track for track in tracker.tracks if track.confirmed])
        if tracker.retired_tracks and unmatched_detection_indexes:
            reid_candidates: list[tuple[float, int, int, FaceTrack]] = []
            for track in tracker.retired_tracks:
                if track.box is None:
                    continue
                for index in unmatched_detection_indexes:
                    score = match_score(track, detections[index])
                    if score >= REID_TRACK_THRESHOLD:
                        reid_candidates.append((float(score), int(track.track_id), int(index), track))

            restored_track_ids: set[int] = set()
            for best_score, _track_id, best_index, track in sorted(reid_candidates, key=lambda item: item[0], reverse=True):
                if best_index not in unmatched_detection_indexes or track.track_id in restored_track_ids:
                    continue
                if current_count >= max_faces:
                    break
                box = detections[best_index]
                track.box = smooth_box(track.box, box, smoothing)
                track.last_seen_frame = frame_index
                track.detector_name = box.source
                track.last_detector_source = box.source
                track.status = face_status_from_source(box.source)
                track.hits += 1
                track.confirmed = True
                track.missed_frames = 0
                track.last_iou_with_existing = float(best_score)
                track.merged_from = None
                tracker.tracks.append(track)
                restored_track_ids.add(track.track_id)
                unmatched_detection_indexes.remove(best_index)
                current_count += 1
                stats["reidentified_tracks"] += 1
                stats["reidentified"].append(
                    {
                        "track_id": int(track.track_id),
                        "detection_index": int(best_index),
                        "score": round(float(best_score), 4),
                        "source": box.source,
                    }
                )
            if restored_track_ids:
                tracker.retired_tracks = [track for track in tracker.retired_tracks if track.track_id not in restored_track_ids]

        for index in sorted(unmatched_detection_indexes, key=lambda idx: detections[idx].confidence, reverse=True):
            box = detections[index]
            overlap_track = None
            overlap_iou = 0.0
            for track in tracker.tracks:
                if track.box is None:
                    continue
                score = box_iou(track.box, box)
                if score > overlap_iou:
                    overlap_iou = score
                    overlap_track = track

            if overlap_track is not None and overlap_iou >= float(max_box_overlap):
                if overlap_track.last_seen_frame == frame_index:
                    stats["duplicate_suppressed"] += 1
                elif overlap_track.confirmed:
                    overlap_track.box = smooth_box(overlap_track.box, box, smoothing)
                    overlap_track.last_seen_frame = frame_index
                    overlap_track.detector_name = box.source
                    overlap_track.last_detector_source = box.source
                    overlap_track.status = face_status_from_source(box.source)
                    overlap_track.hits += 1
                    overlap_track.missed_frames = 0
                    overlap_track.last_iou_with_existing = float(overlap_iou)
                    overlap_track.merged_from = None
                    stats["merged_detections"] += 1
                else:
                    stats["duplicate_suppressed"] += 1
                continue

            if current_count >= max_faces:
                stats["duplicate_suppressed"] += 1
                continue

            hits = 1
            confirmed = hits >= required_hits(box.source, min_face_hits, head_confirm_frames)
            tracker.tracks.append(
                FaceTrack(
                    track_id=tracker.next_track_id,
                    box=box,
                    first_seen_frame=frame_index,
                    last_seen_frame=frame_index,
                    detector_name=box.source,
                    last_detector_source=box.source,
                    status=face_status_from_source(box.source),
                    missed_frames=0,
                    hits=hits,
                    confirmed=confirmed,
                    last_iou_with_existing=0.0,
                )
            )
            tracker.next_track_id += 1
            stats["new_tracks"] += 1
            if confirmed:
                current_count += 1

    confirmed_tracks = tracker.active_tracks(confirmed_only=True)
    candidate_tracks = [track for track in tracker.active_tracks() if not track.confirmed]
    tracker.tracks = confirmed_tracks[:max_faces] + candidate_tracks[:max_faces]
    stats["active_tracks_after"] = int(len([track for track in tracker.tracks if track.box is not None]))
    stats["confirmed_tracks_after"] = int(len([track for track in tracker.tracks if track.confirmed]))
    stats["candidate_tracks_after"] = int(len([track for track in tracker.tracks if not track.confirmed]))
    stats["retired_tracks_after"] = int(len([track for track in tracker.retired_tracks if track.box is not None]))
    tracker.last_stats = stats
    return tracker


def update_face_track(
    detector: DigitalFaceDetector,
    track: FaceTrack,
    digital_frame: np.ndarray,
    frame_index: int,
    detect_interval: int,
    hold_frames: int,
    smoothing: float,
) -> FaceTrack:
    tracker = FaceTracker(tracks=[track] if track.box is not None else [], next_track_id=max(track.track_id + 1, 2))
    update_face_tracks(detector, tracker, digital_frame, frame_index, detect_interval, hold_frames, smoothing, max_faces=1)
    active = tracker.active_tracks()
    return active[0] if active else FaceTrack()
