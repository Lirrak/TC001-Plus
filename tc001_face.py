from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


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
    last_seen_frame: int = 0
    detector_name: str = "none"
    status: str = "NO FACE"
    missed_frames: int = 0
    hits: int = 0
    confirmed: bool = False


@dataclass
class FaceTracker:
    tracks: list[FaceTrack] = field(default_factory=list)
    next_track_id: int = 1

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
    def __init__(self, model: str = "auto", min_confidence: float = 0.55) -> None:
        self.requested_model = model
        self.min_confidence = float(min_confidence)
        self.name = "none"
        self._mp_detector = None
        self._haar_frontal = None
        self._haar_profile = None
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        if model in ("auto", "mediapipe"):
            try:
                import mediapipe as mp  # type: ignore

                self._mp_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=self.min_confidence,
                )
            except Exception as exc:
                if model == "mediapipe":
                    print(f"WARNING: MediaPipe face detector unavailable: {exc}")

        if model in ("auto", "mediapipe", "haar"):
            frontal_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            profile_path = os.path.join(cv2.data.haarcascades, "haarcascade_profileface.xml")
            self._haar_frontal = self._load_cascade(frontal_path, "frontal Haar")
            self._haar_profile = self._load_cascade(profile_path, "profile Haar")

        available = []
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
    ) -> list[FaceBox]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        prepared_bgr, gray = self._preprocess(frame_bgr)
        primary_candidates: list[FaceBox] = []

        if self._mp_detector is not None:
            primary_candidates.extend(self._detect_mediapipe_all(frame_bgr if frame_bgr.ndim == 3 else prepared_bgr))
            primary_candidates.extend(self._detect_mediapipe_all(prepared_bgr))

        if self._haar_frontal is not None:
            primary_candidates.extend(self._detect_haar_all(gray, self._haar_frontal, "haar", 0.80, min_neighbors=4))

        if self._haar_profile is not None:
            primary_candidates.extend(self._detect_profile_all(gray))

        max_faces = max(1, int(max_faces))
        primary = self._nms(primary_candidates, iou_threshold=0.35, max_boxes=max_faces)
        head_fallback = (head_fallback or "auto").lower()
        candidates = list(primary)

        head_target = min(max_faces, 2)
        use_head = head_fallback == "always" or (head_fallback == "auto" and len(primary) < head_target)
        if use_head and head_fallback != "off":
            heads = self._detect_head_all(gray, float(min_head_confidence))
            heads = [head for head in heads if not self._is_near_primary(head, primary)]
            candidates.extend(heads)

        return self._nms(candidates, iou_threshold=0.35, max_boxes=max_faces)

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
        thresh_value = max(70.0, float(np.percentile(blurred, 72.0)))
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
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
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
        if not boxes:
            return []

        def score(box: FaceBox) -> tuple[float, float, float]:
            priority = {"mediapipe": 3.0, "haar": 2.0, "profile": 1.5, "head": 1.0}.get(box.source, 0.0)
            area = box.w * box.h
            return priority, box.confidence, area

        selected: list[FaceBox] = []
        for box in sorted(boxes, key=score, reverse=True):
            if all(box_iou(box, chosen) < iou_threshold for chosen in selected):
                selected.append(box)
            if len(selected) >= max_boxes:
                break
        return sorted(selected, key=lambda box: box.x)

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
    if source in ("mediapipe", "haar"):
        return "FACE"
    return source.upper() if source else "NO FACE"


def required_hits_for_source(source: str, head_confirm_frames: int) -> int:
    if source == "head":
        return max(1, int(head_confirm_frames))
    return 1


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
) -> FaceTracker:
    max_faces = max(1, int(max_faces))
    should_detect = not tracker.tracks or frame_index % max(1, detect_interval) == 0
    detections = (
        detector.detect_all(
            digital_frame,
            max_faces=max_faces,
            head_fallback=head_fallback,
            min_head_confidence=min_head_confidence,
        )
        if should_detect
        else []
    )
    unmatched_detection_indexes = set(range(len(detections)))

    for track in tracker.tracks:
        if track.box is None:
            continue
        if not should_detect:
            continue

        best_index = None
        best_score = 0.0
        for index in list(unmatched_detection_indexes):
            score = match_score(track, detections[index])
            if score > best_score:
                best_score = score
                best_index = index

        if best_index is not None and best_score >= 0.18:
            new_box = detections[best_index]
            track.box = smooth_box(track.box, new_box, smoothing)
            track.last_seen_frame = frame_index
            track.detector_name = new_box.source
            track.status = face_status_from_source(new_box.source)
            track.hits += 1
            track.confirmed = track.confirmed or track.hits >= required_hits_for_source(new_box.source, head_confirm_frames)
            track.missed_frames = 0
            unmatched_detection_indexes.remove(best_index)
        elif track.confirmed and frame_index - track.last_seen_frame <= hold_frames:
            track.missed_frames += 1
            track.detector_name = "held"
            track.status = "HELD"
        else:
            track.missed_frames += 1

    tracker.tracks = [
        track
        for track in tracker.tracks
        if track.box is not None
        and (
            (track.confirmed and frame_index - track.last_seen_frame <= hold_frames)
            or (not track.confirmed and track.missed_frames <= 0)
        )
    ]

    if should_detect:
        current_count = len([track for track in tracker.tracks if track.confirmed])
        for index in sorted(unmatched_detection_indexes, key=lambda idx: detections[idx].confidence, reverse=True):
            if current_count >= max_faces:
                break
            box = detections[index]
            hits = 1
            confirmed = hits >= required_hits_for_source(box.source, head_confirm_frames)
            tracker.tracks.append(
                FaceTrack(
                    track_id=tracker.next_track_id,
                    box=box,
                    last_seen_frame=frame_index,
                    detector_name=box.source,
                    status=face_status_from_source(box.source),
                    missed_frames=0,
                    hits=hits,
                    confirmed=confirmed,
                )
            )
            tracker.next_track_id += 1
            if confirmed:
                current_count += 1

    confirmed_tracks = tracker.active_tracks(confirmed_only=True)
    candidate_tracks = [track for track in tracker.active_tracks() if not track.confirmed]
    tracker.tracks = confirmed_tracks[:max_faces] + candidate_tracks[:max_faces]
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
