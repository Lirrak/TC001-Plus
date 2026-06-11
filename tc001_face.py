from __future__ import annotations

import os
from dataclasses import dataclass
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
    box: Optional[FaceBox] = None
    last_seen_frame: int = 0
    detector_name: str = "none"
    status: str = "NO FACE"
    missed_frames: int = 0


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
                self.name = "multi-mediapipe"
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
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        prepared_bgr, gray = self._preprocess(frame_bgr)
        candidates = []
        if self._mp_detector is not None:
            box = self._detect_mediapipe(frame_bgr if frame_bgr.ndim == 3 else prepared_bgr)
            if box is None:
                box = self._detect_mediapipe(prepared_bgr)
            if box is not None:
                candidates.append(box)
        if self._haar_frontal is not None:
            box = self._detect_haar(gray, self._haar_frontal, "haar", 0.80, min_neighbors=4)
            if box is not None:
                candidates.append(box)
        if self._haar_profile is not None:
            box = self._detect_profile(gray)
            if box is not None:
                candidates.append(box)

        if candidates:
            return max(candidates, key=lambda b: (b.confidence, b.w * b.h))

        return self._detect_head(gray)

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

    def _detect_mediapipe(self, frame_bgr: np.ndarray) -> Optional[FaceBox]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._mp_detector.process(rgb)
        detections = result.detections if result and result.detections else []
        best: Optional[FaceBox] = None
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
            box = self._clip_box(box, w, h)
            if best is None or box.confidence > best.confidence:
                best = box
        return best

    def _detect_haar(
        self,
        gray: np.ndarray,
        cascade: cv2.CascadeClassifier,
        source: str,
        confidence: float,
        min_neighbors: int = 5,
    ) -> Optional[FaceBox]:
        frame_h, frame_w = gray.shape[:2]
        min_side = max(28, int(min(frame_w, frame_h) * 0.14))
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=min_neighbors,
            minSize=(min_side, min_side),
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda r: int(r[2]) * int(r[3]))
        return self._clip_box(FaceBox(float(x), float(y), float(w), float(h), confidence, source), frame_w, frame_h)

    def _detect_profile(self, gray: np.ndarray) -> Optional[FaceBox]:
        assert self._haar_profile is not None
        frame_h, frame_w = gray.shape[:2]
        right = self._detect_haar(gray, self._haar_profile, "profile", 0.72, min_neighbors=3)

        flipped = cv2.flip(gray, 1)
        left = self._detect_haar(flipped, self._haar_profile, "profile", 0.72, min_neighbors=3)
        if left is not None:
            left = FaceBox(
                x=float(frame_w - left.x - left.w),
                y=left.y,
                w=left.w,
                h=left.h,
                confidence=left.confidence,
                source="profile",
            )
            left = self._clip_box(left, frame_w, frame_h)

        if right is None:
            return left
        if left is None:
            return right
        return right if right.w * right.h >= left.w * left.h else left

    def _detect_head(self, gray: np.ndarray) -> Optional[FaceBox]:
        frame_h, frame_w = gray.shape[:2]
        if frame_h < 40 or frame_w < 40:
            return None

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh_value = max(70.0, float(np.percentile(blurred, 72.0)))
        _, mask = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        min_area = frame_w * frame_h * 0.025
        max_area = frame_w * frame_h * 0.75
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < frame_w * 0.12 or h < frame_h * 0.15:
                continue
            aspect = w / max(h, 1)
            if not 0.35 <= aspect <= 1.8:
                continue
            # Warm blobs can include neck/shoulders. Keep the top head-like part.
            head_h = min(h, max(int(w * 1.25), int(frame_h * 0.18)))
            candidates.append((area, FaceBox(float(x), float(y), float(w), float(head_h), 0.55, "head")))

        if not candidates:
            return None

        _area, box = max(candidates, key=lambda item: item[0])
        return self._clip_box(box, frame_w, frame_h)

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


def update_face_track(
    detector: DigitalFaceDetector,
    track: FaceTrack,
    digital_frame: np.ndarray,
    frame_index: int,
    detect_interval: int,
    hold_frames: int,
    smoothing: float,
) -> FaceTrack:
    should_detect = track.box is None or frame_index % max(1, detect_interval) == 0
    new_box = detector.detect(digital_frame) if should_detect else None

    if new_box is not None:
        if track.box is not None:
            a = min(max(float(smoothing), 0.0), 0.95)
            new_box = FaceBox(
                x=track.box.x * a + new_box.x * (1.0 - a),
                y=track.box.y * a + new_box.y * (1.0 - a),
                w=track.box.w * a + new_box.w * (1.0 - a),
                h=track.box.h * a + new_box.h * (1.0 - a),
                confidence=new_box.confidence,
                source=new_box.source,
            )
        track.box = new_box
        track.last_seen_frame = frame_index
        track.detector_name = new_box.source
        track.status = face_status_from_source(new_box.source)
        track.missed_frames = 0
    elif track.box is not None and frame_index - track.last_seen_frame <= hold_frames:
        if should_detect:
            track.missed_frames += 1
            track.detector_name = "held"
            track.status = "HELD"
    elif track.box is not None and frame_index - track.last_seen_frame > hold_frames:
        track.box = None
        track.detector_name = "none"
        track.status = "NO FACE"
        track.missed_frames = 0
    else:
        track.status = "NO FACE"
        track.detector_name = "none"
        track.missed_frames = 0
    return track
