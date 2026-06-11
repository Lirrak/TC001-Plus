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


class DigitalFaceDetector:
    def __init__(self, model: str = "auto", min_confidence: float = 0.55) -> None:
        self.requested_model = model
        self.min_confidence = float(min_confidence)
        self.name = "none"
        self._mp_detector = None
        self._haar = None

        if model in ("auto", "mediapipe"):
            try:
                import mediapipe as mp  # type: ignore

                self._mp_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=self.min_confidence,
                )
                self.name = "mediapipe"
            except Exception as exc:
                if model == "mediapipe":
                    print(f"WARNING: MediaPipe face detector unavailable: {exc}")

        if self._mp_detector is None and model in ("auto", "haar"):
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            self._haar = cv2.CascadeClassifier(cascade_path)
            if self._haar.empty():
                print(f"WARNING: Could not load Haar cascade: {cascade_path}")
                self._haar = None
            else:
                self.name = "haar"

        if self.name == "none":
            print("WARNING: No face detector is available. Install mediapipe or use a full opencv-python build.")

    def detect(self, frame_bgr: np.ndarray) -> Optional[FaceBox]:
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        if self._mp_detector is not None:
            return self._detect_mediapipe(frame_bgr)
        if self._haar is not None:
            return self._detect_haar(frame_bgr)
        return None

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
            if best is None or box.confidence > best.confidence:
                best = box
        return best

    def _detect_haar(self, frame_bgr: np.ndarray) -> Optional[FaceBox]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda r: int(r[2]) * int(r[3]))
        return FaceBox(float(x), float(y), float(w), float(h), 0.80, "haar")


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
        track.detector_name = detector.name
    elif track.box is not None and frame_index - track.last_seen_frame > hold_frames:
        track.box = None
    return track
