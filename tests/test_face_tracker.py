import unittest

import numpy as np

from tc001_face import FaceBox, FaceTracker, update_face_tracks


class FakeDetector:
    name = "fake"
    requested_model = "fake"

    def __init__(self, frames):
        self.frames = list(frames)
        self.last_debug = {}

    def detect_all(self, frame, max_faces=5, **_kwargs):
        boxes = self.frames.pop(0) if self.frames else []
        self.last_debug = {
            "primary_candidates": len(boxes),
            "final_after_nms": len(boxes),
            "final_boxes": [
                {
                    "source": box.source,
                    "bbox": [box.x, box.y, box.w, box.h],
                }
                for box in boxes
            ],
        }
        return boxes[:max_faces]


def update_once(detector, tracker, frame_index=1, frame_shape=(100, 180, 3), hold_frames=5, reid_frames=45):
    frame = np.zeros(frame_shape, dtype=np.uint8)
    return update_face_tracks(
        detector,
        tracker,
        frame,
        frame_index=frame_index,
        detect_interval=1,
        hold_frames=hold_frames,
        smoothing=0.0,
        max_faces=5,
        min_face_hits=1,
        max_face_area_ratio=1.0,
        max_box_overlap=0.30,
        face_reid_frames=reid_frames,
    )


class FaceTrackerTests(unittest.TestCase):
    def test_near_non_duplicate_boxes_create_two_tracks(self):
        detector = FakeDetector(
            [
                [
                    FaceBox(10, 20, 50, 50, 0.9, "haar"),
                    FaceBox(45, 20, 50, 50, 0.9, "haar"),
                ]
            ]
        )
        tracker = update_once(detector, FaceTracker())

        tracks = tracker.active_tracks(confirmed_only=True)
        self.assertEqual(len(tracks), 2)
        self.assertEqual(tracker.last_stats["new_tracks"], 2)
        self.assertEqual(tracker.last_stats["duplicate_suppressed"], 0)

    def test_overlapping_duplicate_box_is_suppressed(self):
        detector = FakeDetector(
            [
                [
                    FaceBox(10, 20, 50, 50, 0.9, "haar"),
                    FaceBox(25, 20, 50, 50, 0.9, "haar"),
                ]
            ]
        )
        tracker = update_once(detector, FaceTracker())

        tracks = tracker.active_tracks(confirmed_only=True)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracker.last_stats["new_tracks"], 1)
        self.assertEqual(tracker.last_stats["duplicate_suppressed"], 1)

    def test_reordered_detections_keep_existing_track_ids(self):
        detector = FakeDetector(
            [
                [
                    FaceBox(10, 20, 40, 40, 0.9, "haar"),
                    FaceBox(100, 20, 40, 40, 0.9, "haar"),
                ],
                [
                    FaceBox(102, 20, 40, 40, 0.9, "haar"),
                    FaceBox(12, 20, 40, 40, 0.9, "haar"),
                ],
            ]
        )
        tracker = update_once(detector, FaceTracker(), frame_index=1)
        tracker = update_once(detector, tracker, frame_index=2)

        tracks = {track.track_id: track for track in tracker.active_tracks(confirmed_only=True)}
        self.assertAlmostEqual(tracks[1].box.x, 12)
        self.assertAlmostEqual(tracks[2].box.x, 102)

    def test_recently_lost_track_is_reidentified_without_showing_stale_box(self):
        detector = FakeDetector(
            [
                [FaceBox(10, 20, 40, 40, 0.9, "haar")],
                [],
                [FaceBox(14, 22, 40, 40, 0.9, "haar")],
            ]
        )
        tracker = update_once(detector, FaceTracker(), frame_index=1, hold_frames=1, reid_frames=10)
        tracker = update_once(detector, tracker, frame_index=4, hold_frames=1, reid_frames=10)

        self.assertEqual(tracker.active_tracks(), [])
        self.assertEqual([track.track_id for track in tracker.retired_tracks], [1])

        tracker = update_once(detector, tracker, frame_index=5, hold_frames=1, reid_frames=10)
        tracks = tracker.active_tracks(confirmed_only=True)
        self.assertEqual([track.track_id for track in tracks], [1])
        self.assertEqual(tracker.last_stats["reidentified_tracks"], 1)


if __name__ == "__main__":
    unittest.main()
