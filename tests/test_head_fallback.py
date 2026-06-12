import unittest

import cv2
import numpy as np

from tc001_face import DigitalFaceDetector


class HeadFallbackTests(unittest.TestCase):
    def test_large_connected_side_profile_contour_yields_head_box(self):
        detector = DigitalFaceDetector(model="haar")
        gray = np.zeros((256, 192), dtype=np.uint8)

        cv2.ellipse(gray, (104, 72), (34, 42), 0, 0, 360, 220, -1)
        cv2.rectangle(gray, (82, 104), (154, 224), 220, -1)
        cv2.rectangle(gray, (50, 142), (110, 206), 180, -1)
        cv2.GaussianBlur(gray, (5, 5), 0, dst=gray)

        boxes = detector._detect_head_all(gray, 0.65)

        self.assertTrue(boxes)
        best = min(boxes, key=lambda box: box.y)
        self.assertLess(best.y, 90)
        self.assertLessEqual((best.w * best.h) / float(gray.shape[0] * gray.shape[1]), 0.20)
        self.assertGreaterEqual(best.w, gray.shape[1] * 0.16)


if __name__ == "__main__":
    unittest.main()
