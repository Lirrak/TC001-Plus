import unittest

import numpy as np

from tc001_thermal_viewer_v5 import polygon_temperature_stats


class RoiTemperatureTests(unittest.TestCase):
    def test_robust_temperature_ignores_small_hot_object_inside_roi(self):
        temp = np.full((60, 60), 30.0, dtype=np.float32)
        temp[10:50, 10:50] = 36.8
        temp[22:27, 22:27] = 43.5
        polygon = np.array([[10, 10], [49, 10], [49, 49], [10, 49]], dtype=np.float32)

        stats = polygon_temperature_stats(
            temp,
            polygon,
            mode="robust",
            percentile=90.0,
            hot_outlier_delta_c=2.5,
            mask_shrink=1.0,
        )

        self.assertAlmostEqual(stats.person_temp_c, 36.8, places=1)
        self.assertAlmostEqual(stats.max_temp_c, 43.5, places=1)
        self.assertTrue(stats.roi_temp_contaminated)
        self.assertGreater(stats.hot_outlier_pixels, 0)

    def test_max_mode_keeps_old_hottest_pixel_behavior(self):
        temp = np.full((40, 40), 36.5, dtype=np.float32)
        temp[18:20, 18:20] = 42.0
        polygon = np.array([[5, 5], [34, 5], [34, 34], [5, 34]], dtype=np.float32)

        stats = polygon_temperature_stats(temp, polygon, mode="max", mask_shrink=1.0)

        self.assertAlmostEqual(stats.person_temp_c, 42.0, places=1)
        self.assertAlmostEqual(stats.max_temp_c, 42.0, places=1)
        self.assertFalse(stats.roi_temp_contaminated)

    def test_broad_hot_human_region_is_not_treated_as_object_outlier(self):
        temp = np.full((50, 50), 30.0, dtype=np.float32)
        temp[8:42, 8:42] = 39.0
        polygon = np.array([[8, 8], [41, 8], [41, 41], [8, 41]], dtype=np.float32)

        stats = polygon_temperature_stats(
            temp,
            polygon,
            mode="robust",
            percentile=90.0,
            hot_outlier_delta_c=2.5,
            mask_shrink=1.0,
        )

        self.assertAlmostEqual(stats.person_temp_c, 39.0, places=1)
        self.assertAlmostEqual(stats.max_temp_c, 39.0, places=1)
        self.assertFalse(stats.roi_temp_contaminated)


if __name__ == "__main__":
    unittest.main()
