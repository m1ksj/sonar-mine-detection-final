import unittest

from sonar_mine_detection.evaluation.darknet_metrics import (
    add_ground_truth_counts,
    parse_darknet_map_text,
)


SAMPLE = """
class_id = 0, name = MILCO, ap = 59.10%       (TP = 36, FP = 16)
class_id = 1, name = NOMBO, ap = 41.52%       (TP = 10, FP = 12)
 for conf_thresh = 0.25, precision = 0.62, recall = 0.50, F1-score = 0.55
 mean average precision (mAP@0.50) = 0.503082, or 50.31 %
"""

SAMPLE_WITHOUT_SUMMARY = """
class_id = 0, name = MILCO, ap = 59.10%       (TP = 36, FP = 16)
class_id = 1, name = NOMBO, ap = 41.52%       (TP = 10, FP = 12)
 mean average precision (mAP@0.50) = 0.503082, or 50.31 %
"""


class TestDarknetMetrics(unittest.TestCase):
    def test_parse_aggregate_and_class_metrics(self):
        aggregate, class_rows = parse_darknet_map_text(SAMPLE)

        self.assertEqual(aggregate["test_precision"], 0.62)
        self.assertEqual(aggregate["test_recall"], 0.50)
        self.assertEqual(aggregate["test_f1"], 0.55)
        self.assertEqual(aggregate["test_map50"], 0.503082)

        self.assertEqual(class_rows[0]["class_name"], "MILCO")
        self.assertAlmostEqual(class_rows[0]["ap50"], 0.591)
        self.assertEqual(class_rows[0]["tp"], 36)
        self.assertEqual(class_rows[0]["fp"], 16)

    def test_add_ground_truth_counts(self):
        _, class_rows = parse_darknet_map_text(SAMPLE)
        rows = add_ground_truth_counts(class_rows, {0: 72, 1: 25})

        self.assertEqual(rows[0]["fn"], 36)
        self.assertEqual(rows[1]["fn"], 15)
        self.assertAlmostEqual(rows[0]["class_recall"], 0.5)

    def test_parse_without_precision_summary(self):
        aggregate, class_rows = parse_darknet_map_text(SAMPLE_WITHOUT_SUMMARY)

        self.assertEqual(aggregate["test_precision"], "")
        self.assertEqual(aggregate["test_recall"], "")
        self.assertEqual(aggregate["test_f1"], "")
        self.assertEqual(aggregate["test_map50"], 0.503082)
        self.assertEqual(len(class_rows), 2)
        self.assertEqual(class_rows[0]["class_name"], "MILCO")


if __name__ == "__main__":
    unittest.main()
