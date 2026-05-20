import tempfile
import unittest
from pathlib import Path

from sonar_mine_detection.data.audit import (
    image_category,
    read_label_classes,
)


class TestAudit(unittest.TestCase):

    def test_label_reading_and_category(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            label_path = Path(tmpdir) / "sample.txt"
            label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            class_ids = read_label_classes(label_path)

            self.assertEqual(class_ids, [0])
            self.assertEqual(image_category(class_ids), "milco_only")


if __name__ == "__main__":
    unittest.main()
