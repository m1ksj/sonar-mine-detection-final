from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import unittest

from sonar_mine_detection.data.yolo_export import export_yolo_datasets


class TestYoloExport(unittest.TestCase):
    def test_darknet_labels_are_written_next_to_images(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            splits_dir = root / "splits"
            yolo26n_dir = root / "yolo26n"
            darknet_dir = root / "darknet"

            source_dir.mkdir()
            splits_dir.mkdir()

            image_path = source_dir / "sample.jpg"
            label_path = source_dir / "sample.txt"

            image_path.write_bytes(b"fake image")
            label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            for split in ["train", "val", "test"]:
                with (splits_dir / f"{split}.csv").open(
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as file:
                    writer = csv.DictWriter(
                        file,
                        fieldnames=["image_path", "label_path"],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "image_path": str(image_path),
                            "label_path": str(label_path),
                        }
                    )

            export_yolo_datasets(
                splits_dir=splits_dir,
                yolo26n_dir=yolo26n_dir,
                darknet_dir=darknet_dir,
            )

            expected_image = darknet_dir / "images" / "train" / "sample.jpg"
            expected_darknet_label = (
                darknet_dir / "images" / "train" / "sample.txt"
            )
            expected_label_copy = (
                darknet_dir / "labels" / "train" / "sample.txt"
            )
            expected_backup = darknet_dir / "backup"

            self.assertTrue(expected_image.exists())
            self.assertTrue(expected_darknet_label.exists())
            self.assertTrue(expected_label_copy.exists())
            self.assertTrue(expected_backup.exists())
            self.assertEqual(
                expected_darknet_label.read_text(encoding="utf-8"),
                "0 0.5 0.5 0.2 0.2\n",
            )


if __name__ == "__main__":
    unittest.main()
