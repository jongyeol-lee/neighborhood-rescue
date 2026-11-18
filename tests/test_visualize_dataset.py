"""합성 JPEG와 JSON으로 EXIF 회전, 두 BBOX 해석과 원본 보호를 검증한다."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "02_visualize_dataset.py"


def run_inspection(raw, output):
    """현재 테스트와 같은 Python 환경에서 CLI를 실행한다."""
    return subprocess.run(
        [sys.executable, "-B", str(SCRIPT), "--data-dir", str(raw), "--output-dir", str(output), "--all"],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )


class VisualizeDatasetTest(unittest.TestCase):
    def test_orientation_bounds_and_unmodified_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, output = root / "raw", root / "results"
            raw.mkdir()
            jpeg, label = raw / "sample.jpg", raw / "sample.json"
            with Image.new("RGB", (120, 80), (230, 0, 0)) as stored:
                stored.paste((0, 0, 230), (60, 0, 120, 80))
                exif = stored.getexif()
                exif[274] = 6
                stored.save(jpeg, exif=exif)
            boxes = [[20, 40, 40, 60], [10, 65, 20, 30]]
            label.write_text(json.dumps({
                "info": {"filename": jpeg.name, "width": 120, "height": 80},
                "description": {"state": "0"},
                "annotations": [{"is_defect": "불량", "annotation_type": "bbox", "annotation_info": boxes}],
            }), encoding="utf-8")
            original = {path: path.read_bytes() for path in (jpeg, label)}

            result = run_inspection(raw, output)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual({path: path.read_bytes() for path in original}, original)
            with (output / "bbox_candidates.csv").open(encoding="utf-8-sig", newline="") as stream:
                first, second = csv.DictReader(stream)
            self.assertEqual((first["jpeg_size"], first["display_size"], first["json_size"]), ("120x80", "80x120", "120x80"))
            self.assertEqual(first["json_vs_display"], "mismatch")
            self.assertEqual(json.loads(first["tlwh_xyxy"]), [40, 20, 80, 80])
            self.assertEqual(json.loads(first["ltwh_xyxy"]), [20, 40, 60, 100])
            self.assertEqual(json.loads(second["bbox_original"]), boxes[1])
            self.assertEqual(json.loads(second["tlwh_xyxy"]), [65, 10, 85, 40])
            self.assertEqual((second["tlwh_bounds"], second["ltwh_bounds"]), ("outside", "inside"))
            self.assertAlmostEqual(float(first["tlwh_area_ratio"]), 0.25)
            self.assertEqual(second["tlwh_area_ratio"], "")

            summary = json.loads((output / "eda_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["state_vs_is_defect"]["mismatch"], 1)
            self.assertEqual(summary["issue_counts"]["DISPLAY_SIZE_MISMATCH"], 1)
            self.assertEqual(summary["issue_counts"]["BBOX_ORDER_REVIEW"], 1)
            self.assertEqual(summary["bbox_bounds"]["TLWH"]["in_bounds_stats"]["width_px"]["count"], 1)
            with Image.open(output / "bbox_previews" / "sample.png") as preview:
                preview.load()
                palette = set(preview.get_flattened_data())
                self.assertIn((52, 211, 153), palette)
                self.assertIn((251, 146, 60), palette)
                column = [preview.getpixel((preview.width // 4, y)) for y in range(preview.height)]
                red_rows = [y for y, (r, g, b) in enumerate(column) if r > 170 and g < 50 and b < 50]
                blue_rows = [y for y, (r, g, b) in enumerate(column) if b > 170 and r < 50 and g < 50]
                # 저장 방향의 좌우 색 띠는 Orientation 6 적용 후 각각 위·아래에 있어야 한다.
                self.assertGreater(min(len(red_rows), len(blue_rows)), 45)
                self.assertLess(max(red_rows), min(blue_rows))

    def test_rejects_output_inside_input(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            output = raw / "results"

            result = run_inspection(raw, output)

            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
