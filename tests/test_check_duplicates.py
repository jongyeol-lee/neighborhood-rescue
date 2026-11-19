"""합성 JPEG로 파일·픽셀 해시, pHash, EXIF와 후보 보고서를 검사한다."""

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageEnhance


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "03_check_duplicates.py"
with patch.object(sys, "path", [str(SCRIPT.parent), *sys.path]):
    duplicates = import_module(SCRIPT.stem)


def run_analysis(raw, output, *args):
    """현재 Python 환경에서 중복 검사의 CLI를 실행한다."""
    command = [sys.executable, "-B", str(SCRIPT), "--data-dir", str(raw), "--output-dir", str(output), *args]
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=False)


def write_label(image, state, city_id="1"):
    """실제 파일명에 대응하는 최소한의 AI-Hub 형식 라벨을 만든다."""
    payload = {"info": {"filename": image.name, "date": "20210906", "city_id": city_id},
               "description": {"state": state}}
    image.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")


def read_report(output):
    """생성된 요약 보고서를 읽는다."""
    return json.loads((output / "duplicate_report.json").read_text(encoding="utf-8"))


class CheckDuplicatesTest(unittest.TestCase):
    def test_phash_identical_brightness_changed_and_unrelated(self):
        rng = np.random.default_rng(64)
        with Image.fromarray(rng.integers(20, 200, (64, 64), dtype=np.uint8)) as image:
            with image.copy() as same, ImageEnhance.Brightness(image).enhance(1.1) as brighter:
                reference = duplicates.perceptual_hash(image)
                self.assertEqual((reference ^ duplicates.perceptual_hash(same)).bit_count(), 0)
                self.assertLessEqual((reference ^ duplicates.perceptual_hash(brighter)).bit_count(), 8)
        with Image.fromarray(rng.integers(20, 200, (64, 64), dtype=np.uint8)) as different:
            self.assertGreater((reference ^ duplicates.perceptual_hash(different)).bit_count(), 8)

    def test_exact_groups_and_previews_use_exif_display_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, output = root / "raw", root / "output"
            raw.mkdir()
            first, copy, metadata_only = (raw / name for name in ("first.jpg", "copy.jpg", "metadata.jpg"))
            with Image.new("RGB", (160, 96), (230, 0, 0)) as image:
                image.paste((0, 0, 230), (80, 0, 160, 96))
                exif = image.getexif()
                exif[274] = 6
                image.save(first, exif=exif)
                exif[315] = "synthetic"
                image.save(metadata_only, exif=exif)
            shutil.copyfile(first, copy)
            write_label(first, "0")
            write_label(copy, "1", city_id="2")
            write_label(metadata_only, "0")
            before = {path: path.read_bytes() for path in raw.iterdir()}

            result = run_analysis(raw, output, "--threshold", "0", "--max-previews", "1")

            self.assertEqual(result.returncode, 0, result.stderr)
            report = read_report(output)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["counts"]["phash_computed"], 3)
            file_duplicates, pixels = report["exact_file_duplicates"], report["exact_display_pixel_duplicates"]
            self.assertEqual((file_duplicates["group_count"], file_duplicates["image_count"]), (1, 2))
            self.assertEqual((pixels["group_count"], pixels["image_count"]), (1, 3))
            self.assertEqual(file_duplicates["groups_with_different_states"], 1)
            self.assertEqual(pixels["groups_with_different_states"], 1)
            records = {record["image"]: record for record in report["images"]}
            self.assertEqual(records[first.name]["file_sha256"], records[copy.name]["file_sha256"])
            self.assertNotEqual(records[first.name]["file_sha256"], records[metadata_only.name]["file_sha256"])
            with Image.open(first) as encoded, encoded.transpose(Image.Transpose.ROTATE_270) as expected:
                self.assertEqual(records[first.name]["display_size"], [96, 160])
                self.assertEqual(records[first.name]["display_pixel_sha256"], duplicates.pixel_sha256(expected))
                self.assertEqual(records[first.name]["phash"], f"{duplicates.perceptual_hash(expected):016x}")
            self.assertEqual(report["phash"]["compared_pairs"], 3)
            self.assertEqual(report["phash"]["distance_distribution"]["0"], 3)
            self.assertEqual(report["phash"]["pairs_with_different_states"], 2)
            self.assertEqual(len(report["review"]["candidate_groups"]), 1)

            with (output / "similar_pairs.csv").open(encoding="utf-8-sig", newline="") as stream:
                best, *other = csv.DictReader(stream)
            self.assertEqual({best["a_image"], best["b_image"]}, {first.name, copy.name})
            self.assertEqual((best["phash_distance"], best["different_states"], best["same_city_id"]), ("0", "True", "False"))
            self.assertEqual(len(other), 2)
            self.assertTrue(all(not row["preview"] for row in other))
            with Image.open(output / best["preview"]) as preview:
                red, _, _ = preview.getpixel((324, 325))
                _, _, blue = preview.getpixel((324, 415))
                self.assertGreater(min(red, blue), 200)
            self.assertEqual({path: path.read_bytes() for path in raw.iterdir()}, before)

    def test_no_candidates_still_writes_a_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, output = root / "raw", root / "output"
            raw.mkdir()
            rng = np.random.default_rng(76)
            for name in ("one.jpg", "two.jpg"):
                image = raw / name
                with Image.fromarray(rng.integers(0, 256, (64, 64), dtype=np.uint8)) as synthetic:
                    synthetic.save(image)
                write_label(image, "0")

            result = run_analysis(raw, output, "--threshold", "0")

            self.assertEqual(result.returncode, 0, result.stderr)
            report = read_report(output)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["phash"]["compared_pairs"], 1)
            self.assertEqual(report["phash"]["candidate_pair_count"], 0)
            self.assertEqual(report["review"]["preview_count"], 0)
            self.assertFalse((output / "review").exists())
            with (output / "similar_pairs.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = csv.DictReader(stream)
                self.assertIn("phash_distance", rows.fieldnames)
                self.assertEqual(list(rows), [])

    def test_output_inside_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            output = raw / "results"

            result = run_analysis(raw, output)

            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
