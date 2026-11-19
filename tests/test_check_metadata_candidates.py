"""메타데이터 후보 기준과 기존 사용자 판단의 적용 범위를 검증한다."""

import sys
import unittest
from importlib import import_module
from pathlib import Path
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
with patch.object(sys, "path", [str(SCRIPT_DIR), *sys.path]):
    metadata = import_module("04_check_metadata_candidates")


class MetadataCandidateTest(unittest.TestCase):
    def test_new_pair_is_unreviewed_even_if_it_shares_a_reviewed_image(self):
        def image(number, image_id, city_id, filename_date="20211001"):
            return {"image": f"sample_{filename_date}_{number:010d}.jpg", "date": "20211001",
                    "city_id": city_id, "image_id": str(image_id), "phash": "aaaaaaaaaaaaaaaa", "state": "0"}

        first, second = image(100, 70, "4"), image(101, 71, "4")
        third, other_city = image(102, 90, "4"), image(103, 91, "3")
        different_filename_date = image(103, 93, "4", filename_date="20211002")
        key = metadata.pair_key(first["image"], second["image"])
        prior = {key: {"pair_id": "3"}}
        manual = {key: {"review_status": "검토 완료"}}

        (new, overlap), counts = metadata.metadata_candidates(
            [first, second, third, other_city, different_filename_date], prior, manual,
        )

        self.assertEqual((new["a"], new["b"]), (second, third))
        self.assertEqual((new["filename_number_gap"], new["image_id_gap"], new["review_status"]), (1, 19, "미검토"))
        self.assertIsNone(new["prior_phash_pair_id"])
        self.assertEqual((overlap["review_status"], overlap["prior_phash_pair_id"]), ("검토 완료", "3"))
        self.assertEqual(counts["same_date_city_pairs"], 6)
        self.assertEqual(counts["excluded_without_adjacent_number"], 4)


if __name__ == "__main__":
    unittest.main()
