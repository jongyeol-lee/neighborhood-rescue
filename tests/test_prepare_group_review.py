"""검토 승인, 연결 모순, 그룹 ID와 검토표 보존을 검증한다."""

import sys
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
with patch.object(sys, "path", [str(SCRIPT_DIR), *sys.path]):
    review = import_module("05_prepare_group_review")


def decision(a, b, result, *, final=review.YES, place="", source="메타데이터"):
    """사람이 내린 판정을 담은 검토표 행을 구성한다."""
    return {"review_id": f"{source}-{a}-{b}", "source": source, "a_image": a, "b_image": b,
            "review_result": result, "review_basis": "시험 근거", "reviewer": "사용자",
            "final_confirmed": final, "same_place_confirmed": place, "relation_for_group": result}


def image_records(paths):
    """그룹 계산에 필요한 원본 경로, 해시와 상태를 구성한다."""
    return {path: {"image": path, "file_sha256": str(index) * 64, "state": "0", "state_name": "양호"}
            for index, path in enumerate(paths)}


class GroupReviewTest(unittest.TestCase):
    def test_only_confirmed_relations_group_and_ids_survive_row_reordering(self):
        records = image_records(("a", "b", "c", "d"))
        queue = [decision("a", "b", review.SAME_FACILITY), decision("b", "c", review.SAME_PLACE),
                 decision("c", "d", review.SAME_FACILITY, final=review.NO)]

        mapping, groups = review.build_groups(records, queue)

        self.assertEqual([group["images"] for group in groups["facility_groups"]], [["a", "b"]])
        self.assertEqual([group["images"] for group in groups["location_groups"]], [["b", "c"]])
        self.assertEqual(mapping[0]["provisional_unit_id"], mapping[2]["provisional_unit_id"])
        self.assertNotEqual(mapping[2]["provisional_unit_id"], mapping[3]["provisional_unit_id"])
        self.assertEqual(mapping[3]["status"], "후보 검토 필요")
        reordered = dict(reversed(tuple(records.items())))
        self.assertEqual((mapping, groups), review.build_groups(reordered, reversed(queue)))

    def test_conflicting_final_relation_withholds_entire_component_and_keeps_evidence(self):
        records = image_records(("a", "b", "c"))
        first = decision("a", "b", review.SAME_FACILITY, place=review.YES)
        second = decision("b", "c", review.SAME_FACILITY, place=review.YES)
        opposing = decision("a", "c", review.DIFFERENT, source="pHash")

        mapping, groups = review.build_groups(records, [first, second, opposing])

        self.assertEqual((groups["facility_count"], groups["location_count"]), (0, 0))
        for field in ("facility_conflicts", "location_conflicts"):
            group, = groups[field]
            self.assertEqual(group["images"], ["a", "b", "c"])
            self.assertEqual(group["supporting_reviews"], [first, second])
            self.assertEqual(group["opposing_reviews"], [opposing])
        self.assertEqual({row["status"] for row in mapping}, {"판정 모순 확인 필요"})

    def test_manual_ledger_is_preserved_and_requires_basis_for_final(self):
        records = image_records(("a.jpg", "b.jpg"))
        key = ("a.jpg", "b.jpg")
        candidates = {key: {"pair_id": "1", "a_image": key[0], "b_image": key[1]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata_user_reviews.csv"
            reviews = review.load_or_create_reviews(path, candidates, records)
            row = reviews[key]
            self.assertEqual(row["final_confirmed"], review.NO)
            row.update(observation="검토 메모", review_result=review.SAME_FACILITY, final_confirmed=review.YES,
                       reviewer="사용자")
            review.write_csv(path, review.LEDGER_FIELDS, [row])
            with self.assertRaisesRegex(ValueError, "판단 근거"):
                review.load_or_create_reviews(path, candidates, records)

            row["review_basis"] = "확인 근거"
            review.write_csv(path, review.LEDGER_FIELDS, [row])
            before = path.read_bytes()
            self.assertEqual(review.load_or_create_reviews(path, candidates, records), reviews)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
