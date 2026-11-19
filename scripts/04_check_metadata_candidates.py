"""날짜·지역 코드와 인접한 ID 값을 이용해 볼라드 사진의 검토 후보를 찾는다."""

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from importlib import import_module
from itertools import combinations
from pathlib import Path


duplicates = import_module("03_check_duplicates")
ANALYSIS_DIR = duplicates.OUTPUT_DIR
FILENAME_SUFFIX = re.compile(r"_(?P<date>[0-9]{8})_(?P<number>[0-9]{10})$")
DEFAULT_MAX_PREVIEWS = 30
PAIR_FIELDS = (
    "pair_id", "candidate_source", "review_status", "prior_phash_pair_id", "date", "city_id",
    "filename_number_gap", "image_id_gap", "phash_distance", "candidate_reason", "preview",
)


def pair_key(first, second):
    """CSV의 A/B 순서와 무관하게 두 원본 경로로 후보 쌍을 식별한다."""
    return tuple(sorted((first, second)))


def read_pair_csv(path):
    """원본 경로 쌍을 키로 CSV를 읽고 중복 행을 검사한다."""
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = pair_key(row["a_image"], row["b_image"])
            if key in rows or key[0] == key[1]:
                raise ValueError(f"{path.name}에 중복 또는 자기 자신과의 쌍이 있습니다: {key}")
            rows[key] = row
    return rows


def load_prior_results(phash_dir, review_file, records):
    """기존 분석이 현재 데이터와 일치하는지 확인한 뒤 사용자 기록을 읽기 전용으로 읽는다."""
    report = json.loads((phash_dir / "duplicate_report.json").read_text(encoding="utf-8"))
    fields = ("file_sha256", "phash", "json", "date", "city_id", "image_id", "state")
    old_images = {row["image"]: tuple(row[field] for field in fields) for row in report["images"]}
    current_images = {row["image"]: tuple(row[field] for field in fields) for row in records}
    if report["status"] != "complete" or old_images != current_images:
        raise ValueError("현재 JPEG·JSON과 기존 pHash 보고서가 일치하지 않습니다. 기존 분석 결과를 확인하세요.")

    threshold = report["phash"]["threshold"]
    prior = read_pair_csv(phash_dir / "similar_pairs.csv")
    recomputed, _ = duplicates.compare_pairs(records, threshold)
    expected = {pair_key(pair["a"]["image"], pair["b"]["image"]): str(pair["phash_distance"]) for pair in recomputed}
    if {key: row["phash_distance"] for key, row in prior.items()} != expected:
        raise ValueError("기존 후보 CSV의 쌍 또는 거리가 현재 pHash 검사와 일치하지 않습니다.")

    manual = read_pair_csv(review_file)
    for key, row in manual.items():
        if row["review_status"] not in ("검토 완료", "미검토"):
            raise ValueError(f"사용자 기록의 검토 상태를 확인하세요: {key}")
        if (key not in prior or row["phash_distance"] != prior[key]["phash_distance"]
                or any(row[f"{side}_file_sha256"] != current_images[row[f"{side}_image"]][0] for side in ("a", "b"))):
            raise ValueError(f"현재 후보의 경로·거리·SHA-256과 일치하지 않는 사용자 검토 기록이 있습니다: {key}")
    return threshold, prior, manual


def metadata_candidates(records, prior, manual):
    """날짜·지역 코드가 같고 두 숫자 단서 중 하나가 인접한 고유 쌍을 찾는다."""
    by_date_city = defaultdict(list)
    for image in records:
        recorded_date, city_code = image["date"], image["city_id"]
        if not recorded_date or not city_code or not re.fullmatch(r"[0-9]{8}", recorded_date):
            continue
        try:
            date.fromisoformat(recorded_date)
        except ValueError:
            continue
        suffix = FILENAME_SUFFIX.search(Path(image["image"]).stem)
        filename_number = int(suffix["number"]) if suffix and suffix["date"] == recorded_date else None
        image_id = image["image_id"]
        numeric_id = int(image_id) if image_id and re.fullmatch(r"[0-9]+", image_id) else None
        by_date_city[recorded_date, city_code].append((image, filename_number, numeric_id))

    pairs = []
    same_date_city_pairs = 0
    for (recorded_date, city_code), images in sorted(by_date_city.items()):
        for (first, file_a, id_a), (second, file_b, id_b) in combinations(images, 2):
            same_date_city_pairs += 1
            file_gap = None if file_a is None or file_b is None else abs(file_a - file_b)
            id_gap = None if id_a is None or id_b is None else abs(id_a - id_b)
            if 1 not in (file_gap, id_gap):
                continue
            key = pair_key(first["image"], second["image"])
            existing = prior.get(key)
            reasons = ["날짜·지역 코드 일치"]
            if file_gap == 1:
                reasons.append("파일명 끝 번호 차이 1")
            if id_gap == 1:
                reasons.append("info.image_id 차이 1")
            distance = (int(first["phash"], 16) ^ int(second["phash"], 16)).bit_count()
            pairs.append({
                "a": first, "b": second, "date": recorded_date, "city_id": city_code,
                "filename_number_gap": file_gap, "image_id_gap": id_gap, "phash_distance": distance,
                "different_states": first["state"] != second["state"], "candidate_reason": " / ".join(reasons),
                "prior_phash_pair_id": existing["pair_id"] if existing else None,
                "candidate_source": "기존 pHash와 겹침" if existing else "신규 메타데이터 후보",
                "review_status": manual.get(key, {}).get("review_status", "미검토"),
                "preview": None,
            })
    pairs.sort(key=lambda pair: (
        pair["prior_phash_pair_id"] is not None,
        max(float("inf") if gap is None else gap for gap in (pair["filename_number_gap"], pair["image_id_gap"])),
        pair["phash_distance"], pair["a"]["image"], pair["b"]["image"],
    ))
    for pair_id, pair in enumerate(pairs, start=1):
        pair["pair_id"] = pair_id

    total = len(records) * (len(records) - 1) // 2
    return pairs, {
        "images": len(records), "date_city_groups": len(by_date_city),
        "images_with_valid_date_city": sum(map(len, by_date_city.values())), "possible_pairs": total,
        "same_date_city_pairs": same_date_city_pairs,
        "excluded_different_or_missing_date_city": total - same_date_city_pairs,
        "excluded_without_adjacent_number": same_date_city_pairs - len(pairs), "metadata_candidate_pairs": len(pairs),
    }


def write_results(output, root, pairs, counts, threshold, prior, manual, args):
    """새 후보의 비교 PNG와 전체 후보 CSV, 사용자 검토를 참조한 JSON을 생성한다."""
    new_pairs = [pair for pair in pairs if pair["prior_phash_pair_id"] is None]
    selected = new_pairs[:args.max_previews]
    output.mkdir(parents=True, exist_ok=True)
    if selected:
        signature = hashlib.sha256(json.dumps(selected, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
        preview_dir = output / "review" / signature
        if preview_dir.resolve() != preview_dir:
            raise ValueError(f"비교 이미지 디렉터리의 연결 경로를 확인하세요: {preview_dir}")
        preview_dir.mkdir(parents=True, exist_ok=True)
        for pair in selected:
            path = preview_dir / f"metadata_pair_{pair['pair_id']:04d}.png"
            duplicates.save_preview(path, root, pair, title_prefix="Metadata candidate")
            pair["preview"] = path.relative_to(output).as_posix()

    side_fields = ("image", "json", "state_name", "image_id")
    columns = (*PAIR_FIELDS, *(f"{side}_{field}" for side in ("a", "b") for field in side_fields))
    rows = []
    for pair in pairs:
        row = {name: pair[name] for name in PAIR_FIELDS}
        row.update({f"{side}_{field}": pair[side][field] for side in ("a", "b") for field in side_fields})
        rows.append(row)
    with duplicates.inspection.output_file(output / "metadata_candidates.csv", mode="w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    review_counts = Counter(row["review_result"] for row in manual.values() if row["review_status"] == "검토 완료")
    report = {
        "status": "complete", "data_root": str(root),
        "path_reference": "image와 json은 data_root 기준이며 preview는 metadata_report.json과 같은 디렉터리 기준입니다.",
        "user_review": {
            "file": str(args.review_file.resolve()), "prior_phash_threshold": threshold, "prior_phash_pairs": len(prior),
            "completed_pairs": review_counts.total(), "results": dict(sorted(review_counts.items())),
            "records": list(manual.values()),
            "confirmed_same_facility_pairs": [list(key) for key, row in manual.items()
                                              if row["review_status"] == "검토 완료" and row["review_result"] == "동일 시설물"],
        },
        "criteria": {
            "date_city": "원본 JSON info.date가 같은 유효 날짜이고 info.city_id가 같은 쌍을 비교합니다.",
            "adjacency": "파일명 끝 번호 또는 숫자로 된 info.image_id의 절대 차이가 1이면 후보에 포함합니다.",
            "filename_check": "파일명 끝의 날짜가 JSON 날짜와 일치하고 마지막 10자가 숫자일 때만 끝 번호를 사용합니다.",
            "priority": "신규 후보 우선, 두 숫자 차이 중 큰 값이 작은 순, pHash 거리 오름차순, 경로순입니다.",
        },
        "counts": {**counts, "overlap_with_prior_phash": len(pairs) - len(new_pairs), "new_candidates": len(new_pairs),
                   "new_unreviewed_candidates": sum(pair["review_status"] == "미검토" for pair in new_pairs),
                   "new_with_both_numbers_adjacent": sum(pair["filename_number_gap"] == pair["image_id_gap"] == 1 for pair in new_pairs),
                   "preview_count": len(selected), "new_without_preview_due_to_limit": len(new_pairs) - len(selected)},
        "candidates_file": "metadata_candidates.csv", "candidates": rows,
        "limitations": [
            "파일명 끝 번호와 image_id는 인접성 단서입니다. 촬영 순서나 장소 식별자로 확인되지 않았습니다.",
            "같은 시설물도 날짜·지역 코드가 다르거나 두 번호의 차이가 1보다 크면 검사에서 빠질 수 있습니다.",
            "촬영 시각, 세부 위치 또는 시설물 식별자는 이번 원본 JSON 구조에서 확인되지 않았습니다.",
            "기존 사용자 판단은 해당 원본 쌍에만 적용했습니다. 미검토 후보는 시설물 그룹으로 확정하지 않습니다.",
            "이번 점검은 분할 전 위험 탐색입니다. 동일 시설물이나 연속 장면으로 확인된 사진은 분할에서 함께 배정해야 합니다.",
        ],
    }
    with duplicates.inspection.output_file(output / "metadata_report.json", mode="w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report["counts"]


def main(argv=None):
    """원본과 기존 검토 기록을 확인하고 추가 후보 분석을 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=duplicates.RAW_DIR)
    parser.add_argument("--phash-dir", type=Path, default=ANALYSIS_DIR / "threshold_16")
    parser.add_argument("--review-file", type=Path, default=ANALYSIS_DIR / "phash_user_review.csv")
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR / "metadata_review")
    parser.add_argument("--max-previews", type=int, default=DEFAULT_MAX_PREVIEWS, help="신규 후보 비교 이미지 상한 (기본: 30)")
    args = parser.parse_args(argv)
    root, output = args.data_dir.resolve(), args.output_dir.resolve()
    if not root.is_dir() or any(output.is_relative_to(path) for path in (root, duplicates.RAW_DIR.resolve())):
        parser.error("입력 디렉터리와 원본 밖의 별도 결과 경로를 확인하세요.")
    if args.review_file.resolve() in (output / "metadata_report.json", output / "metadata_candidates.csv"):
        parser.error("사용자 검토 CSV와 자동 결과 파일의 경로가 겹칩니다.")
    if args.max_previews < 1:
        parser.error("--max-previews는 1 이상이어야 합니다.")

    issues = []
    records, _ = duplicates.inspect_images(root, issues)
    if issues:
        parser.exit(1, f"원본 검사에서 오류 {len(issues)}건이 발견되었습니다. {issues[0]['detail']}\n")
    try:
        threshold, prior, manual = load_prior_results(args.phash_dir, args.review_file, records)
        pairs, counts = metadata_candidates(records, prior, manual)
        totals = write_results(output, root, pairs, counts, threshold, prior, manual, args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"메타데이터 검사 오류: {error}\n")
    print(f"신규 후보 {totals['new_candidates']}쌍, 기존 pHash와 겹침 {totals['overlap_with_prior_phash']}쌍, "
          f"비교 이미지 {totals['preview_count']}장")
    print(f"결과 디렉터리: {output}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
