"""사람이 확인한 후보 관계로 시설물·촬영 장소 그룹과 분할 검토 자료를 만든다."""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from importlib import import_module
from pathlib import Path
from tempfile import NamedTemporaryFile


metadata = import_module("04_check_metadata_candidates")
duplicates = metadata.duplicates
SAME_FACILITY = "동일 시설물"
SAME_PLACE = "같은 촬영 장소의 다른 시설물"
DIFFERENT = "다른 시설물·다른 장소"
OTHER_FACILITY = "서로 다른 시설물"
FINAL_RESULTS = (SAME_FACILITY, SAME_PLACE, DIFFERENT)
REVIEW_RESULTS = (*FINAL_RESULTS, "판단 보류", "미검토")
YES, NO = "예", "아니오"
LEDGER_FIELDS = (
    "pair_id", "a_filename", "b_filename", "a_image", "b_image", "a_file_sha256", "b_file_sha256",
    "label_mismatch", "observation", "observation_source", "label_review_note",
    "review_result", "review_basis", "reviewer", "final_confirmed", "same_place_confirmed",
)
QUEUE_FIELDS = (
    "review_id", "source", "metadata_pair_id", "phash_pair_id", "priority", "review_status", "phash_distance",
    "label_mismatch", "filename_number_gap", "image_id_gap", "candidate_reason", "preview",
    "observation", "observation_source", "label_review_note", "review_result", "review_basis", "reviewer",
    "final_confirmed", "same_place_confirmed", "relation_for_group",
    *(f"{side}_{field}" for side in ("a", "b") for field in (
        "filename", "image", "json", "state", "state_name", "date", "city_id", "image_id",
    )),
)


def load_sources(analysis):
    """원본의 내용과 기존 분석의 경로·후보 쌍이 일치하는지 확인한다."""
    prior_dir = analysis / "threshold_16"
    report = json.loads((prior_dir / "duplicate_report.json").read_text(encoding="utf-8"))
    root = Path(report["data_root"]).resolve(strict=True)
    if analysis.is_relative_to(root) or analysis.is_relative_to(duplicates.RAW_DIR.resolve()):
        raise ValueError("원본 밖의 분석 디렉터리가 필요합니다.")

    records = {record["image"]: record for record in report["images"]}
    issues = []
    images, labels = duplicates.inspection.discover_files(root, issues)
    for discovered, field in ((images, "image"), (labels, "json")):
        paths = {path.relative_to(root).as_posix() for bucket in discovered.values() for path in bucket}
        if issues or paths != {record[field] for record in records.values()} or len(paths) != len(records):
            raise ValueError("원본 JPEG·JSON 목록이 기존 분석과 일치하지 않습니다.")
    for path, record in records.items():
        label = duplicates.read_label(root / record["json"], root / path, issues)
        if (issues or any(label.get(field) != record[field] for field in ("date", "city_id", "image_id", "state"))
                or duplicates.file_sha256(root / path) != record["file_sha256"]):
            raise ValueError(f"원본의 해시 또는 라벨 정보가 기존 분석과 일치하지 않습니다: {path}")

    _, prior, legacy = metadata.load_prior_results(prior_dir, analysis / "phash_user_review.csv", list(records.values()))
    source = metadata.read_pair_csv(analysis / "metadata_review" / "metadata_candidates.csv")
    expected, _ = metadata.metadata_candidates(list(records.values()), prior, legacy)
    computed = {metadata.pair_key(pair["a"]["image"], pair["b"]["image"]): pair for pair in expected}
    fields = ("pair_id", "phash_distance", "prior_phash_pair_id", "date", "city_id")
    if source.keys() != computed.keys() or any(
        row[field] != ("" if computed[key][field] is None else str(computed[key][field]))
        for key, row in source.items() for field in fields
    ):
        raise ValueError("기존 메타데이터 후보의 번호, 거리 또는 원본 쌍이 일치하지 않습니다.")
    fresh = {key: row for key, row in source.items() if key not in prior}
    if fresh.keys() & legacy.keys():
        raise ValueError("신규 후보에 기존 사용자 검토 쌍이 중복되었습니다.")
    return root, records, prior, legacy, source, fresh


def write_csv(path, fields, rows):
    """자동 생성 CSV를 완성한 후 교체한다."""
    with duplicates.inspection.output_file(path, mode="w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_or_create_reviews(path, candidates, records):
    """검토 CSV를 처음 한 번만 만들고 이후에는 항상 읽기 전용으로 열어 검증한다."""
    if not path.exists():
        rows = []
        for pair in candidates.values():
            row = {field: pair[field] for field in ("pair_id", "a_image", "b_image")}
            row["pair_id"] = row["pair_id"].zfill(4)
            a, b = (records[row[f"{side}_image"]] for side in ("a", "b"))
            for side, image in (("a", a), ("b", b)):
                row[f"{side}_filename"] = Path(image["image"]).name
                row[f"{side}_file_sha256"] = image["file_sha256"]
            row.update(label_mismatch=YES if a["state"] != b["state"] else NO, review_result="미검토", final_confirmed=NO)
            rows.append(row)

        # 완성된 초기 표만 게시하며 다른 실행에서 먼저 만든 검토표를 보존한다.
        with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            write_csv(temporary, LEDGER_FIELDS, rows)
            try:
                path.hardlink_to(temporary)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)

    reviews = metadata.read_pair_csv(path)
    if reviews.keys() != candidates.keys():
        raise ValueError(f"{path.name}의 이미지 쌍 목록을 현재 후보와 대조하세요. 기존 기록을 보존했습니다.")
    for key, row in reviews.items():
        if any(row[f"{side}_file_sha256"] != records[row[f"{side}_image"]]["file_sha256"] for side in ("a", "b")):
            raise ValueError(f"검토 CSV의 SHA-256이 원본과 일치하지 않습니다: {key}")
        result, final = row["review_result"], row["final_confirmed"]
        if result not in REVIEW_RESULTS or final not in (YES, NO):
            raise ValueError(f"review_result 또는 final_confirmed 값을 확인하세요: {key}")
        if row["same_place_confirmed"] not in ("", NO, YES) or (row["same_place_confirmed"] == YES and result != SAME_FACILITY):
            raise ValueError(f"same_place_confirmed의 '예'는 동일 시설물 판정에 사용하세요: {key}")
        if final == YES and (result not in FINAL_RESULTS or not row["review_basis"].strip() or not row["reviewer"].strip()):
            raise ValueError(f"최종 확인에는 관계 판정, 판단 근거와 검토 주체가 필요합니다: {key}")
    return reviews


def make_queue(analysis, output, root, records, prior, legacy, source, fresh, reviews):
    """기존 PNG는 재사용하고 나머지 후보의 PNG를 만든 뒤 한 검토표에 모은다."""
    queue = []
    preview_dir = output / "review"
    if preview_dir.resolve() != preview_dir:
        raise ValueError(f"비교 이미지 경로를 확인하세요: {preview_dir}")
    preview_dir.mkdir(exist_ok=True)
    pending = list(fresh.items()) + list(prior.items())
    for key, pair in pending:
        new = key in fresh
        manual = reviews[key] if new else legacy.get(key, {})
        a, b = (records[pair[f"{side}_image"]] for side in ("a", "b"))
        different_states = a["state"] != b["state"]
        pair_id = pair["pair_id"].zfill(4)
        observation = manual.get("observation", "")
        result = manual.get("review_result", "미검토")
        final = manual["final_confirmed"] == YES if new else manual.get("review_status") == "검토 완료"
        basis = manual.get("review_basis", "")
        relation = result if final else ""
        if relation == OTHER_FACILITY and "서로 다른 장소로 판단함" in basis:
            relation = DIFFERENT
        status = "검토 완료" if final else "미검토"
        if not final and (observation or result != "미검토"):
            status = "최종 확인 전"

        preview = ""
        if pair["preview"]:
            folder = analysis / ("metadata_review" if new else "threshold_16")
            path = (folder / pair["preview"]).resolve(strict=True)
            if not path.is_relative_to(folder / "review") or not path.is_file():
                raise ValueError(f"기존 비교 이미지 경로를 확인하세요: {path}")
            preview = path.relative_to(analysis).as_posix()
        elif new:
            suffix = stable_id("", key, records).lstrip("-")
            path = preview_dir / f"metadata_pair_{pair_id}_{suffix}.png"
            if path.resolve() != path:
                raise ValueError(f"생성할 비교 이미지 경로를 확인하세요: {path}")
            if not path.is_file():
                duplicates.save_preview(path, root, {
                    "a": a, "b": b, "pair_id": int(pair_id), "different_states": different_states,
                    "phash_distance": int(pair["phash_distance"]),
                }, title_prefix="Metadata candidate")
            preview = path.relative_to(analysis).as_posix()

        row = {
            "review_id": f"{'M' if new else 'P'}{pair_id}", "source": "메타데이터" if new else "pHash",
            "metadata_pair_id": source[key]["pair_id"].zfill(4) if key in source else "",
            "phash_pair_id": "" if new else pair_id, "priority": "우선" if observation and not final else ("일반" if new else "기존 검토"),
            "review_status": status, "phash_distance": pair["phash_distance"],
            "label_mismatch": YES if different_states else NO, "filename_number_gap": pair.get("filename_number_gap", ""),
            "image_id_gap": pair.get("image_id_gap", ""), "candidate_reason": pair.get("candidate_reason", "기존 pHash 후보"),
            "preview": preview, "observation": observation, "observation_source": manual.get("observation_source", ""),
            "label_review_note": manual.get("label_review_note", ""), "review_result": result, "review_basis": basis,
            "reviewer": manual.get("reviewer", ""), "final_confirmed": YES if final else NO,
            "same_place_confirmed": manual.get("same_place_confirmed", ""), "relation_for_group": relation,
        }
        for side, image in (("a", a), ("b", b)):
            row[f"{side}_filename"] = Path(image["image"]).name
            row.update({f"{side}_{field}": image[field] for field in ("image", "json", "state", "state_name", "date", "city_id", "image_id")})
        queue.append(row)
    return sorted(queue, key=lambda row: (row["source"] != "메타데이터", row["priority"] != "우선", row["review_id"]))


def stable_id(prefix, images, records):
    """정렬된 원본 경로와 SHA-256으로 같은 구성원에 같은 ID를 부여한다."""
    members = [(image, records[image]["file_sha256"]) for image in sorted(images)]
    digest = hashlib.sha256(json.dumps(members, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}" if prefix else digest


def components(edges, singletons=()):
    """확인된 연결의 구성원 집합을 반환하고 필요하면 단일 이미지도 포함한다."""
    neighbors = {image: set() for image in singletons}
    for a, b in edges:
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)
    visited = set()
    for image in sorted(neighbors):
        if image in visited:
            continue
        pending = {image}
        members = set()
        while pending:
            current = pending.pop()
            members.add(current)
            pending.update(neighbors[current] - members)
        visited.update(members)
        yield members


def confirmed_groups(records, evidence, positive, negative, prefix):
    """연결 그룹 내부에 반대 판정이 있으면 통째로 확인 대상으로 남긴다."""
    matches = {key: row for key, row in evidence.items() if positive(row)}
    rejects = {key: row for key, row in evidence.items() if row["relation_for_group"] in negative}
    groups, conflicts = [], []
    for members in components(matches):
        supporting = [row for key, row in matches.items() if set(key) <= members]
        opposing = [row for key, row in rejects.items() if set(key) <= members]
        group = {"group_id": stable_id(prefix, members, records), "images": sorted(members),
                 "states": sorted({records[image]["state_name"] for image in members}), "supporting_reviews": supporting}
        if opposing:
            conflicts.append({**group, "opposing_reviews": opposing, "status": "판정 모순 확인 필요"})
        else:
            groups.append(group)
    return groups, conflicts


def build_groups(records, queue):
    """최종 확인한 관계로 두 종류의 그룹과 잠정 분할 단위를 계산한다."""
    evidence, unresolved = {}, set()
    fields = ("review_id", "source", "a_image", "b_image", "review_result", "review_basis", "reviewer",
              "final_confirmed", "same_place_confirmed", "relation_for_group")
    for row in sorted(queue, key=lambda row: row["review_id"]):
        key = metadata.pair_key(row["a_image"], row["b_image"])
        if row["final_confirmed"] == YES and row["relation_for_group"]:
            evidence[key] = {field: row[field] for field in fields}
        else:
            unresolved.update(key)

    facility, facility_conflicts = confirmed_groups(records, evidence, lambda row: row["relation_for_group"] == SAME_FACILITY,
                                                     (SAME_PLACE, DIFFERENT, OTHER_FACILITY), "FAC")
    location, location_conflicts = confirmed_groups(records, evidence, lambda row: (
        row["relation_for_group"] == SAME_PLACE or (row["relation_for_group"] == SAME_FACILITY and row["same_place_confirmed"] == YES)
    ), (DIFFERENT,), "LOC")
    facility_ids = {image: group["group_id"] for group in facility for image in group["images"]}
    location_ids = {image: group["group_id"] for group in location for image in group["images"]}
    conflicted = {image for group in facility_conflicts + location_conflicts for image in group["images"]}
    links = [(group["images"][0], other) for group in facility + location for other in group["images"][1:]]
    units = list(components(links, records))
    mapping = []
    for members in units:
        status = "잠정"
        if members & conflicted:
            status = "판정 모순 확인 필요"
        elif members & unresolved:
            status = "후보 검토 필요"
        split_id = stable_id("UNIT", members, records)
        for image in sorted(members):
            mapping.append({"image": image, "state_name": records[image]["state_name"], "facility_group_id": facility_ids.get(image, ""),
                            "location_group_id": location_ids.get(image, ""), "provisional_unit_id": split_id, "status": status})
    return sorted(mapping, key=lambda row: row["image"]), {
        "facility_groups": facility, "location_groups": location,
        "facility_conflicts": facility_conflicts, "location_conflicts": location_conflicts,
        "facility_count": len(facility), "location_count": len(location),
        "provisional_facility_units_including_singletons": len(facility) + len(records) - len(facility_ids),
        "provisional_location_units_including_singletons": len(location) + len(records) - len(location_ids),
        "provisional_split_units_including_singletons": len(units), "images_with_unresolved_pairs": len(unresolved),
        "mixed_state_facility_groups": [group["group_id"] for group in facility if len(group["states"]) > 1],
    }


def write_results(output, analysis, root, queue, mapping, groups):
    """검토표, 라벨 비교, 이미지별 그룹 및 분할 준비 요약을 저장한다."""
    write_csv(output / "review_queue.csv", QUEUE_FIELDS, queue)
    write_csv(output / "image_groups.csv", tuple(mapping[0]), mapping)
    label_fields = ("review_id", "source", "review_status", "review_result", "observation", "label_review_note", "preview",
                    "a_json", "a_state", "a_state_name", "a_is_defect", "b_json", "b_state", "b_state_name", "b_is_defect")
    state_rows = []
    for pair in queue:
        if pair["label_mismatch"] != YES:
            continue
        row = {field: pair[field] for field in label_fields if not field.endswith("_is_defect")}
        for side in ("a", "b"):
            raw = json.loads((root / pair[f"{side}_json"]).read_text(encoding="utf-8-sig"))
            row[f"{side}_is_defect"] = ", ".join(str(entry.get("is_defect", "")) for entry in raw["annotations"])
        state_rows.append(row)
    write_csv(output / "state_label_review.csv", label_fields, state_rows)
    counts = {}
    for source in ("메타데이터", "pHash"):
        statuses = Counter(row["review_status"] for row in queue if row["source"] == source)
        counts[source] = {"completed": statuses["검토 완료"], "pending": statuses.total() - statuses["검토 완료"],
                          "by_status": dict(statuses)}
    report = {
        "data_root": str(root), "analysis_root": str(analysis),
        "path_reference": "원본 image·json은 data_root, 비교 preview는 analysis_root 기준입니다.",
        "review_files": {"editable_metadata_reviews": "metadata_user_reviews.csv", "prior_phash_reviews": "../phash_user_review.csv",
                         "queue": "review_queue.csv", "image_groups": "image_groups.csv", "state_labels": "state_label_review.csv"},
        "review_counts": counts, "priority_review_ids": [row["review_id"] for row in queue if row["priority"] == "우선"],
        "label_mismatch_review_ids": [row["review_id"] for row in state_rows], **groups,
        "review_workflow": {
            "review_result_options": REVIEW_RESULTS,
            "final_confirmed": "판정, 근거, 검토 주체를 기입한 뒤 final_confirmed에 예를 넣으면 그룹 근거가 됩니다.",
            "same_place_confirmed": "동일 시설물 판정에서 장소도 확인했으면 same_place_confirmed에 예를 넣습니다. 같은 촬영 장소의 다른 시설물 판정은 장소 관계를 포함합니다.",
            "group_id": "구성원의 상대 경로와 원본 SHA-256으로 계산합니다. 같은 구성원을 다시 읽으면 같은 ID가 나오며 구성원이 달라지면 ID도 달라집니다.",
            "split_readiness": "미확정 후보와 모순을 검토하고, 확인한 시설물·장소 관계로 연결된 단위를 한 분할에 함께 배정하세요. 잠정 단위 수는 실제 독립 그룹 수를 보증하지 않습니다.",
        },
    }
    with duplicates.inspection.output_file(output / "group_report.json", mode="w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return counts


def main(argv=None):
    """기존 분석과 검토 기록에서 그룹 준비 자료를 만들며 데이터 분할은 수행하지 않는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=metadata.ANALYSIS_DIR)
    args = parser.parse_args(argv)
    analysis = args.analysis_dir.resolve()
    output = analysis / "group_review"
    try:
        root, records, prior, legacy, source, fresh = load_sources(analysis)
        if output.resolve() != output:
            raise ValueError(f"그룹 검토 결과 경로를 확인하세요: {output}")
        output.mkdir(exist_ok=True)
        reviews = load_or_create_reviews(output / "metadata_user_reviews.csv", fresh, records)
        queue = make_queue(analysis, output, root, records, prior, legacy, source, fresh, reviews)
        mapping, groups = build_groups(records, queue)
        counts = write_results(output, analysis, root, queue, mapping, groups)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"그룹 검토 준비 오류: {error}\n")
    print(f"메타데이터 미완료 {counts['메타데이터']['pending']}쌍, pHash 검토 완료 {counts['pHash']['completed']}쌍")
    print(f"시설물 그룹 {groups['facility_count']}개, 장소 그룹 {groups['location_count']}개, 결과 경로 {output}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
