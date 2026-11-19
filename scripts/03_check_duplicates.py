"""AI-Hub 볼라드 JPEG의 완전 중복과 시각적 유사 후보를 검사한다."""

import argparse
import csv
import hashlib
import json
import sys
import textwrap
from collections import Counter, defaultdict
from importlib import import_module
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


# 기존 CLI의 파일 탐색, 상태 코드, 결과 저장 규칙을 재사용한다.
inspection = import_module("02_visualize_dataset")
RAW_DIR = inspection.RAW_DIR
OUTPUT_DIR = inspection.PROJECT_DIR / "data" / "processed" / "duplicate_analysis"
DEFAULT_THRESHOLD = 8
DEFAULT_MAX_PREVIEWS = 40
PHASH_BITS = 64
PAIR_FIELDS = (
    "pair_id", "candidate_group_id", "phash_distance", "phash_within_threshold", "same_file_sha256",
    "same_display_pixels", "different_states", "same_date", "same_city_id", "review_reason", "preview",
)
IMAGE_FIELDS = ("image", "json", "state", "state_name", "date", "city_id", "image_id")


def file_sha256(path):
    """JPEG 파일 전체의 SHA-256을 계산한다."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pixel_sha256(rgb):
    """표시 너비·높이와 RGB 픽셀을 해싱해 크기가 다른 이미지를 구분한다."""
    digest = hashlib.sha256()
    digest.update(rgb.width.to_bytes(8, "big"))
    digest.update(rgb.height.to_bytes(8, "big"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def perceptual_hash(display):
    """표시 이미지를 32×32 회색조로 축소하고 저주파 DCT로 64비트 pHash를 만든다."""
    with display.convert("L") as grayscale:
        with grayscale.resize((32, 32), Image.Resampling.LANCZOS) as thumbnail:
            coefficients = cv2.dct(np.asarray(thumbnail, dtype=np.float64))[:8, :8]
    bits = coefficients > np.median(coefficients)
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def read_label(path, image, issues):
    """대응 JSON의 상태와 촬영 참고 필드를 읽는다. 파일명 불일치 시 연결을 보류한다."""
    try:
        with path.open(encoding="utf-8-sig") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError("최상위 JSON 객체가 필요합니다.")
    except (OSError, ValueError, RecursionError) as error:
        inspection.record_issue(issues, "JSON_READ", path, str(error), level="error")
        return {}

    info = data.get("info")
    if not isinstance(info, dict) or info.get("filename") != image.name:
        inspection.record_issue(issues, "IMAGE_FILENAME_MISMATCH", path,
                                f"info.filename과 실제 JPEG 이름({image.name})을 확인하세요.", level="error")
        return {}

    description = data.get("description")
    state = description.get("state") if isinstance(description, dict) else None
    metadata = {field: info[field] for field in ("date", "city_id", "image_id")
                if isinstance(info.get(field), str) and info[field].strip()}
    if not isinstance(state, str) or state not in inspection.STATES:
        inspection.record_issue(issues, "STATE_VALUE", path, f"description.state={state!r}", level="error")
        return metadata
    return {**metadata, "state": state, "state_name": inspection.STATES[state][0]}


def inspect_images(root, issues):
    """발견한 모든 JPEG를 검사하고 유일하게 대응되는 JSON의 상태를 연결한다."""
    images, labels = inspection.discover_files(root, issues)
    records = []
    matched = 0
    if not images:
        inspection.record_issue(issues, "NO_IMAGES", root, "검사할 JPEG가 없습니다.", level="error")
    for stem in sorted(images.keys() | labels.keys()):
        image_paths, label_paths = images.get(stem, []), labels.get(stem, [])
        label_path = None
        if len(image_paths) == len(label_paths) == 1:
            label_path = label_paths[0]
            matched += 1
        else:
            inspection.record_issue(issues, "PAIR_NOT_UNIQUE", (image_paths or label_paths)[0],
                                    f"stem={stem!r}, JPEG {len(image_paths)}개, JSON {len(label_paths)}개", level="error")
        for path in image_paths:
            record = dict.fromkeys(IMAGE_FIELDS)
            record["image"] = path.relative_to(root).as_posix()
            if label_path is not None:
                record["json"] = label_path.relative_to(root).as_posix()
                record.update(read_label(label_path, path, issues))
            record.update(file_sha256=None, display_pixel_sha256=None, phash=None,
                          exif_orientation=None, display_size=None)
            try:
                record["file_sha256"] = file_sha256(path)
            except OSError as error:
                inspection.record_issue(issues, "FILE_HASH", path, str(error), level="error")
            try:
                with Image.open(path) as stored:
                    if stored.format != "JPEG":
                        raise ValueError(f"JPEG 확장자와 실제 형식이 다름 ({stored.format})")
                    orientation = stored.getexif().get(274)
                    if orientation is not None and (type(orientation) is not int or not 1 <= orientation <= 8):
                        raise ValueError(f"지원 범위를 벗어난 EXIF 방향 {orientation!r}")
                    record["exif_orientation"] = orientation
                    with ImageOps.exif_transpose(stored) as display, display.convert("RGB") as rgb:
                        record["display_size"] = list(rgb.size)
                        record["display_pixel_sha256"] = pixel_sha256(rgb)
                        record["phash"] = f"{perceptual_hash(rgb):016x}"
            except (OSError, ValueError, SyntaxError, TypeError, Image.DecompressionBombError, cv2.error) as error:
                inspection.record_issue(issues, "IMAGE_READ", path, str(error), level="error")
            records.append(record)
    counts = {"images_found": len(records), "json_files_found": sum(map(len, labels.values())), "matched_stems": matched}
    return sorted(records, key=lambda image: image["image"]), counts


def equal_if_known(first, second, field):
    """두 값이 있을 때만 일치 여부를 계산한다."""
    left, right = first[field], second[field]
    return None if left is None or right is None else left == right


def exact_duplicates(records, field):
    """해시별 완전 중복 그룹과 두 상태가 섞인 그룹의 수를 집계한다."""
    by_hash = defaultdict(list)
    for image in records:
        if image[field] is not None:
            by_hash[image[field]].append(image)
    groups = []
    for digest, members in sorted(by_hash.items()):
        if len(members) > 1:
            states = {image["state"] for image in members if image["state"] is not None}
            groups.append({"sha256": digest, "different_states": len(states) > 1,
                           "images": [{field: image[field] for field in IMAGE_FIELDS} for image in members]})
    return {"group_count": len(groups), "image_count": sum(len(group["images"]) for group in groups),
            "groups_with_different_states": sum(group["different_states"] for group in groups), "groups": groups}


def compare_pairs(records, threshold):
    """고유한 쌍의 해밍 거리와 완전 중복 여부를 계산해 검토 우선순위로 정렬한다."""
    hashes = {image["image"]: int(image["phash"], 16) for image in records if image["phash"] is not None}
    distances = Counter()
    pairs = []
    for first, second in combinations(records, 2):
        left, right = hashes.get(first["image"]), hashes.get(second["image"])
        distance = None
        if left is not None and right is not None:
            distance = (left ^ right).bit_count()
            distances[distance] += 1
        matches_phash = None if distance is None else distance <= threshold
        same_file = equal_if_known(first, second, "file_sha256")
        same_pixels = equal_if_known(first, second, "display_pixel_sha256")
        if not (matches_phash or same_file or same_pixels):
            continue
        same_state = equal_if_known(first, second, "state")
        different_states = None if same_state is None else not same_state
        if same_file:
            reason = "파일 SHA-256 일치"
        elif same_pixels:
            reason = "EXIF 적용 후 크기와 RGB 픽셀 일치"
        else:
            reason = "pHash 거리 임계값 이내"
        if different_states:
            reason += " / 양호·불량 상태가 다름"
        pairs.append({"a": first, "b": second, "phash_distance": distance, "phash_within_threshold": matches_phash,
                      "same_file_sha256": same_file, "same_display_pixels": same_pixels, "different_states": different_states,
                      "same_date": equal_if_known(first, second, "date"),
                      "same_city_id": equal_if_known(first, second, "city_id"), "review_reason": reason})

    pairs.sort(key=lambda pair: (
        not (pair["same_file_sha256"] or pair["same_display_pixels"]), pair["different_states"] is not True,
        pair["phash_distance"] if pair["phash_distance"] is not None else PHASH_BITS + 1,
        pair["a"]["image"], pair["b"]["image"],
    ))
    for index, pair in enumerate(pairs, start=1):
        pair.update(pair_id=index, preview=None)
    return pairs, distances


def candidate_groups(pairs):
    """직접 연결된 후보들을 한데 모으고 각 연결의 후보 쌍 ID를 남긴다."""
    neighbors = defaultdict(set)
    for pair in pairs:
        a, b = pair["a"]["image"], pair["b"]["image"]
        neighbors[a].add(b)
        neighbors[b].add(a)

    group_ids = {}
    groups = []
    for image in sorted(neighbors):
        if image in group_ids:
            continue
        group = {"candidate_group_id": len(groups) + 1, "images": [], "pair_ids": []}
        pending = {image}
        while pending:
            member = pending.pop()
            group_ids[member] = group["candidate_group_id"]
            group["images"].append(member)
            pending.update(neighbors[member] - group_ids.keys())
        group["images"].sort()
        groups.append(group)
    for pair in pairs:
        group_id = group_ids[pair["a"]["image"]]
        pair["candidate_group_id"] = group_id
        groups[group_id - 1]["pair_ids"].append(pair["pair_id"])
    return groups


def save_preview(path, root, pair, *, title_prefix="Pair"):
    """두 EXIF 표시 이미지를 비율을 유지해 배치하고 파일명·상태·거리를 표시한다."""
    margin, panel_width, panel_height = 24, 600, 620
    ink, secondary = "#17233a", "#4f6075"
    with Image.new("RGB", (2 * panel_width + 3 * margin, panel_height + 180), "#eef2f6") as canvas:
        draw = ImageDraw.Draw(canvas)
        title, body = ImageFont.load_default(size=24), ImageFont.load_default(size=17)
        distance = pair["phash_distance"]
        distance_text = "unavailable" if distance is None else f"{distance} / {PHASH_BITS}"
        heading = f"{title_prefix} {pair['pair_id']:04d}    |    pHash distance {distance_text}"
        draw.text((margin, 18), heading, font=title, fill=ink)
        if pair["different_states"]:
            draw.text((canvas.width - margin, 25), "Different state labels", anchor="ra", font=body, fill="#9b4511")
        for index, side in enumerate(("a", "b")):
            image = pair[side]
            x, y = margin + index * (panel_width + margin), 60
            draw.rectangle((x, y, x + panel_width - 1, y + panel_height - 1), fill="#dfe6ee")
            with Image.open(root / image["image"]) as stored:
                with ImageOps.exif_transpose(stored) as display, display.convert("RGB") as rgb:
                    rgb.thumbnail((panel_width, panel_height), Image.Resampling.LANCZOS)
                    canvas.paste(rgb, (x + (panel_width - rgb.width) // 2, y + (panel_height - rgb.height) // 2))
            state = inspection.STATES[image["state"]][2] if image["state"] is not None else "Unknown"
            date, city = image["date"] or "-", image["city_id"] or "-"
            draw.text((x, y + panel_height + 12), f"{side.upper()}    {state}    |    Date {date}    |    City code {city}",
                      font=body, fill=secondary)
            filename = textwrap.fill(Path(image["image"]).name, width=56)
            draw.multiline_text((x, y + panel_height + 38), filename, font=body, fill=ink)
        with inspection.output_file(path, mode="wb") as stream:
            canvas.save(stream, format="PNG")


def write_results(output, root, records, counts, pairs, distances, args, issues):
    """비교 이미지, 후보 CSV와 판정 조건을 포함한 JSON 보고서를 저장한다."""
    groups = candidate_groups(pairs)
    selected = [pair for pair in pairs if pair["a"]["display_size"] and pair["b"]["display_size"]][:args.max_previews]
    output.mkdir(parents=True, exist_ok=True)
    if selected:
        # 조건을 바꿔 재실행하더라도 이전 보고서가 가리키는 비교 이미지는 덮어쓰지 않는다.
        signature = hashlib.sha256(json.dumps(selected, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
        preview_dir = output / "review" / signature
        if preview_dir.resolve() != preview_dir:
            raise OSError(f"비교 이미지 경로에 다른 위치로 연결되는 디렉터리가 있습니다: {preview_dir}")
        preview_dir.mkdir(parents=True, exist_ok=True)
        for pair in selected:
            path = preview_dir / f"pair_{pair['pair_id']:04d}.png"
            try:
                save_preview(path, root, pair)
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                inspection.record_issue(issues, "PREVIEW_WRITE", path, str(error), level="error")
                continue
            pair["preview"] = path.relative_to(output).as_posix()

    pair_rows = []
    side_fields = tuple(f"{side}_{field}" for side in ("a", "b") for field in IMAGE_FIELDS)
    for pair in pairs:
        row = {field: pair[field] for field in PAIR_FIELDS}
        row.update({f"{side}_{field}": pair[side][field] for side in ("a", "b") for field in IMAGE_FIELDS})
        pair_rows.append(row)
    with inspection.output_file(output / "similar_pairs.csv", mode="w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(*PAIR_FIELDS, *side_fields))
        writer.writeheader()
        writer.writerows(pair_rows)

    phash_pairs = [row for row in pair_rows if row["phash_within_threshold"]]
    for issue in issues:
        path = Path(issue["path"])
        if path.is_relative_to(root):
            issue["path"] = path.relative_to(root).as_posix()
    report = {
        "status": "complete" if not issues else "incomplete",
        "data_root": str(root),
        "path_reference": "image와 json은 data_root 기준, preview는 보고서 디렉터리 기준입니다.",
        "counts": {**counts, "file_sha256_computed": sum(image["file_sha256"] is not None for image in records),
                   "display_pixel_sha256_computed": sum(image["display_pixel_sha256"] is not None for image in records),
                   "phash_computed": sum(image["phash"] is not None for image in records),
                   "states": dict(sorted(Counter(image["state_name"] or "미확인" for image in records).items()))},
        "exact_file_duplicates": exact_duplicates(records, "file_sha256"),
        "exact_display_pixel_duplicates": exact_duplicates(records, "display_pixel_sha256"),
        "phash": {
            "method": "EXIF 표시 RGB → 회색조 → Lanczos 32×32 → 2D DCT의 좌상단 8×8을 중앙값과 비교 (DC 포함)",
            "bits": PHASH_BITS,
            "threshold": args.threshold,
            "threshold_selection": "기본값 8은 64비트 중 최대 8비트가 다른 쌍을 우선 검토하는 시작점입니다. 이 데이터로 보정하지 않았습니다.",
            "possible_pairs": len(records) * (len(records) - 1) // 2,
            "compared_pairs": distances.total(),
            "distance_distribution": {str(distance): distances[distance] for distance in range(PHASH_BITS + 1)},
            "candidate_pair_count": len(phash_pairs),
            "candidate_distance_distribution": dict(sorted(Counter(row["phash_distance"] for row in phash_pairs).items())),
            "pairs_with_different_states": sum(row["different_states"] is True for row in phash_pairs),
        },
        "review": {
            "priority": "완전 중복 우선, 양호·불량 교차 쌍 우선, pHash 거리 오름차순, 경로순",
            "candidate_pair_count": len(pair_rows),
            "max_previews": args.max_previews,
            "preview_count": sum(bool(row["preview"]) for row in pair_rows),
            "pairs_with_different_states": [row["pair_id"] for row in pair_rows if row["different_states"]],
            "candidate_groups": groups,
            "manual_review_status": "pending" if pairs else "no_candidates",
            "pairs_file": "similar_pairs.csv",
        },
        "limitations": [
            "후보는 사람이 배경, 시설물 특징과 촬영 단서를 비교해야 합니다. 다른 장소의 비슷한 볼라드도 후보에 포함될 수 있습니다.",
            "같은 시설물을 다른 각도, 거리, 조명에서 촬영한 경우 pHash 임계값을 벗어날 수 있습니다.",
            "info.date, info.city_id, info.image_id와 파일명은 검토용 단서입니다. 날짜·지역 코드가 같아도 촬영 장소나 시설물은 확인되지 않습니다.",
            "양호·불량 교차 후보는 상태 정보 확인 대상입니다. 라벨 오류 여부는 미판정입니다.",
            "연결된 후보 그룹에는 직접 비교에서 임계값을 벗어난 쌍도 포함될 수 있습니다. 같은 시설물이나 촬영 장면은 사람 검토를 거쳐 확정합니다.",
            "이 검사는 분할 전 후보 탐색입니다. 후보 유무에 관계없이 실제 데이터 누수 여부는 분할 후 확인해야 합니다.",
        ],
        "issues": issues,
        "images": records,
    }
    with inspection.output_file(output / "duplicate_report.json", mode="w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main(argv=None):
    """입력과 출력 경로를 확인한 후 전체 중복 검사를 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=RAW_DIR, help="입력 루트 (기본: data/raw)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="결과 루트 (기본: data/processed/duplicate_analysis)")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help="최대 pHash 해밍 거리 (0~64, 기본: 8)")
    parser.add_argument("--max-previews", type=int, default=DEFAULT_MAX_PREVIEWS, help="비교 이미지 상한 (기본: 40)")
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= PHASH_BITS:
        parser.error(f"--threshold는 0~{PHASH_BITS}여야 합니다.")
    if args.max_previews < 1:
        parser.error("--max-previews는 1 이상이어야 합니다.")
    root, output = args.data_dir.resolve(), args.output_dir.resolve()
    if not root.is_dir():
        parser.error(f"입력 디렉터리가 없습니다: {root}")
    if any(output.is_relative_to(path) for path in (root, RAW_DIR.resolve())):
        parser.error("입력·원본 디렉터리 밖에만 결과를 저장할 수 있습니다.")

    issues = []
    records, counts = inspect_images(root, issues)
    pairs, distances = compare_pairs(records, args.threshold)
    try:
        report = write_results(output, root, records, counts, pairs, distances, args, issues)
    except OSError as error:
        parser.exit(1, f"결과 저장 오류: {error}\n")
    print(f"JPEG {counts['images_found']}장, SHA-256 {report['counts']['file_sha256_computed']}장, "
          f"pHash {report['counts']['phash_computed']}장, 비교 {distances.total()}쌍")
    print(f"파일 중복 {report['exact_file_duplicates']['group_count']}그룹, "
          f"표시 픽셀 중복 {report['exact_display_pixel_duplicates']['group_count']}그룹, "
          f"pHash 후보 {report['phash']['candidate_pair_count']}쌍, 비교 이미지 {report['review']['preview_count']}장")
    print(f"결과 디렉터리: {output}")
    if issues:
        print(f"확인이 필요한 검사 오류 {len(issues)}건이 보고서에 기록되었습니다.", file=sys.stderr)
    return int(bool(issues))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
