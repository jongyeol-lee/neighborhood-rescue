"""AI-Hub 볼라드의 EXIF 표시 이미지에 두 BBOX 좌표 해석을 그려 비교하고 EDA를 저장한다."""

import argparse
import csv
import json
import math
import random
import statistics
import sys
import textwrap
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "inspection"
STATES = {"0": ("양호", "정상", "Good"), "1": ("불량", "불량", "Defective")}
ORDERS = {"TLWH": "top, left, width, height", "LTWH": "left, top, width, height"}
REVIEW_FILES = ("1_01_0_5_4_1_20211001_0000572131.json", "1_01_1_5_6_3_20210906_0000457520.json")
SAMPLE_SEED = 42
SAMPLES_PER_GROUP = 3
BOX_FIELDS = (
    "annotation_index", "bbox_index", "is_defect", "state_check", "bbox_original", "bbox_width", "bbox_height",
    "tlwh_xyxy", "tlwh_bounds", "tlwh_area_ratio", "ltwh_xyxy", "ltwh_bounds", "ltwh_area_ratio",
)


def record_issue(issues, code, path, detail, *, level="review"):
    """파일이나 필드의 확인 사항을 결과에 추가한다."""
    issues.append({"level": level, "code": code, "path": str(path), "detail": detail})


def discover_files(root, issues):
    """대소문자를 무시한 stem으로 JPEG와 JSON 경로를 각각 묶는다."""
    images, labels = defaultdict(list), defaultdict(list)

    def on_error(error):
        record_issue(issues, "DIRECTORY_READ", error.filename or root, str(error), level="error")

    for directory, dirs, filenames in root.walk(on_error=on_error):
        dirs.sort()
        for name in sorted(filenames):
            path = directory / name
            extension = path.suffix.lower()
            if extension in {".jpg", ".jpeg"}:
                images[path.stem.casefold()].append(path)
            elif extension == ".json":
                labels[path.stem.casefold()].append(path)
    return images, labels


def size_text(size):
    """크기를 가로x세로 문자열로 돌려준다."""
    return f"{size[0]}x{size[1]}" if size else "unknown"


def inspect_image(path, issues):
    """JPEG를 디코딩하고 EXIF 적용 전후 크기와 방향을 반환한다."""
    try:
        with Image.open(path) as image:
            if image.format != "JPEG":
                raise ValueError(f"JPEG 확장자와 실제 형식이 다름 ({image.format})")
            orientation = image.getexif().get(274)
            with ImageOps.exif_transpose(image) as display:
                metadata = {"orientation": orientation, "jpeg_size": image.size, "display_size": display.size}
    except (OSError, ValueError, SyntaxError, TypeError, Image.DecompressionBombError) as error:
        record_issue(issues, "IMAGE_READ", path, str(error), level="error")
        return {}
    if orientation is not None and (type(orientation) is not int or not 1 <= orientation <= 8):
        record_issue(issues, "EXIF_ORIENTATION", path, f"지원 범위를 벗어난 방향 값 {orientation!r}")
    return metadata


def bbox_to_xyxy(box, order):
    """TLWH 또는 LTWH 배열을 (왼쪽, 위쪽, 오른쪽, 아래쪽) 경계 좌표로 해석한다."""
    if not isinstance(box, list) or len(box) != 4 or any(type(value) not in (int, float) for value in box):
        raise ValueError("숫자 4개의 BBOX 배열이 필요합니다.")
    first, second, width, height = box
    if order == "TLWH":
        left, top = second, first
    elif order == "LTWH":
        left, top = first, second
    else:
        raise ValueError(f"알 수 없는 BBOX 해석 방식 {order!r}")
    try:
        edges = (left, top, left + width, top + height)
        if all(math.isfinite(value) for value in (*box, *edges)) and width > 0 and height > 0:
            return edges
    except OverflowError:
        pass
    raise ValueError("유한한 값과 양의 폭·높이가 필요합니다.")


def check_bbox(box, display_size):
    """원본 배열과 두 해석의 경계, 범위 검사, 범위 안인 경우의 면적 비율을 반환한다."""
    row = {"bbox_original": json.dumps(box, ensure_ascii=False)}
    for order in ORDERS:
        prefix = order.lower()
        try:
            edges = bbox_to_xyxy(box, order)
        except ValueError:
            row[f"{prefix}_bounds"] = "invalid"
            continue
        row[f"{prefix}_xyxy"] = edges
        width, height = box[2:]
        row.update(bbox_width=width, bbox_height=height)
        if display_size is None:
            row[f"{prefix}_bounds"] = "unavailable"
            continue
        image_width, image_height = display_size
        left, top, right, bottom = edges
        # 오른쪽·아래쪽 경계는 픽셀 인덱스가 아닌 연속 좌표이므로 이미지 폭·높이와 같을 수 있다.
        inside = 0 <= left < right <= image_width and 0 <= top < bottom <= image_height
        row[f"{prefix}_bounds"] = "inside" if inside else "outside"
        if inside:
            row[f"{prefix}_area_ratio"] = width * height / (image_width * image_height)
    return row


def inspect_label(path, image_path, display_size, issues):
    """JSON의 크기·상태·어노테이션을 읽고 BBOX별 검사 행을 반환한다."""
    try:
        with path.open(encoding="utf-8-sig") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError("최상위 JSON 객체가 필요합니다.")
    except (OSError, ValueError, RecursionError) as error:
        record_issue(issues, "JSON_READ", path, str(error), level="error")
        return {}

    info = data.get("info")
    json_size = None
    if isinstance(info, dict):
        size = (info.get("width"), info.get("height"))
        if all(type(value) is int and value > 0 for value in size):
            json_size = size
        if image_path is not None and info.get("filename") != image_path.name:
            record_issue(issues, "IMAGE_FILENAME_MISMATCH", path, f"JSON {info.get('filename')!r}, JPEG {image_path.name!r}")
    if json_size is None:
        record_issue(issues, "JSON_IMAGE_SIZE", path, "info.width와 info.height에 양의 정수가 필요합니다.", level="error")

    description = data.get("description")
    state = description.get("state") if isinstance(description, dict) else None
    if not isinstance(state, str) or state not in STATES:
        record_issue(issues, "STATE_VALUE", path, f"description.state 값 확인 필요 ({state!r})", level="error")
        state = None

    annotations = data.get("annotations")
    rows = []
    if not isinstance(annotations, list):
        record_issue(issues, "ANNOTATIONS_TYPE", path, "annotations 배열이 필요합니다.", level="error")
        return {"json_size": json_size, "state": state, "rows": rows}
    if not annotations:
        record_issue(issues, "ANNOTATIONS_EMPTY", path, "annotations가 비어 있습니다.")
    for index, annotation in enumerate(annotations):
        location = f"annotations[{index}]"
        if not isinstance(annotation, dict):
            record_issue(issues, "ANNOTATION_TYPE", path, f"{location}에 객체가 필요합니다.", level="error")
            rows.append({"annotation_index": index, "state_check": "unavailable"})
            continue
        defect = annotation.get("is_defect")
        if state is None or defect not in ("정상", "불량"):
            state_check = "unavailable"
            record_issue(issues, "STATE_CHECK_UNAVAILABLE", path, f"{location}.is_defect={defect!r}, description.state={state!r}")
        elif defect == STATES[state][1]:
            state_check = "match"
        else:
            state_check = "mismatch"
            record_issue(issues, "STATE_MISMATCH", path, f"{location}.is_defect={defect!r}, description.state={state!r}")
        annotation_row = {"annotation_index": index, "is_defect": defect, "state_check": state_check}
        if annotation.get("annotation_type") != "bbox":
            record_issue(issues, "GEOMETRY_UNSUPPORTED", path, f"{location}의 annotation_type={annotation.get('annotation_type')!r}")
            rows.append(annotation_row)
            continue
        boxes = annotation.get("annotation_info")
        if not isinstance(boxes, list) or not boxes:
            record_issue(issues, "BBOX_LIST", path, f"{location}.annotation_info에 비어 있지 않은 배열이 필요합니다.", level="error")
            rows.append(annotation_row)
            continue
        for bbox_index, box in enumerate(boxes):
            row = check_bbox(box, display_size)
            results = (row["tlwh_bounds"], row["ltwh_bounds"])
            detail = f"{location}.annotation_info[{bbox_index}]={row['bbox_original']}, TLWH={results[0]}, LTWH={results[1]}"
            if "invalid" in results:
                record_issue(issues, "BBOX_INVALID", path, detail, level="error")
            if "outside" in results:
                record_issue(issues, "BBOX_ORDER_REVIEW" if results[0] != results[1] else "BBOX_RANGE_REVIEW", path, detail)
            rows.append({**annotation_row, "bbox_index": bbox_index, **row})
    return {"json_size": json_size, "state": state, "annotation_count": len(annotations), "rows": rows}


def compare_sizes(reference, actual):
    """유효한 두 크기의 일치 여부를 문자열로 반환한다."""
    if reference is None or actual is None:
        return "unavailable"
    return "match" if reference == actual else "mismatch"


def inspect_dataset(root, issues):
    """일대일로 대응되는 파일의 크기를 대조하고, 나머지 파일도 개별 검사한다."""
    images, labels = discover_files(root, issues)
    counts = Counter(images=sum(map(len, images.values())), json_files=sum(map(len, labels.values())),
                     pairs=0, image_only=0, json_only=0, ambiguous_keys=0)
    if not counts["images"] or not counts["json_files"]:
        record_issue(issues, "DATASET_EMPTY", root, "JPEG와 JSON이 각각 1개 이상 필요합니다.", level="error")
    records = []
    for stem in sorted(images.keys() | labels.keys()):
        image_paths, label_paths = images.get(stem, []), labels.get(stem, [])
        if len(image_paths) == len(label_paths) == 1:
            counts["pairs"] += 1
            inputs = [(image_paths[0], label_paths[0], "matched")]
        else:
            if len(image_paths) > 1 or len(label_paths) > 1:
                counts["ambiguous_keys"] += 1
                status = "ambiguous"
            elif image_paths:
                counts["image_only"] += 1
                status = "image_only"
            else:
                counts["json_only"] += 1
                status = "json_only"
            for path in image_paths + label_paths:
                record_issue(issues, "PAIR_" + status.upper(), path, f"stem={stem!r}", level="error")
            inputs = [(path, None, status) for path in image_paths] + [(None, path, status) for path in label_paths]
        for image_path, label_path, status in inputs:
            image = inspect_image(image_path, issues) if image_path else {}
            display_size = image.get("display_size")
            jpeg_size = image.get("jpeg_size")
            label = inspect_label(label_path, image_path, display_size, issues) if label_path else {}
            json_size = label.get("json_size")
            jpeg_check = compare_sizes(json_size, jpeg_size)
            display_check = compare_sizes(json_size, display_size)
            if display_check == "mismatch" or jpeg_check == "mismatch":
                detail = f"JSON={size_text(json_size)}, JPEG={size_text(jpeg_size)}, EXIF={size_text(display_size)}"
                if display_check == "mismatch":
                    reason = "JSON이 JPEG 저장 크기와 일치" if jpeg_check == "match" else "JSON이 저장·표시 크기 모두와 불일치"
                    record_issue(issues, "DISPLAY_SIZE_MISMATCH", label_path, f"{detail}. {reason}. 좌표 보정 없음.")
                else:
                    record_issue(issues, "EXIF_SIZE_FRAME", label_path, f"{detail}. JSON은 EXIF 표시 크기와 일치.")
            records.append({"image_path": image_path, "json_path": label_path, "pair_status": status,
                            "json_vs_jpeg": jpeg_check, "json_vs_display": display_check, **image, **label})
    return counts, records


def select_previews(records, all_images, issues):
    """고정 시드로 네 그룹의 크기를 다양하게 표집하고 지정한 파일을 추가한다."""
    eligible = [record for record in records if record.get("display_size") and "rows" in record]
    required = {record["json_path"].name: record for record in eligible if record["json_path"].name in REVIEW_FILES}
    for name in REVIEW_FILES:
        if name not in required:
            record_issue(issues, "REQUIRED_PREVIEW_UNAVAILABLE", name, "시각화할 수 있는 이미지·JSON 쌍을 찾지 못했습니다.")
    if all_images:
        return eligible

    rng = random.Random(SAMPLE_SEED)
    selected, seen_sizes = [], set()
    for state in STATES:
        for orientation in (1, 6):
            group = [record for record in eligible if record.get("state") == state and record["orientation"] == orientation]
            rng.shuffle(group)
            group_sizes = set()
            for _ in range(min(SAMPLES_PER_GROUP, len(group))):
                record = min(group, key=lambda item: (item["display_size"] in seen_sizes, item["display_size"] in group_sizes))
                size = record["display_size"]
                group.remove(record)
                selected.append(record)
                seen_sizes.add(size)
                group_sizes.add(size)
    selected_paths = {record["json_path"] for record in selected}
    selected.extend(record for record in required.values() if record["json_path"] not in selected_paths)
    return selected


@contextmanager
def output_file(path, **options):
    """임시 파일에 완성한 결과를 같은 디렉터리의 대상 파일로 교체한다."""
    with NamedTemporaryFile(dir=path.parent, delete=False, **options) as stream:
        temporary_path = Path(stream.name)
        try:
            yield stream
            stream.close()
            temporary_path.replace(path)
        finally:
            stream.close()
            temporary_path.unlink(missing_ok=True)


def draw_box(draw, edges, original_size, scaled_size, color):
    """두 축의 실제 축소율로 경계를 옮기고 사진에 보이는 선분만 그린다."""
    width, height = scaled_size
    scale_x, scale_y = width / original_size[0], height / original_size[1]
    left, top, right, bottom = edges
    if right < 0 or left > original_size[0] or bottom < 0 or top > original_size[1]:
        return
    x0 = round(max(0, min(width - 1, left * scale_x)))
    x1 = round(max(0, min(width - 1, right * scale_x)))
    y0 = round(max(0, min(height - 1, top * scale_y)))
    y1 = round(max(0, min(height - 1, bottom * scale_y)))
    # 사진 밖의 경계를 가장자리의 가짜 선으로 표시하지 않는다.
    segments = []
    for edge, x in ((left, x0), (right, x1)):
        if 0 <= edge <= original_size[0]:
            segments.append((x, y0, x, y1))
    for edge, y in ((top, y0), (bottom, y1)):
        if 0 <= edge <= original_size[1]:
            segments.append((x0, y, x1, y))
    for segment in segments:
        draw.line(segment, fill="#202020", width=7)
    for segment in segments:
        draw.line(segment, fill=color, width=4)


def save_preview(path, record, *, full_resolution=False):
    """EXIF 적용 사진, 두 후보의 시설물 경계 및 원본 배열을 한 PNG에 표시한다."""
    panel_width, gap = 560, 16
    state = record.get("state")
    state_text = f"{STATES[state][2]} ({state})" if state in STATES else "Unknown"
    with Image.open(record["image_path"]) as source, ImageOps.exif_transpose(source) as display:
        original_size = display.size
        if original_size != record["display_size"]:
            raise ValueError("메타데이터 검사 이후 EXIF 표시 크기가 변경되었습니다.")
        photo = display.convert("RGB")
        if full_resolution:
            panel_width = max(panel_width, photo.width)
        else:
            photo.thumbnail((panel_width, 760), Image.Resampling.LANCZOS)

    rows = [row for row in record["rows"] if "bbox_original" in row]
    captions = []
    for row in rows:
        location = f"[{row['annotation_index']}][{row['bbox_index']}]"
        bounds = f"TLWH: {row['tlwh_bounds']} / LTWH: {row['ltwh_bounds']}"
        captions.extend(textwrap.wrap(f"{location} {row['bbox_original']}  |  {bounds}", width=110))
    canvas_width = 2 * panel_width + 3 * gap
    photo_y = 156
    footer_y = photo_y + photo.height + 14
    with Image.new("RGB", (canvas_width, footer_y + 48 + 21 * len(captions)), "#111827") as canvas:
        draw = ImageDraw.Draw(canvas)
        title = ImageFont.load_default(size=24)
        normal = ImageFont.load_default(size=16)
        ink, muted = "#f9fafb", "#d1d5db"
        draw.text((gap, 12), record["image_path"].name, font=title, fill=ink)
        draw.text((gap, 44), f"State: {state_text}  |  EXIF Orientation: {record['orientation']}  |  Facility BBOX",
                  font=normal, fill=muted)
        sizes = f"JPEG: {size_text(record['jpeg_size'])}  |  EXIF display: {size_text(original_size)}  |  JSON: {size_text(record.get('json_size'))}"
        draw.text((gap, 68), sizes, font=normal, fill=muted)
        warning = "JSON size matches EXIF display." if record["json_vs_display"] == "match" else "JSON size differs or is unavailable. Raw coordinates plotted as-is."
        draw.text((gap, 92), warning, font=normal, fill=muted)
        for index, (order, fields) in enumerate(ORDERS.items()):
            x = gap + index * (panel_width + gap)
            prefix = order.lower()
            with photo.copy() as panel:
                panel_draw = ImageDraw.Draw(panel)
                for row in rows:
                    edges = row.get(f"{prefix}_xyxy")
                    if edges:
                        color = "#34d399" if row[f"{prefix}_bounds"] == "inside" else "#fb923c"
                        draw_box(panel_draw, edges, original_size, photo.size, color)
                canvas.paste(panel, (x + (panel_width - photo.width) // 2, photo_y))
            bounds = Counter(row[f"{prefix}_bounds"] for row in rows)
            draw.text((x, 124), f"{order} [{fields}]  |  in {bounds['inside']} / out {bounds['outside']} / invalid {bounds['invalid']}",
                      font=normal, fill="#fdba74" if bounds["outside"] else ink)
        draw.text((gap, footer_y), "Original BBOX [annotation][box]  |  Green: inside  |  Orange: outside", font=normal, fill=ink)
        for index, line in enumerate(captions, start=1):
            draw.text((gap, footer_y + index * 21), line, font=normal, fill=muted)
        with output_file(path, mode="wb") as stream:
            canvas.save(stream, format="PNG")
    photo.close()


def distribution(values):
    """누락값을 포함해 값별 건수를 문자열 키로 집계한다."""
    return dict(sorted(Counter("unknown" if value is None else str(value) for value in values).items()))


def describe(values):
    """측정값의 표본 수, 최솟값, 중앙값, 평균, 최댓값을 요약한다."""
    if not values:
        return {"count": 0}
    return {"count": len(values), "min": min(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "max": max(values)}


def build_summary(counts, records, previews, issues, all_images):
    """이미지·JSON별 분포와 EXIF 표시 크기를 기준으로 한 후보별 통계를 만든다."""
    images = [record for record in records if record["image_path"] is not None]
    labels = [record for record in records if record["json_path"] is not None]
    pairs = [record for record in labels if record["pair_status"] == "matched"]
    boxes = [row for record in labels for row in record.get("rows", []) if "bbox_original" in row]
    annotation_checks = Counter()
    for record in labels:
        checks = {row["annotation_index"]: row["state_check"] for row in record.get("rows", [])}
        annotation_checks.update(checks.values())
    count_fields = {"images_read": sum("display_size" in record for record in images),
                    "json_objects_read": sum("rows" in record for record in labels),
                    "annotations": sum(record.get("annotation_count", 0) for record in labels), "boxes": len(boxes)}
    bounds = {}
    for order in ORDERS:
        prefix = order.lower()
        results = Counter(row[f"{prefix}_bounds"] for row in boxes)
        valid = [row for row in boxes if row[f"{prefix}_bounds"] == "inside"]
        bounds[order] = {result: results[result] for result in ("inside", "outside", "invalid", "unavailable")}
        bounds[order]["in_bounds_stats"] = {
            "width_px": describe([row["bbox_width"] for row in valid]),
            "height_px": describe([row["bbox_height"] for row in valid]),
            "image_area_ratio": describe([row[f"{prefix}_area_ratio"] for row in valid]),
        }
    return {
        "files": {**counts, **count_fields},
        "state": distribution(record.get("state") for record in labels),
        "state_codes": {code: data[0] for code, data in STATES.items()},
        "exif_orientation": distribution(record.get("orientation") for record in images),
        "orientation_by_state": {
            state or "unknown": distribution(record.get("orientation") for record in pairs if record.get("state") == state)
            for state in (*STATES, None)
        },
        "jpeg_size": distribution(size_text(record.get("jpeg_size")) for record in images),
        "display_size": distribution(size_text(record.get("display_size")) for record in images),
        "json_size": distribution(size_text(record.get("json_size")) for record in labels),
        "json_vs_jpeg": distribution(record["json_vs_jpeg"] for record in pairs),
        "json_vs_display": distribution(record["json_vs_display"] for record in pairs),
        "annotations_per_json": distribution(record.get("annotation_count") for record in labels),
        "state_vs_is_defect": {result: annotation_checks[result] for result in ("match", "mismatch", "unavailable")},
        "bbox_bounds": bounds,
        "bbox_assumptions": [
            "TLWH=[top,left,width,height], LTWH=[left,top,width,height]. 범위만으로 좌표 해석을 선택하지 않습니다.",
            "범위 검사 및 면적 비율의 기준은 실제 EXIF 표시 크기입니다. JSON 크기 불일치 시에도 좌표를 그대로 사용합니다.",
            "폭·높이·면적 통계는 각 해석에서 범위 안인 BBOX의 집합에 대한 가정 결과입니다. 면적 비율 1은 이미지 전체입니다.",
            "BBOX는 시설물 전체를 가리킵니다. 실제 시설물과의 정렬은 시각적으로 확인해야 합니다.",
        ],
        "previews": {"mode": "all" if all_images else "sample", "seed": SAMPLE_SEED, "per_group": SAMPLES_PER_GROUP,
                     "count": len(previews), "files": previews},
        "issue_counts": dict(sorted(Counter(issue["code"] for issue in issues).items())),
        "issues": issues,
    }


def write_csv(path, records, root):
    """이미지·JSON 메타데이터 및 BBOX별 후보를 UTF-8 CSV로 저장한다."""
    fields = ("image", "json", "pair_status", "state", "state_name", "orientation", "jpeg_size", "display_size",
              "json_size", "json_vs_jpeg", "json_vs_display", "annotation_count")
    with output_file(path, mode="w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(*fields, *BOX_FIELDS))
        writer.writeheader()
        for record in records:
            base = {key: record.get(key) for key in fields}
            for kind in ("image", "json"):
                original = record[f"{kind}_path"]
                base[kind] = original.relative_to(root).as_posix() if original else None
            for field in ("jpeg_size", "display_size", "json_size"):
                base[field] = size_text(base[field])
            base["state_name"] = STATES[base["state"]][0] if base["state"] in STATES else None
            for row in record.get("rows") or [{}]:
                formatted = {**row}
                for field in ("tlwh_xyxy", "ltwh_xyxy"):
                    if field in row:
                        formatted[field] = json.dumps(row[field])
                writer.writerow({**base, **formatted})


def main(argv=None):
    """CLI 인자를 검증하고 전체 자료 검사와 비교 이미지·EDA 생성을 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=RAW_DIR, help="입력 루트 (기본: data/raw)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="결과 루트 (기본: data/processed/inspection)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="읽을 수 있는 일대일 대응 쌍을 모두 시각화")
    mode.add_argument("--full-resolution", metavar="FILE", help="지정 JPEG 또는 JSON 파일 1건의 비교 이미지를 원본 픽셀 크기로 저장")
    args = parser.parse_args(argv)
    root, output = args.data_dir.resolve(), args.output_dir.resolve()
    preview_dir = output / ("bbox_previews_full_resolution" if args.full_resolution else "bbox_previews")
    if not root.is_dir():
        parser.error(f"입력 디렉터리가 없습니다: {root}")
    if (any(preview_dir.is_relative_to(path) for path in (root, RAW_DIR.resolve()))
            or preview_dir.resolve() != preview_dir):
        parser.error("입력·원본 디렉터리 밖의 실제 경로에만 결과를 저장할 수 있습니다.")

    issues = []
    counts, records = inspect_dataset(root, issues)
    if args.full_resolution:
        filename = args.full_resolution.casefold()
        matches = [record for record in records if record.get("display_size") and "rows" in record
                   and filename in (record["image_path"].name.casefold(), record["json_path"].name.casefold())]
        if len(matches) != 1:
            parser.error(f"시각화할 일대일 대응 파일을 찾지 못했습니다: {args.full_resolution}")
        record = matches[0]
        preview_dir.mkdir(parents=True, exist_ok=True)
        path = preview_dir / record["json_path"].with_suffix(".png").name
        save_preview(path, record, full_resolution=True)
        print(f"원본 픽셀 크기 비교 이미지: {path}")
        return 0

    selected = select_previews(records, args.all, issues)
    preview_dir.mkdir(parents=True, exist_ok=True)
    previews = []
    for record in selected:
        name = record["json_path"].with_suffix(".png").name
        try:
            save_preview(preview_dir / name, record)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            record_issue(issues, "PREVIEW_WRITE", record["image_path"], str(error), level="error")
            continue
        previews.append({"file": f"bbox_previews/{name}", "json": record["json_path"].name,
                         "state": record.get("state"), "orientation": record["orientation"],
                         "display_size": size_text(record["display_size"])})

    # 파일별 상세 경로는 입력 루트를 기준으로 저장해 다른 환경에서도 비교할 수 있게 한다.
    for issue in issues:
        path = Path(issue["path"])
        if path.is_relative_to(root):
            issue["path"] = path.relative_to(root).as_posix()
    summary = build_summary(counts, records, previews, issues, args.all)
    write_csv(output / "bbox_candidates.csv", records, root)
    with output_file(output / "eda_summary.json", mode="w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"이미지 {counts['images']}개, JSON {counts['json_files']}개, 일대일 대응 {counts['pairs']}쌍, 비교 이미지 {len(previews)}장")
    for order, results in summary["bbox_bounds"].items():
        print(f"{order} / EXIF 표시 크기: 범위 안 {results['inside']}개, 밖 {results['outside']}개, "
              f"무효 {results['invalid']}개, 검사 불가 {results['unavailable']}개")
    print(f"결과 디렉터리: {output}")
    return int(any(issue["level"] == "error" for issue in issues))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
