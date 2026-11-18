"""AI-Hub 볼라드 원본을 읽기 전용으로 검사한다. 좌표 변환이나 파일 저장은 하지 않는다."""

import argparse
import json
import math
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
STATE_NAMES = {"0": "양호", "1": "불량"}


def add_issue(report, level, code, path, reason):
    """검사 결과에 항목을 추가한다. 입력은 보고서·심각도·유형·경로·원인이며 반환값은 없다."""
    report["issues"].append((level, code, path, reason))


def discover_files(root, report):
    """입력 루트를 재귀 탐색해 이미지·JSON 경로 목록을 반환하고 확장자·탐색 오류를 기록한다."""
    images, labels = [], []

    def on_error(error):
        add_issue(report, "오류", "DIRECTORY_READ", Path(error.filename or root), str(error))

    for directory, _, filenames in root.walk(on_error=on_error):
        for name in filenames:
            path = directory / name
            report["distributions"]["파일 확장자"][path.suffix or "<없음>"] += 1
            extension = path.suffix.lower()
            if extension in IMAGE_EXTENSIONS:
                images.append(path)
            elif extension == ".json":
                labels.append(path)
            else:
                add_issue(report, "오류", "FILE_EXTENSION", path, "허용 확장자는 .jpg, .jpeg, .json")
    return sorted(images), sorted(labels)


def index_files(paths, kind, report):
    """경로 목록을 대소문자 무시 stem 키로 묶어 반환하고, 파일명·대응 키 중복을 별도로 기록한다."""
    by_name, by_key = defaultdict(list), defaultdict(list)
    for path in paths:
        by_name[path.name.casefold()].append(path)
        by_key[path.stem.casefold()].append(path)
    for title, index in (("파일명", by_name), ("대응 키", by_key)):
        groups = [members for members in index.values() if len(members) > 1]
        report["stats"][f"{kind} 중복 {title} 그룹"] = len(groups)
        report["stats"][f"{kind} 중복 {title} 파일"] = sum(len(members) for members in groups)
        for members in groups:
            code = "DUPLICATE_FILENAME" if title == "파일명" else "DUPLICATE_MATCH_KEY"
            add_issue(report, "오류", code, members[0], f"{kind} {title} 중복: " + ", ".join(str(p) for p in members))
    return by_key


def match_files(images, labels, report):
    """이미지·JSON 목록에서 유일한 stem 쌍만 JSON→이미지 사전으로 반환하고 누락·모호성을 기록한다."""
    image_index = index_files(images, "이미지", report)
    label_index = index_files(labels, "JSON", report)
    pairs = {}
    for key in sorted(image_index.keys() | label_index.keys()):
        image_paths, label_paths = image_index.get(key, []), label_index.get(key, [])
        if not label_paths:
            report["stats"]["이미지만 존재"] += len(image_paths)
            for path in image_paths:
                add_issue(report, "오류", "JSON_MISSING", path, "같은 stem의 JSON 없음")
        elif not image_paths:
            report["stats"]["JSON만 존재"] += len(label_paths)
            for path in label_paths:
                add_issue(report, "오류", "IMAGE_MISSING", path, "같은 stem의 이미지 없음")
        elif len(image_paths) == len(label_paths) == 1:
            pairs[label_paths[0]] = image_paths[0]
        else:
            # 중복이 있으면 임의로 한 파일을 고르지 않고 교차 비교를 보류한다.
            report["stats"]["대응이 모호한 키"] += 1
    report["stats"]["일대일 대응 쌍"] = len(pairs)
    return pairs


def inspect_image(path, report):
    """이미지를 verify·전체 디코딩하고 저장 크기·EXIF 표시 크기·방향 사전을 반환한다. 실패도 기록한다."""
    metadata = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                metadata["size"] = image.size
                report["distributions"]["JPEG 저장 크기"][str(image.size)] += 1
                if image.format != "JPEG":
                    add_issue(report, "오류", "IMAGE_FORMAT", path, f"확장자와 달리 실제 형식은 {image.format}")
                try:
                    orientation = image.getexif().get(274)
                    metadata["orientation"] = orientation
                    report["distributions"]["EXIF Orientation"][repr(orientation) if orientation is not None else "없음"] += 1
                    if orientation is not None and (type(orientation) is not int or orientation not in range(1, 9)):
                        add_issue(report, "오류", "EXIF_ORIENTATION", path, f"허용되지 않는 방향 값: {orientation!r}")
                    else:
                        # 픽셀을 회전하지 않고 EXIF 적용 시의 논리적인 가로·세로만 계산한다.
                        metadata["display_size"] = image.size[::-1] if orientation in (5, 6, 7, 8) else image.size
                except Exception as error:
                    report["distributions"]["EXIF Orientation"]["읽기 실패"] += 1
                    add_issue(report, "오류", "EXIF_READ", path, f"{type(error).__name__}: {error}")
                image.verify()
            # verify는 픽셀 전체 디코딩을 대신하지 않으므로 다시 열어서 load까지 수행한다.
            with Image.open(path) as image:
                image.load()
        report["stats"]["이미지 유효성 검사 성공"] += 1
    except Exception as error:
        # Pillow 플러그인별 예외가 달라도 한 파일의 실패가 전체 검사를 막지 않게 한다.
        report["stats"]["이미지 유효성 검사 실패"] += 1
        if not metadata:
            report["distributions"]["EXIF Orientation"]["헤더 읽기 실패"] += 1
        add_issue(report, "오류", "IMAGE_READ", path, f"{type(error).__name__}: {error}")
    return metadata


def check_fields(data, fields, path, prefix, report):
    """객체와 필수 필드→타입 사전을 받아 누락·타입·빈 문자열을 기록한다. 반환값은 없다."""
    for field, expected in fields.items():
        name = f"{prefix}.{field}" if prefix else field
        if field not in data:
            add_issue(report, "오류", "FIELD_MISSING", path, f"필수 필드 누락: {name}")
        elif type(data[field]) is not expected:
            # bool은 int의 하위 타입이지만 이미지 크기로 허용하지 않는다.
            add_issue(report, "오류", "FIELD_TYPE", path, f"{name}: {expected.__name__} 필요, 실제 {type(data[field]).__name__}")
        elif expected is str and not data[field].strip():
            add_issue(report, "오류", "FIELD_EMPTY", path, f"빈 문자열: {name}")


def count_field(data, field, title, report):
    """객체의 필드 값을 문자열화해 지정한 분포에 집계한다. 누락도 집계하며 반환값은 없다."""
    value = json.dumps(data[field], ensure_ascii=False) if field in data else "<누락>"
    if title == "상태 (JSON 단위)" and type(data.get(field)) is str:
        value = f"{STATE_NAMES.get(data[field], '알 수 없음')} ({data[field]})"
    report["distributions"][title][value] += 1


def load_json(path, report):
    """UTF-8(BOM 허용) JSON 객체를 반환한다. 읽기·파싱·최상위 타입 오류 시 기록 후 None을 반환한다."""
    try:
        with path.open(encoding="utf-8-sig") as stream:
            data = json.load(stream)
    except (OSError, ValueError, UnicodeError, RecursionError) as error:
        add_issue(report, "오류", "JSON_READ", path, f"{type(error).__name__}: {error}")
        return None
    report["stats"]["JSON 파싱 성공"] += 1
    if not isinstance(data, dict):
        add_issue(report, "오류", "JSON_ROOT_TYPE", path, f"최상위 객체가 dict가 아님: {type(data).__name__}")
        return None
    return data


def compare_image_info(info, path, image_path, metadata, report):
    """info를 실제 대응 이미지의 이름·두 크기 기준과 비교하고 유효한 JSON (가로, 세로) 또는 None을 반환한다."""
    filename = info.get("filename")
    if isinstance(filename, str) and filename:
        if Path(filename).name != filename or Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
            add_issue(report, "오류", "JSON_FILENAME", path, f"info.filename은 경로 없는 JPEG 파일명이어야 함: {filename!r}")
        if Path(filename).stem.casefold() != path.stem.casefold():
            add_issue(report, "오류", "JSON_MATCH_KEY", path, f"JSON stem과 info.filename의 stem이 다름: {filename!r}")
        if image_path is not None and filename != image_path.name:
            add_issue(report, "오류", "IMAGE_FILENAME_MISMATCH", path, f"JSON={filename!r}, 실제={image_path.name!r}")
    size = (info.get("width"), info.get("height"))
    if not all(type(value) is int and value > 0 for value in size):
        add_issue(report, "오류", "JSON_IMAGE_SIZE", path, f"width·height는 양의 정수여야 함: {size!r}")
        return None
    report["distributions"]["JSON 크기"][str(size)] += 1
    raw_size, display_size = metadata.get("size"), metadata.get("display_size")
    if raw_size is None:
        report["stats"]["이미지 크기 교차 비교 불가"] += 1
        return size
    report["stats"]["저장 크기 일치" if size == raw_size else "저장 크기 불일치"] += 1
    if display_size is not None:
        report["stats"]["EXIF 표시 크기 일치" if size == display_size else "EXIF 표시 크기 불일치"] += 1
    if size != raw_size:
        detail = f"JSON={size}, JPEG 저장={raw_size}, EXIF 표시={display_size}"
        if size == display_size:
            add_issue(report, "확인", "EXIF_SIZE_FRAME", path, detail + "; 라벨 좌표 기준은 시각화 필요")
        else:
            add_issue(report, "오류", "IMAGE_SIZE_MISMATCH", path, detail + "; 어느 크기와도 일치하지 않음")
    return size


def is_finite_number(value):
    """입력값이 bool을 제외한 유한한 int·float인지 반환한다. 실수 범위를 넘는 정수도 거부한다."""
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def check_bbox(boxes, path, location, sizes, report):
    """중첩 bbox 목록의 형식·값을 검사하고 크기·좌표 순서별 범위를 가정하여 집계한다. 반환값은 없다."""
    if not boxes:
        add_issue(report, "오류", "BBOX_EMPTY", path, f"{location}.annotation_info가 비어 있음")
    for index, box in enumerate(boxes):
        report["stats"]["bbox 전체"] += 1
        where = f"{location}.annotation_info[{index}]"
        if not isinstance(box, list) or len(box) != 4:
            add_issue(report, "오류", "BBOX_SHAPE", path, f"{where}: 값 4개의 목록 필요, 원본={box!r}")
            continue
        if not all(is_finite_number(value) for value in box):
            add_issue(report, "오류", "BBOX_NUMBER", path, f"{where}: 유한한 숫자 4개 필요(bool·문자열 제외), 원본={box!r}")
            continue
        first, second, width, height = box
        if first < 0 or second < 0 or width <= 0 or height <= 0:
            add_issue(report, "오류", "BBOX_VALUE", path, f"{where}: 시작점은 0 이상, 폭·높이는 양수 필요, 원본={box!r}")
            continue
        report["stats"]["bbox 형식·값 정상"] += 1
        results = {}
        for canvas, size in sizes.items():
            for order in ("TLWH(문서)", "LTWH(비교)"):
                inside = None
                if size is not None:
                    # 원본을 바꾸지 않고 두 순서의 끝점만 계산한다. 두 해석 모두 폭·높이 위치는 동일하다.
                    left, top = (second, first) if order == "TLWH(문서)" else (first, second)
                    inside = left + width <= size[0] and top + height <= size[1]
                result = "비교 불가" if inside is None else "범위 안" if inside else "범위 밖"
                report["bounds"][(canvas, order, result)] += 1
                results[(canvas, order)] = inside
        documented = results[("JSON", "TLWH(문서)")]
        alternative = results[("JSON", "LTWH(비교)")]
        if documented is False or alternative is False:
            code = "BBOX_ORDER_REVIEW" if documented != alternative else "BBOX_RANGE_REVIEW"
            add_issue(report, "확인", code, path,
                      f"{where}: 원본={box!r}, JSON 크기={sizes['JSON']}, "
                      f"TLWH 범위 안={documented}, LTWH 범위 안={alternative}; 해석 확정 전 오류 판정 보류")


def inspect_json(path, image_path, metadata, report):
    """JSON·대응 이미지 정보·보고서를 받아 필수 구조·분포·이미지 대응·bbox를 검사한다. 반환값은 없다."""
    data = load_json(path, report)
    if data is None:
        return
    check_fields(data, {"info": dict, "description": dict, "annotations": list}, path, "", report)
    json_size = None
    info = data.get("info")
    if isinstance(info, dict):
        check_fields(info, {"image_id": str, "date": str, "city_id": str, "filename": str,
                            "width": int, "height": int}, path, "info", report)
        json_size = compare_image_info(info, path, image_path, metadata, report)
    description = data.get("description")
    if isinstance(description, dict):
        check_fields(description, {"facility": str, "sub_class": str, "state": str}, path, "description", report)
        for field, title in (("facility", "시설물 코드 (JSON 단위)"), ("sub_class", "하위 유형 (JSON 단위)"),
                             ("state", "상태 (JSON 단위)")):
            count_field(description, field, title, report)
        state = description.get("state")
        if not isinstance(state, str) or state not in STATE_NAMES:
            add_issue(report, "오류", "STATE_VALUE", path, f"description.state는 '0'(양호) 또는 '1'(불량) 필요: {state!r}")
    annotations = data.get("annotations")
    if not isinstance(annotations, list):
        return
    report["distributions"]["JSON별 어노테이션 수"][str(len(annotations))] += 1
    if not annotations:
        add_issue(report, "확인", "ANNOTATIONS_EMPTY", path, "annotations가 비어 있음; 객체 미표기인지 의도한 음성 샘플인지 확인 필요")
    for index, annotation in enumerate(annotations):
        report["stats"]["어노테이션 전체"] += 1
        location = f"annotations[{index}]"
        if not isinstance(annotation, dict):
            add_issue(report, "오류", "ANNOTATION_TYPE", path, f"{location}: dict 필요")
            continue
        check_fields(annotation, {"annotation_id": str, "label_id": str, "label_name": str,
                                  "is_defect": str, "annotation_type": str, "annotation_info": list},
                     path, location, report)
        for field, title in (("label_name", "시설물 이름 (어노테이션 단위)"),
                             ("annotation_type", "annotation_type"), ("is_defect", "is_defect")):
            count_field(annotation, field, title, report)
        defect = annotation.get("is_defect")
        if not isinstance(defect, str) or defect not in ("정상", "불량"):
            add_issue(report, "오류", "DEFECT_VALUE", path, f"{location}.is_defect: '정상' 또는 '불량' 필요, 실제={defect!r}")
        if annotation.get("annotation_type") != "bbox":
            add_issue(report, "확인", "GEOMETRY_UNSUPPORTED", path, f"{location}: bbox 외 좌표 검증은 미지원")
        elif isinstance(annotation.get("annotation_info"), list):
            sizes = {"JSON": json_size, "JPEG 저장": metadata.get("size"), "EXIF 표시": metadata.get("display_size")}
            check_bbox(annotation["annotation_info"], path, location, sizes, report)


def print_report(root, report, max_examples):
    """루트·집계 결과·유형별 사례 제한(0은 전체)을 받아 터미널 보고서를 출력한다. 반환값은 없다."""
    stats = report["stats"]
    print(f"검사 대상: {root}")
    print("대응 키: 확장자를 제외한 파일명(stem), 대소문자 무시; 중복 시 자동 대응하지 않음")
    print("\n[파일·읽기·대응 검사]")
    for title in ("이미지 전체", "JSON 전체", "일대일 대응 쌍", "이미지만 존재", "JSON만 존재", "대응이 모호한 키",
                  "이미지 중복 파일명 그룹", "이미지 중복 파일명 파일", "JSON 중복 파일명 그룹", "JSON 중복 파일명 파일",
                  "이미지 중복 대응 키 그룹", "이미지 중복 대응 키 파일", "JSON 중복 대응 키 그룹", "JSON 중복 대응 키 파일",
                  "이미지 유효성 검사 성공", "이미지 유효성 검사 실패", "JSON 파싱 성공",
                  "저장 크기 일치", "저장 크기 불일치", "EXIF 표시 크기 일치", "EXIF 표시 크기 불일치",
                  "이미지 크기 교차 비교 불가", "어노테이션 전체", "bbox 전체", "bbox 형식·값 정상"):
        print(f"  {title}: {stats[title]}")
    print("  일대일 대응은 파일 연결 결과이며, 모든 검사를 통과했다는 뜻은 아닙니다.")
    print("\n[분포]")
    for title, counts in report["distributions"].items():
        print(f"  {title}:")
        for value, count in sorted(counts.items()):
            print(f"    {value}: {count}")
    print("\n[bbox 범위: 좌표 해석·크기 기준별 가정 결과]")
    print("  TLWH=[top,left,width,height], LTWH=[left,top,width,height]")
    print("  형식·값 검사를 통과한 상자만 비교. 저장 크기와 EXIF 표시 크기를 별도로 사용합니다.")
    for canvas in ("JSON", "JPEG 저장", "EXIF 표시"):
        for order in ("TLWH(문서)", "LTWH(비교)"):
            counts = [f"{result}={report['bounds'][(canvas, order, result)]}" for result in ("범위 안", "범위 밖", "비교 불가")]
            print(f"  {canvas} / {order}: " + ", ".join(counts))
    print("  범위 안이어도 실제 객체와의 정렬은 보장하지 않습니다. 범위 밖도 라벨 오류로 확정하지 않습니다.")
    print("\n[오류·추가 확인 사항]")
    for level in ("오류", "확인"):
        items = [issue for issue in report["issues"] if issue[0] == level]
        print(f"  {level}: {len(items)}건 / 관련 파일·경로 {len({item[2] for item in items})}개")
        groups = defaultdict(list)
        for _, code, path, reason in items:
            groups[code].append((path, reason))
        for code, members in sorted(groups.items()):
            print(f"    {code}: {len(members)}건")
            for path, reason in members[:max_examples or None]:
                display = path.relative_to(root) if path.is_relative_to(root) else path
                print(f"      {display}: {reason}")
            if max_examples and len(members) > max_examples:
                print(f"      ... {len(members) - max_examples}건 생략 (--max-examples 0으로 전체 출력)")
    print("  건수는 검사 항목 기준이므로 같은 파일이 여러 항목에 포함될 수 있습니다.")
    print("\n원본 수정·좌표 변환·EXIF 회전은 수행하지 않았습니다.")
    print("학습 전 bbox 시각화로 좌표 순서와 EXIF 기준을 확인하고, 중복·유사 이미지 및 분할 누수를 점검해야 합니다.")


def main(argv=None):
    """명령행 인자를 받아 전체 검사를 실행하고 종료 코드(0=확정 오류 없음, 1=오류, 2=잘못된 경로)를 반환한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="검사 루트 (기본: 프로젝트 data/raw)")
    parser.add_argument("--max-examples", type=int, default=3, help="오류 유형별 사례 수 (기본: 3, 0: 전체)")
    args = parser.parse_args(argv)
    if args.max_examples < 0:
        parser.error("--max-examples는 0 이상이어야 합니다.")
    root = args.data_dir.resolve()
    if not root.is_dir():
        print(f"검사할 디렉터리가 없습니다: {root}", file=sys.stderr)
        return 2
    report = {"stats": Counter(), "distributions": defaultdict(Counter), "bounds": Counter(), "issues": []}
    images, labels = discover_files(root, report)
    report["stats"].update({"이미지 전체": len(images), "JSON 전체": len(labels)})
    if not images or not labels:
        add_issue(report, "오류", "DATASET_EMPTY", root, "이미지와 JSON이 각각 1개 이상 필요")
    pairs = match_files(images, labels, report)
    metadata = {path: inspect_image(path, report) for path in images}
    for path in labels:
        image_path = pairs.get(path)
        inspect_json(path, image_path, metadata.get(image_path, {}), report)
    print_report(root, report, args.max_examples)
    return 1 if any(issue[0] == "오류" for issue in report["issues"]) else 0


if __name__ == "__main__":
    # Windows 파이프·리디렉션에서도 한국어 파일명과 보고서를 UTF-8로 출력한다.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
