#!/usr/bin/env python
"""Reorganize flat ImageNet validation images into ImageFolder layout.

Examples:
    python tools/data/imagenet/reorganize_val.py \
        --src-dir /path/to/imagenet/val_flat \
        --ann-file /path/to/val_annotations.txt \
        --dst-dir /path/to/imagenet/val

    python tools/data/imagenet/reorganize_val.py \
        --src-dir /path/to/imagenet/val_flat \
        --ann-file /path/to/val.txt \
        --label-source path_parent \
        --dst-dir /path/to/imagenet/val

    python tools/data/imagenet/reorganize_val.py \
        --src-dir /path/to/imagenet/val_flat \
        --ann-file /path/to/ILSVRC2012_validation_ground_truth.txt \
        --class-map /path/to/imagenet1000_clsidx_to_labels.txt \
        --class-index-offset 1 \
        --dst-dir /path/to/imagenet/val \
        --mode move
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert flat ImageNet val images into ImageFolder format."
    )
    parser.add_argument(
        "--src-dir",
        required=True,
        help="Directory containing flat validation images.",
    )
    parser.add_argument(
        "--ann-file",
        required=True,
        help=(
            "Annotation file. Each line can be either "
            "'image_name label' / 'image_name,label' / 'image_name\\tlabel', "
            "or a single label per line aligned with sorted images."
        ),
    )
    parser.add_argument(
        "--dst-dir",
        required=True,
        help="Output directory in ImageFolder layout.",
    )
    parser.add_argument(
        "--label-source",
        choices=("auto", "second_field", "path_parent"),
        default="auto",
        help=(
            "How to determine the class name when each annotation line contains "
            "at least two fields. 'path_parent' uses the parent folder from the "
            "first field, 'second_field' uses the second field, and 'auto' prefers "
            "path_parent when the first field includes a subdirectory."
        ),
    )
    parser.add_argument(
        "--class-map",
        default=None,
        help=(
            "Optional mapping file when labels are numeric. Supports either "
            "'index class_name' per line or one class_name per line."
        ),
    )
    parser.add_argument(
        "--class-index-offset",
        type=int,
        default=0,
        help=(
            "Offset applied before looking up numeric labels in --class-map. "
            "Use 1 for ImageNet ground-truth files that are 1-based."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "move", "symlink"),
        default="copy",
        help="How to place files into the target directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing target files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned work without modifying files.",
    )
    return parser.parse_args()


def collect_images(src_dir: Path) -> List[Path]:
    images = [
        path
        for path in src_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]
    images.sort(key=lambda path: path.name)
    if not images:
        raise FileNotFoundError(f"No images found in {src_dir}")
    return images


def load_class_map(class_map_path: Path, offset: int) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with class_map_path.open("r", encoding="utf-8") as file:
        raw_lines = [
            line.strip()
            for line in file
            if line.strip() and not line.lstrip().startswith("#")
        ]

    if not raw_lines:
        raise ValueError(f"Class map file is empty: {class_map_path}")

    first_fields = split_annotation_line(raw_lines[0])
    if len(first_fields) >= 2:
        for line in raw_lines:
            fields = split_annotation_line(line)
            if len(fields) < 2:
                raise ValueError(
                    "Mixed class-map format detected. Please use either two-column "
                    "'index class_name' lines or one class_name per line."
                )
            mapping[str(int(fields[0]))] = fields[1]
    else:
        for idx, class_name in enumerate(raw_lines):
            mapping[str(idx + offset)] = class_name

    return mapping


def split_annotation_line(line: str) -> List[str]:
    line = line.strip()
    if not line:
        return []
    if "," in line:
        parsed = next(csv.reader([line]))
        return [field.strip() for field in parsed if field.strip()]
    return line.split()


def parse_annotations(
    ann_file: Path,
    image_names: Sequence[str],
    label_source: str,
) -> List[Tuple[str, str]]:
    with ann_file.open("r", encoding="utf-8") as file:
        lines = [
            line.strip()
            for line in file
            if line.strip() and not line.lstrip().startswith("#")
        ]

    if not lines:
        raise ValueError(f"Annotation file is empty: {ann_file}")

    parsed_lines = [split_annotation_line(line) for line in lines]
    first = parsed_lines[0]
    if len(first) >= 2:
        pairs: List[Tuple[str, str]] = []
        for fields in parsed_lines:
            if len(fields) < 2:
                raise ValueError(
                    "Mixed annotation format detected. Expected each line to contain "
                    "at least image_name and label."
                )
            rel_path = Path(fields[0])
            image_name = rel_path.name
            use_path_parent = (
                label_source == "path_parent"
                or (
                    label_source == "auto"
                    and rel_path.parent != Path(".")
                    and rel_path.parent.name
                )
            )
            label = rel_path.parent.name if use_path_parent else fields[1]
            pairs.append((image_name, label))
        return pairs

    if len(parsed_lines) != len(image_names):
        raise ValueError(
            "Annotation count does not match image count when using label-only mode: "
            f"{len(parsed_lines)} labels vs {len(image_names)} images."
        )
    return [(image_name, fields[0]) for image_name, fields in zip(image_names, parsed_lines)]


def resolve_label(label: str, class_map: Dict[str, str] | None, offset: int) -> str:
    if class_map is None:
        return label

    if label.isdigit():
        mapped = class_map.get(str(int(label)))
        if mapped is not None:
            return mapped

        mapped = class_map.get(str(int(label) - offset))
        if mapped is not None:
            return mapped

    mapped = class_map.get(label)
    if mapped is not None:
        return mapped

    raise KeyError(
        f"Unable to resolve label '{label}' with the provided class map. "
        "Check whether the annotation labels are 0-based or 1-based."
    )


def ensure_target(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target file already exists: {path}. Use --overwrite to replace it."
            )
        if path.is_dir():
            raise IsADirectoryError(f"Target path is a directory: {path}")
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def place_file(src: Path, dst: Path, mode: str, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return

    ensure_target(dst, overwrite=overwrite)

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    args = parse_args()

    src_dir = Path(args.src_dir).expanduser().resolve()
    ann_file = Path(args.ann_file).expanduser().resolve()
    dst_dir = Path(args.dst_dir).expanduser().resolve()

    if not src_dir.is_dir():
        raise NotADirectoryError(f"Source directory does not exist: {src_dir}")
    if not ann_file.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {ann_file}")

    images = collect_images(src_dir)
    image_lookup = {path.name: path for path in images}
    pairs = parse_annotations(
        ann_file,
        [path.name for path in images],
        label_source=args.label_source,
    )

    class_map = None
    if args.class_map:
        class_map = load_class_map(
            Path(args.class_map).expanduser().resolve(),
            offset=args.class_index_offset,
        )

    planned: List[Tuple[Path, Path]] = []
    missing_images: List[str] = []
    class_names = set()

    for image_name, raw_label in pairs:
        src_path = image_lookup.get(image_name)
        if src_path is None:
            missing_images.append(image_name)
            continue
        class_name = resolve_label(
            raw_label,
            class_map=class_map,
            offset=args.class_index_offset,
        )
        dst_path = dst_dir / class_name / image_name
        planned.append((src_path, dst_path))
        class_names.add(class_name)

    if missing_images:
        preview = ", ".join(missing_images[:5])
        raise FileNotFoundError(
            f"{len(missing_images)} images referenced in annotations were not found in "
            f"{src_dir}. First few: {preview}"
        )

    print(f"Source images: {len(images)}")
    print(f"Planned files: {len(planned)}")
    print(f"Target classes: {len(class_names)}")
    print(f"Mode: {args.mode}")
    print(f"Dry run: {args.dry_run}")

    for src_path, dst_path in planned:
        place_file(
            src=src_path,
            dst=dst_path,
            mode=args.mode,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

    if planned:
        print(f"Done. Example output: {planned[0][1]}")
    else:
        print("Nothing to do.")


if __name__ == "__main__":
    main()
