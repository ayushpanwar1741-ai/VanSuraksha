import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_CLASSES = {0, 1}  # 0=fire, 1=smoke


def is_valid_yolo_line(line: str) -> bool:
    parts = line.strip().split()
    if not parts:
        return True
    if len(parts) != 5:
        return False
    try:
        cls = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])
    except ValueError:
        return False
    if cls not in ALLOWED_CLASSES:
        return False
    return all(0.0 <= v <= 1.0 for v in (x, y, w, h))


def validate_or_empty_label(src_label: Path, keep_empty_for_missing: bool) -> str | None:
    if not src_label.exists():
        return "" if keep_empty_for_missing else None

    lines = src_label.read_text(encoding="utf-8", errors="ignore").splitlines()
    valid = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if is_valid_yolo_line(ln):
            valid.append(ln)

    return "\n".join(valid)


def collect_samples(source_root: Path, keep_empty_for_missing: bool):
    images_root = source_root / "images"
    labels_root = source_root / "labels"
    if not images_root.exists() or not labels_root.exists():
        raise FileNotFoundError(
            f"Expected '{images_root}' and '{labels_root}' in source: {source_root}"
        )

    samples = []
    for img in images_root.rglob("*"):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = img.relative_to(images_root)
        lbl = (labels_root / rel).with_suffix(".txt")
        label_text = validate_or_empty_label(lbl, keep_empty_for_missing)
        if label_text is None:
            continue
        samples.append((img, rel, label_text))

    return samples


def write_split(out_root: Path, split_name: str, rows):
    img_dir = out_root / "images" / split_name
    lbl_dir = out_root / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        src_img, out_name, label_text = row
        dst_img = img_dir / out_name
        dst_lbl = lbl_dir / Path(out_name).with_suffix(".txt")
        shutil.copy2(src_img, dst_img)
        dst_lbl.write_text(label_text + ("\n" if label_text else ""), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Merge YOLO fire/smoke datasets, validate labels, and split into train/val/test."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Source dataset roots. Each source must contain 'images/' and 'labels/'",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output dataset root (will create images/{train,val,test} and labels/{train,val,test})",
    )
    parser.add_argument("--train", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val", type=float, default=0.2, help="Val split ratio")
    parser.add_argument("--test", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--drop-missing-label",
        action="store_true",
        help="Drop images that do not have a matching label file (default keeps them as hard negatives)",
    )

    args = parser.parse_args()
    if round(args.train + args.val + args.test, 6) != 1.0:
        raise ValueError("train + val + test must sum to 1.0")

    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for idx, src in enumerate(args.sources):
        src_root = Path(src).resolve()
        samples = collect_samples(src_root, keep_empty_for_missing=not args.drop_missing_label)
        ds_name = src_root.name.replace(" ", "_")
        for n, (img, _rel, label_text) in enumerate(samples):
            out_name = f"{idx}_{ds_name}_{n:07d}{img.suffix.lower()}"
            all_rows.append((img, out_name, label_text))

    if not all_rows:
        raise RuntimeError("No usable images found after validation.")

    random.seed(args.seed)
    random.shuffle(all_rows)

    total = len(all_rows)
    n_train = int(total * args.train)
    n_val = int(total * args.val)
    n_test = total - n_train - n_val

    train_rows = all_rows[:n_train]
    val_rows = all_rows[n_train:n_train + n_val]
    test_rows = all_rows[n_train + n_val:]

    write_split(out_root, "train", train_rows)
    write_split(out_root, "val", val_rows)
    write_split(out_root, "test", test_rows)

    yaml_path = out_root / "fire_smoke.yaml"
    yaml_path.write_text(
        "\n".join([
            f"path: {out_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            "nc: 2",
            "names: [fire, smoke]",
            "",
        ]),
        encoding="utf-8",
    )

    empty_labels = sum(1 for _img, _name, lbl in all_rows if not lbl.strip())
    print(f"Total samples: {total}")
    print(f"Train/Val/Test: {len(train_rows)}/{len(val_rows)}/{len(test_rows)}")
    print(f"Empty-label negatives: {empty_labels}")
    print(f"YOLO config written: {yaml_path}")


if __name__ == "__main__":
    main()
