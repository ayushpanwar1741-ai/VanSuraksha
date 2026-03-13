import argparse
from pathlib import Path

import cv2
import yolov5

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def should_use_upload(name: str) -> bool:
    n = name.lower()
    if "result" in n or "non_fire" in n or "signature" in n:
        return False
    return ("fire" in n) or ("smoke" in n)


def write_yolo_label(path: Path, lines):
    text = "\n".join(lines)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Auto-label positive fire/smoke images using existing yolocff model")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--max-shots", type=int, default=120)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    out = Path(args.output).resolve()
    images_out = out / "images"
    labels_out = out / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    model_path = root / "Models" / "yolocff.pt"
    model = yolov5.load(str(model_path))

    sources = []

    shots = sorted((root / "static" / "shots").glob("*"))
    shot_imgs = [p for p in shots if p.suffix.lower() in IMAGE_EXTS][: max(0, args.max_shots)]
    for p in shot_imgs:
        sources.append(("shot", p))

    uploads = sorted((root / "static" / "uploads").glob("*"))
    for p in uploads:
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        if should_use_upload(p.name):
            sources.append(("upload", p))

    keep = 0
    for idx, (_src_type, img_path) in enumerate(sources):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        pred = model([frame], size=640)
        names = model.names
        lines = []

        if pred.xyxyn is None or len(pred.xyxyn) == 0 or len(pred.xyxyn[0]) == 0:
            continue

        rows = pred.xyxyn[0].cpu().numpy()
        for row in rows:
            conf = float(row[4])
            if conf < args.conf:
                continue
            cls_id = int(row[5])
            cls_name = str(names.get(cls_id, "")).lower()
            if "fire" in cls_name:
                out_cls = 0
            elif "smoke" in cls_name:
                out_cls = 1
            else:
                continue

            x1, y1, x2, y2 = [float(v) for v in row[:4]]
            xc = (x1 + x2) / 2.0
            yc = (y1 + y2) / 2.0
            bw = (x2 - x1)
            bh = (y2 - y1)
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{out_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            continue

        stem = f"pos_{idx:05d}_{img_path.stem.replace(' ', '_')}"
        dst_img = images_out / f"{stem}.jpg"
        dst_lbl = labels_out / f"{stem}.txt"
        cv2.imwrite(str(dst_img), frame)
        write_yolo_label(dst_lbl, lines)
        keep += 1

    print(f"Total candidate images: {len(sources)}")
    print(f"Auto-labeled positives: {keep}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
