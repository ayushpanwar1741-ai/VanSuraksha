import argparse
import re
from pathlib import Path

import cv2

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".wmv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Conservative pattern: likely non-fire images by filename.
NEG_NAME_PATTERN = re.compile(r"(non[_\s-]?fire|no[_\s-]?fire|safe|clear|normal|signature)", re.IGNORECASE)


def ensure_dirs(out_root: Path):
    images_dir = out_root / "images"
    labels_dir = out_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def write_empty_label(labels_dir: Path, stem: str):
    (labels_dir / f"{stem}.txt").write_text("", encoding="utf-8")


def extract_video_frames(video_dir: Path, images_dir: Path, labels_dir: Path, sample_every: int, max_frames: int):
    count = 0
    for vf in sorted(video_dir.glob("*")):
        if vf.suffix.lower() not in VIDEO_EXTS:
            continue

        cap = cv2.VideoCapture(str(vf))
        if not cap.isOpened():
            continue

        frame_idx = 0
        saved_for_video = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % sample_every == 0:
                stem = f"neg_{vf.stem.replace(' ', '_')}_{frame_idx:06d}"
                out_img = images_dir / f"{stem}.jpg"
                cv2.imwrite(str(out_img), frame)
                write_empty_label(labels_dir, stem)
                count += 1
                saved_for_video += 1
                if max_frames > 0 and saved_for_video >= max_frames:
                    break
            frame_idx += 1

        cap.release()
    return count


def collect_upload_negatives(uploads_dir: Path, images_dir: Path, labels_dir: Path, include_all_uploads: bool):
    count = 0
    for img in sorted(uploads_dir.glob("*")):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        if not include_all_uploads and not NEG_NAME_PATTERN.search(img.stem):
            continue

        loaded = cv2.imread(str(img))
        if loaded is None:
            continue

        stem = f"neg_upload_{img.stem.replace(' ', '_')}"
        out_img = images_dir / f"{stem}.jpg"
        cv2.imwrite(str(out_img), loaded)
        write_empty_label(labels_dir, stem)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Build YOLO-ready camera negatives dataset from project videos/uploads.")
    parser.add_argument("--project-root", required=True, help="Path to Fire_Eye project root")
    parser.add_argument("--output", required=True, help="Output folder containing images/ and labels/")
    parser.add_argument("--sample-every", type=int, default=45, help="Save every Nth frame from each video")
    parser.add_argument("--max-frames-per-video", type=int, default=200, help="Cap extracted frames per video (0 = unlimited)")
    parser.add_argument("--include-all-uploads", action="store_true", help="Include all upload images as negatives")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_root = Path(args.output).resolve()

    video_dir = project_root / "static" / "video"
    uploads_dir = project_root / "static" / "uploads"

    images_dir, labels_dir = ensure_dirs(out_root)

    vid_count = 0
    upload_count = 0

    if video_dir.exists():
        vid_count = extract_video_frames(
            video_dir=video_dir,
            images_dir=images_dir,
            labels_dir=labels_dir,
            sample_every=max(1, args.sample_every),
            max_frames=max(0, args.max_frames_per_video),
        )

    if uploads_dir.exists():
        upload_count = collect_upload_negatives(
            uploads_dir=uploads_dir,
            images_dir=images_dir,
            labels_dir=labels_dir,
            include_all_uploads=args.include_all_uploads,
        )

    total = vid_count + upload_count
    print(f"Video-frame negatives: {vid_count}")
    print(f"Upload-image negatives: {upload_count}")
    print(f"Total negatives created: {total}")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
