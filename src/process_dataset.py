from pathlib import Path

from frame_extractor import extract_frames

# ==========================
# Configuration
# ==========================

RAW_DATA = Path("data/raw")

ORIGINAL_DIR = RAW_DATA / "original_sequences" / "youtube" / "c23" / "videos"

DEEPFAKE_DIR = RAW_DATA / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"

PROCESSED_DIR = Path("data/processed")

NUM_FRAMES = 60


def process_videos(input_dir, output_root):
    """
    Process every video in the given directory.
    """

    videos = sorted(input_dir.glob("*.mp4"))

    print(f"\nFound {len(videos)} videos in {input_dir}\n")

    for index, video_path in enumerate(videos, start=1):

        output_dir = output_root / video_path.stem

        print(f"[{index}/{len(videos)}] Processing {video_path.name}")

        saved = extract_frames(
            video_path=video_path,
            output_dir=output_dir,
            num_frames=NUM_FRAMES
        )

        print(f"✓ Saved {saved} frames\n")


if __name__ == "__main__":

    print("=" * 60)
    print("Processing Original Videos")
    print("=" * 60)

    process_videos(
        ORIGINAL_DIR,
        PROCESSED_DIR / "original"
    )

    print("=" * 60)
    print("Processing Deepfake Videos")
    print("=" * 60)

    process_videos(
        DEEPFAKE_DIR,
        PROCESSED_DIR / "deepfakes"
    )

    print("=" * 60)
    print("Frame extraction completed successfully!")
    print("=" * 60)