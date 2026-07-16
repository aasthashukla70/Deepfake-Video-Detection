import cv2
import numpy as np
from pathlib import Path


def extract_frames(video_path, output_dir, num_frames=60):
    """
    Extract uniformly sampled frames from a video.

    Args:
        video_path (str or Path): Path to the input video.
        output_dir (str or Path): Directory where extracted frames will be saved.
        num_frames (int): Number of frames to extract.

    Returns:
        int: Number of frames successfully saved.
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print(f"Error: No frames found in: {video_path}")
        cap.release()
        return 0

    # If the video has fewer frames than requested,
    # extract all available frames.
    frames_to_extract = min(num_frames, total_frames)

    # Uniformly spaced frame indices
    frame_indices = np.linspace(
        0,
        total_frames - 1,
        frames_to_extract,
        dtype=int
    )

    selected_indices = set(frame_indices)

    saved_count = 0
    current_frame = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if current_frame in selected_indices:
            frame_name = f"frame_{saved_count + 1:04d}.jpg"
            frame_path = output_dir / frame_name

            cv2.imwrite(str(frame_path), frame)

            saved_count += 1

        current_frame += 1

    cap.release()

    print(f"Saved {saved_count} frames from {video_path.name}")

    return saved_count


if __name__ == "__main__":

    video_path = "data/raw/original_sequences/youtube/c23/videos/033.mp4"

    output_dir = "data/processed/original/033"

    saved = extract_frames(
        video_path=video_path,
        output_dir=output_dir,
        num_frames=60
    )

    print(f"Total frames saved: {saved}")