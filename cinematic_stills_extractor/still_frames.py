from pathlib import Path
from typing import NamedTuple
from rich.progress import (
    Progress,
)
from collections.abc import Iterable
import cv2
import polars as pl
import itertools
import numpy as np

FRAME_SIZE = 320


class Frame(NamedTuple):
    index: int
    quality: float


def frame_quality_from(frame: np.ndarray) -> float:
    """Compute frame quality based on blur and luminance."""
    frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur_measure = cv2.Laplacian(gray, cv2.CV_64F).var()
    luma_std = gray.std()
    return float(0.7 * blur_measure + 0.3 * luma_std)


def extract_best_frames_from(
    video_path: Path,
    *,
    window_sec: float = 2.0,
    consideration_treshold: float = 0.5,
    progress_keeper: Progress,
) -> pl.DataFrame:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    def take_frames_from(
        capture,
        *,
        f_number: int,
        offset: int,
    ) -> Iterable[tuple[int, np.ndarray]]:
        for count in range(f_number):
            success, frame = capture.read()
            if not success:
                continue
            idx = offset + count
            yield idx, frame

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    window_frames = int(window_sec * fps)
    picks_per_window = int(window_frames * consideration_treshold)
    task = progress_keeper.add_task("Extracting frames...", total=frame_count)

    frame_keeper: list[list[Frame]] = []
    offset = 0
    while offset < frame_count:
        window = take_frames_from(
            capture,
            f_number=window_frames,
            offset=offset,
        )
        frames = (
            Frame(index=idx, quality=frame_quality_from(frame))
            for idx, frame in window
        )
        top_frames = sorted(frames, key=lambda x: x.quality)
        if top_frames:
            start_index = min(
                picks_per_window,
                len(top_frames),
            )

            frame_keeper.append(top_frames[-start_index:])
        progress_keeper.advance(task, advance=window_frames)
        offset += window_frames

    frame_idx, quality_scores = zip(
        *itertools.chain.from_iterable(frame_keeper)
    )

    capture.release()
    progress_keeper.update(task, completed=True)
    return pl.concat(
        (
            pl.Series(
                "frame_idx",
                frame_idx,
                dtype=pl.Int32,
            ).to_frame(),
            pl.Series(
                "quality_score",
                quality_scores,
                dtype=pl.Float32,
            ).to_frame(),
        ),
        how="horizontal",
    )
