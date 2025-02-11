import multiprocessing as mp
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
import polars as pl
import typer
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

console = Console()
app = typer.Typer()

# CPU and OpenCL setup
PHYSICAL_CORES = mp.cpu_count()

if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(True)
    cv2.setNumThreads(PHYSICAL_CORES)
    console.print("OpenCL: ✓", style="green")
else:
    raise RuntimeError(
        "OpenCL not available. Install OpenCV with OpenCL support: brew install opencv"
    )


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp: float
    quality: float
    image: np.ndarray


class VideoQualityAnalyzer:
    def __init__(
        self,
        video_path: Path,
        window_duration: float,
        picks_per_window: int,
        cache_dir: Path,
    ) -> None:
        self.video_path = video_path
        self.cache_dir = cache_dir
        self.cache_file = cache_dir / f"{video_path.stem}_analysis.parquet"

        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        # Set thread-safe video capture properties
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, PHYSICAL_CORES * 2)

        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.window_frames = int(window_duration * self.fps)
        self.picks_per_window = picks_per_window

        console.print("Video Info:", style="bold green")
        console.print(f"├── FPS: {self.fps:.2f}")
        console.print(f"├── Frames: {self.frame_count:,}")
        console.print(f"├── Duration: {self.frame_count / self.fps:.1f}s")
        console.print(f"└── Processing Threads: {PHYSICAL_CORES}")

    def analyze_frame(self, frame: np.ndarray) -> float:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(np.var(laplacian))

    def process_frames(self) -> Iterator[Frame]:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing frames...", total=self.frame_count)
            frame_buffer = []

            for frame_idx in range(self.frame_count):
                success, frame = self.capture.read()
                if not success:
                    break

                quality = self.analyze_frame(frame)
                frame_buffer.append(
                    Frame(
                        index=frame_idx,
                        timestamp=frame_idx / self.fps,
                        quality=quality,
                        image=frame,
                    )
                )

                if len(frame_buffer) >= self.window_frames:
                    sorted_frames = sorted(
                        frame_buffer, key=lambda x: x.quality, reverse=True
                    )
                    for frame in sorted_frames[: self.picks_per_window]:
                        yield frame
                    frame_buffer.clear()

                progress.update(task, advance=1)

            if frame_buffer:
                sorted_frames = sorted(
                    frame_buffer, key=lambda x: x.quality, reverse=True
                )
                for frame in sorted_frames[: self.picks_per_window]:
                    yield frame

    def get_cached_analysis(self) -> Optional[list[Frame]]:
        if not self.cache_file.exists():
            return None

        df = pl.read_parquet(self.cache_file)
        console.print("Using cached analysis", style="blue")

        frames = []
        for row in df.iter_rows(named=True):
            success, image = self.capture.read()
            if success:
                frames.append(
                    Frame(
                        index=row["index"],
                        timestamp=row["timestamp"],
                        quality=row["quality"],
                        image=image,
                    )
                )
        return frames

    def save_analysis(self, frames: list[Frame]) -> None:
        df = pl.DataFrame(
            [
                {k: v for k, v in asdict(frame).items() if k != "image"}
                for frame in frames
            ]
        )
        df.write_parquet(self.cache_file)

    def select_best_frames(self) -> list[Frame]:
        frames = self.get_cached_analysis()
        if frames is None:
            frames = list(self.process_frames())
            self.save_analysis(frames)

        if not frames:
            return []

        console.print(f"\nSelected {len(frames):,} frames", style="blue")
        return frames

    def __enter__(self) -> "VideoQualityAnalyzer":
        return self

    def __exit__(self, *args) -> None:
        self.capture.release()


@app.command()
def extract_frames(
    video_path: Path = typer.Argument(..., help="Path to input video file"),
    base_dir: Path = typer.Option("output", help="Base output directory"),
    window_duration: float = typer.Option(1.0, help="Window size in seconds"),
    picks_per_window: int = typer.Option(1, help="Number of frames to pick per window"),
) -> None:
    base_dir = Path(base_dir)
    frames_dir = base_dir / "frames"
    cache_dir = base_dir / "cache"

    for dir_path in [frames_dir, cache_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    with VideoQualityAnalyzer(
        video_path, window_duration, picks_per_window, cache_dir
    ) as analyzer:
        best_frames = analyzer.select_best_frames()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Saving frames...", total=len(best_frames))

        for frame in best_frames:
            output_path = (
                frames_dir / f"frame_{frame.index:06d}_{frame.quality:.0f}.jpg"
            )
            cv2.imwrite(str(output_path), frame.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            progress.advance(task)

    console.print(f"\nFrames saved to: {frames_dir.absolute()}", style="bold green")


if __name__ == "__main__":
    app()
