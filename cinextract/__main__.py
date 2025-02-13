from pathlib import Path
import cv2
import polars as pl
import torch
from time import time
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)
import typer

app = typer.Typer()

from cinextract.scene import SceneAnalyzer
from cinextract.still_frames import extract_best_frames_from
from cinextract.embedding import generate_embeddings_from
from cinextract.aesthetic_predictor import (
    AestheticScorer,
    ensure_weights,
)


def setup_device() -> str:
    """Determine the most appropriate device for computation."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def process_video(
    video_path: Path,
    *,
    window_sec: float = 2.0,
    device: str,
    embedding_batch_size: int,
    console: Console,
    patience_factor: int,
) -> pl.DataFrame:
    start_time = time()

    console.print(f"[bold green]Processing video: {video_path}[/bold green]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(compact=True),
    ) as progress:
        frame_df = extract_best_frames_from(
            video_path,
            window_sec=window_sec,
            progress_keeper=progress,
            patience_factor=patience_factor,
        )
        embeddings = generate_embeddings_from(
            video_path,
            frame_df=frame_df,
            device=device,
            progress_keeper=progress,
            batch_size=embedding_batch_size,
        )

    cluster_labels, probabilities = SceneAnalyzer().analyze_scenes(
        embeddings, frame_df["frame_idx"].to_numpy()
    )
    ae_scores = AestheticScorer(ensure_weights(console), device=device).score(
        embeddings
    )
    scene_df = pl.concat(
        (
            frame_df,
            ae_scores.to_frame(),
            cluster_labels.to_frame(),
            probabilities.to_frame(),
        ),
        how="horizontal",
    )

    # Compute and display processing statistics
    processing_time = time() - start_time
    cluster_count = len(scene_df["cluster_id"].unique() - 1)

    console.print(
        f"[bold green]Processed {len(scene_df)} frames across {cluster_count} clusters[/bold green]"
    )
    console.print(
        f"[bold yellow]Processing time: {processing_time:.2f} seconds[/bold yellow]"
    )

    return scene_df


def export_best_frames(
    video_path: Path,
    df: pl.DataFrame,
    output_dir: Path,
    *,
    top_percentile: float = 10.0,
    min_export_count: int = 10,
    console: Console,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    best_frames = (
        df.filter(pl.col("cluster_id") >= 0)
        .group_by("cluster_id")
        .agg(
            pl.all().first(),
            prob_threshold=pl.col("cluster_probability").quantile(0.2),
        )
        .filter(pl.col("cluster_probability") >= pl.col("prob_threshold"))
        .sort(["cluster_id", "aesthetic_score"], descending=[False, True])
        .group_by("cluster_id", maintain_order=True)
        .first()
        .sort("aesthetic_score", descending=True)
        .head(
            max(
                min_export_count,
                int(len(df["cluster_id"].unique()) * top_percentile / 100),
            )
        )
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    console.print(
        f"[bold yellow]Exporting {len(best_frames)} frames to {output_dir}[/bold yellow]"
    )

    for frame_data in best_frames.iter_rows(named=True):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_data["frame_idx"])
        success, frame = capture.read()
        if not success:
            continue

        output_path = (
            output_dir
            / f"frame_{frame_data['frame_idx']:06d}_c{frame_data['cluster_id']}_a{frame_data['aesthetic_score']:.3f}.jpg"
        )
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    capture.release()

    console.print(
        f"[bold green]Exported {len(best_frames)} frames from {len(best_frames['cluster_id'].unique())} clusters[/bold green]\n"
        f"[yellow]Aesthetic Scores: {best_frames['aesthetic_score'].describe()}[/yellow]"
    )


@app.command()
def main(
    video_path: Path = typer.Argument(..., help="Input video file"),
    output_dir: Path = typer.Option(Path("output"), help="Output directory"),
    device: str = typer.Option(setup_device(), help="Device for ML models"),
    top_percentile: float = typer.Option(
        10.0, help="Top percentage of frames to export"
    ),
    min_export_count: int = typer.Option(10, help="Minimum number of frames to export"),
    window_sec: float = typer.Option(2.0, help="Analysis window in seconds"),
    embedding_batch_size: int = 256,
    patience_factor: int = 4,
) -> None:
    console = Console()
    results = process_video(
        video_path,
        window_sec=window_sec,
        device=device or setup_device(),
        console=console,
        embedding_batch_size=embedding_batch_size,
        patience_factor=patience_factor,
    )
    export_best_frames(
        video_path,
        results,
        output_dir,
        top_percentile=top_percentile,
        min_export_count=min_export_count,
        console=console,
    )


if __name__ == "__main__":
    app()
