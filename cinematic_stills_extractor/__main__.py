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


from cinematic_stills_extractor.scene import SceneAnalyzer
from cinematic_stills_extractor.still_frames import extract_best_frames_from
from cinematic_stills_extractor.embedding import generate_embeddings_from
from cinematic_stills_extractor.aesthetic_predictor import (
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
    picks_per_window: int = 2,
    device: str = setup_device(),
    console: Console,
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
            picks_per_window=picks_per_window,
            progress_keeper=progress,
        )
        embeddings = generate_embeddings_from(
            video_path,
            frame_df=frame_df,
            device=device,
            progress_keeper=progress,
        )

    analyzer = SceneAnalyzer()
    cluster_labels, probabilities = analyzer.analyze_scenes(
        embeddings, frame_df["timestamp"].to_numpy()
    )

    scene_df = frame_df.with_columns(
        [
            pl.Series("cluster_id", cluster_labels).cast(pl.Int32),
            pl.Series("cluster_probability", probabilities.max(axis=1)).cast(
                pl.Float32
            ),
        ]
    )

    # Filter out noise points but keep all meaningful clusters
    scene_df = scene_df.filter(pl.col("cluster_id") >= -1)
    cluster_masks = scene_df["cluster_id"] >= 0
    scene_df = scene_df.filter(cluster_masks)
    embeddings = embeddings[cluster_masks.to_list()]

    # Score ALL frames, not just exemplars
    scorer = AestheticScorer(
        ensure_weights(console),
        device=device,
    )
    scene_df = scorer.score_exemplars(scene_df, embeddings)

    # Find exemplars that are close to cluster centroid AND have high aesthetic score
    scene_df = analyzer.find_exemplars(scene_df, embeddings)

    # Compute and display processing statistics
    processing_time = time() - start_time
    cluster_count = len(scene_df["cluster_id"].unique())
    exemplar_count = len(scene_df.filter(pl.col("is_exemplar")))

    console.print(
        f"[bold green]Processed {len(scene_df)} frames across {cluster_count} clusters[/bold green]"
    )
    console.print(
        f"[bold yellow]Processing time: {processing_time:.2f} seconds[/bold yellow]"
    )
    console.print(f"[bold yellow]Exemplar frames: {exemplar_count}[/bold yellow]")

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

    exemplar_count = len(df.filter(pl.col("is_exemplar")))
    top_n = max(min_export_count, int(max(1, exemplar_count * top_percentile / 100)))

    best_frames = (
        df.filter(pl.col("is_exemplar"))
        .sort("aesthetic_score", descending=True)
        .head(top_n)
    )

    if len(best_frames) == 0:
        # Fallback: export top frames by default if no exemplars found
        best_frames = df.sort("cluster_probability", descending=True).head(top_n)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    console.print(
        f"[bold yellow]Exporting top {top_n} frames to {output_dir}[/bold yellow]"
    )

    exported_count = 0
    scores = []
    for row in best_frames.iter_rows(named=True):
        capture.set(cv2.CAP_PROP_POS_FRAMES, row["frame_idx"])
        success, frame = capture.read()
        if not success:
            continue

        # Truncate aesthetic score to 3 decimal places for filename
        aesthetic_score = row.get("aesthetic_score", 0.0)
        cluster_prob = row.get("cluster_probability", 0.0)

        output_path = (
            output_dir
            / f"frame_{row['frame_idx']:06d}_a{aesthetic_score:.3f}_p{cluster_prob:.3f}.jpg"
        )
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        scores.append(aesthetic_score)
        exported_count += 1

    capture.release()

    if exported_count == 0:
        console.print(
            "[bold red]No frames could be exported. Check video file.[/bold red]"
        )
    else:
        console.print(
            f"[bold green]Exported {exported_count} frames successfully![/bold green]"
        )
        console.print(
            f"[yellow]Aesthetic Scores: min={min(scores):.3f}, max={max(scores):.3f}, avg={sum(scores) / len(scores):.3f}[/yellow]"
        )


def main(
    video_path: Path = typer.Argument(..., help="Input video file"),
    window_sec: float = typer.Option(2.0, help="Analysis window in seconds"),
    picks_per_window: int = typer.Option(2, help="Frames to pick per window"),
    device: str = typer.Option(setup_device(), help="Device for ML models"),
    top_percentile: float = typer.Option(
        10.0, help="Top percentage of exemplars to export"
    ),
    output_dir: Path = typer.Option(Path("output"), help="Output directory"),
    min_export_count: int = typer.Option(1, help="Minimum number of frames to export"),
) -> None:
    console = Console()
    results = process_video(
        video_path,
        window_sec=window_sec,
        picks_per_window=picks_per_window,
        device=device or setup_device(),
        console=console,
    )
    export_best_frames(
        video_path,
        results,
        output_dir,
        top_percentile=top_percentile,
        min_export_count=min_export_count,
        console=console,
    )


typer.run(main)
