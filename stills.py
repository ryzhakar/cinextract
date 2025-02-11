import itertools
from collections.abc import Iterable
import math
import numpy as np
import cv2
import polars as pl
from pathlib import Path
from typing import NamedTuple, Optional
import torch
import open_clip
from PIL import Image
import umap
import hdbscan
import urllib.request
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


FRAME_SIZE = 320

AESTHETIC_PREDICTOR_WEIGHTS_PATH = Path("laion_weights.pth")
AESTHETIC_PREDICTOR_WEIGHTS_URL = (
    "https://github.com/christophschuhmann/"
    "improved-aesthetic-predictor/raw/main/"
    "sac%2Blogos%2Bava1-l14-linearMSE.pth"
)


class Frame(NamedTuple):
    index: int
    timestamp: float
    quality: float


def setup_device() -> str:
    """Determine the most appropriate device for computation."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def ensure_weights(console: Console) -> Path:
    """Download pre-trained weights if not existing."""
    if not AESTHETIC_PREDICTOR_WEIGHTS_PATH.exists():
        console.print(
            "[yellow]Downloading aesthetic predictor weights...[/yellow]",
        )
        urllib.request.urlretrieve(
            AESTHETIC_PREDICTOR_WEIGHTS_URL,
            AESTHETIC_PREDICTOR_WEIGHTS_PATH,
        )
    return AESTHETIC_PREDICTOR_WEIGHTS_PATH


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
    picks_per_window: int = 2,
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
            Frame(
                index=idx,
                timestamp=idx / fps,
                quality=frame_quality_from(frame)
            )
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

    frame_idx, timestamps, quality_scores = zip(*itertools.chain.from_iterable(frame_keeper))

    capture.release()
    progress_keeper.update(task, completed=True)
    return pl.concat(
        (
            pl.Series(
                'frame_idx',
                frame_idx,
                dtype=pl.Int32,
            ).to_frame(),
            pl.Series(
                'timestamp',
                timestamps,
                dtype=pl.Float32,
            ).to_frame(),
            pl.Series(
                'quality_score',
                quality_scores,
                dtype=pl.Float32,
            ).to_frame(),
        ),
        how='horizontal',
    )

class EmbeddingGenerator:
    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device or setup_device())
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self.model = model.to(self.device)

    def extract_frame(self, video_path: Path, frame_idx: int) -> Image.Image:
        capture = cv2.VideoCapture(str(video_path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = capture.read()
        capture.release()

        if not success:
            raise ValueError(f"Failed to extract frame {frame_idx}")

        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def generate_embeddings(
        self,
        video_path: Path,
        frame_indices: list[int],
        batch_size: int = 256,
    ) -> np.ndarray:
        embeddings = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                "Generating embeddings...", total=len(frame_indices)
            )

            with torch.no_grad():
                for i in range(0, len(frame_indices), batch_size):
                    batch_indices = frame_indices[i : i + batch_size]

                    # Provide a hint about current progress
                    console.print(
                        f"[yellow]Processing batch {i // batch_size + 1}/{math.ceil(len(frame_indices) / batch_size)}: "
                        f"{len(batch_indices)} frames[/yellow]"
                    )

                    try:
                        batch_images = torch.stack(
                            [
                                self.preprocess(self.extract_frame(video_path, idx))
                                for idx in batch_indices
                            ]
                        ).to(self.device)

                        batch_embeddings = self.model.encode_image(batch_images)
                        embeddings.append(batch_embeddings.cpu().numpy())

                        progress.advance(task, len(batch_indices))

                    except Exception as e:
                        console.print(
                            f"[bold red]Error processing batch: {e}[/bold red]"
                        )
                        raise

        return np.vstack(embeddings)


class SceneAnalyzer:
    def __init__(self):
        self.umap = umap.UMAP(
            n_components=3,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        self.clusterer = hdbscan.HDBSCAN(
            min_samples=3, cluster_selection_epsilon=0.5, prediction_data=True
        )

    def analyze_scenes(
        self, embeddings: np.ndarray, timestamps: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        temporal_features = np.expand_dims(timestamps, axis=1)
        combined_features = np.hstack([embeddings, temporal_features])

        reduced = self.umap.fit_transform(combined_features)
        cluster_labels = self.clusterer.fit(reduced)
        probabilities = hdbscan.all_points_membership_vectors(self.clusterer)

        return cluster_labels.labels_, probabilities

    def find_exemplars(self, df: pl.DataFrame, embeddings: np.ndarray) -> pl.DataFrame:
        """Find representative frames for each cluster with combined similarity and aesthetic criteria."""

        def find_cluster_exemplar(
            cluster_mask: np.ndarray, aesthetic_scores: np.ndarray
        ) -> int:
            cluster_embs = embeddings[cluster_mask]
            cluster_scores = aesthetic_scores[cluster_mask]
            centroid = cluster_embs.mean(axis=0)

            # Compute cosine similarities to find most representative frame
            similarities = np.dot(cluster_embs, centroid) / (
                np.linalg.norm(cluster_embs, axis=1) * np.linalg.norm(centroid)
            )

            # Normalize aesthetic scores
            normalized_scores = (cluster_scores - cluster_scores.min()) / (
                cluster_scores.max() - cluster_scores.min()
            )

            # Combined score: balance between centroid proximity and aesthetic quality
            combined_scores = 0.5 * similarities + 0.5 * normalized_scores

            # Select frame with highest combined score as exemplar
            return np.argmax(combined_scores)

        # Track exemplars with high uniqueness and aesthetic value
        exemplars = []
        aesthetic_scores = df["aesthetic_score"].to_numpy()

        for cluster_id in df["cluster_id"].unique():
            if cluster_id < 0:
                continue

            cluster_mask = (df["cluster_id"] == cluster_id).to_list()
            exemplar_idx = find_cluster_exemplar(cluster_mask, aesthetic_scores)
            exemplars.append(cluster_mask.index(True) + exemplar_idx)

        return df.with_columns(
            pl.Series("is_exemplar", [i in exemplars for i in range(len(df))])
        )


class AestheticScorer:
    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device or setup_device())
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self.model = model.to(self.device)
        self.fc = torch.nn.Linear(768, 1).to(self.device)

        weights_path = ensure_weights()
        weights = torch.load(weights_path, map_location=self.device)

        for k, v in weights.items():
            if "fc.weight" in k:
                self.fc.weight.data = v
            elif "fc.bias" in k:
                self.fc.bias.data = v

        self.fc.eval()

    def score_exemplars(self, df: pl.DataFrame, embeddings: np.ndarray) -> pl.DataFrame:
        with torch.no_grad():
            # Score ALL embeddings
            full_embeddings = torch.from_numpy(embeddings).to(self.device)
            all_scores = self.fc(full_embeddings).cpu().numpy().flatten()

        return df.with_columns(
            pl.Series("aesthetic_score", all_scores).cast(pl.Float32)
        )


def process_video(
    video_path: Path,
    window_sec: float = 2.0,
    picks_per_window: int = 2,
    device: Optional[str] = None,
) -> pl.DataFrame:
    console = Console()
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
    total_frames = len(frame_df)
    console.print(f"[yellow]Total frames processed: {total_frames}[/yellow]")

    embedding_gen = EmbeddingGenerator(device)
    embeddings = embedding_gen.generate_embeddings(
        video_path, frame_df["frame_idx"].to_list()
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
    scorer = AestheticScorer(device)
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
    top_percentile: float = 10.0,
    min_export_count: int = 10,
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


if __name__ == "__main__":
    import typer

    def main(
        video_path: Path = typer.Argument(..., help="Input video file"),
        window_sec: float = typer.Option(2.0, help="Analysis window in seconds"),
        picks_per_window: int = typer.Option(2, help="Frames to pick per window"),
        device: Optional[str] = typer.Option(None, help="Device for ML models"),
        top_percentile: float = typer.Option(
            10.0, help="Top percentage of exemplars to export"
        ),
        output_dir: Path = typer.Option(Path("output"), help="Output directory"),
        min_export_count: int = typer.Option(
            1, help="Minimum number of frames to export"
        ),
    ) -> None:
        results = process_video(video_path, window_sec, picks_per_window, device)
        export_best_frames(
            video_path, results, output_dir, top_percentile, min_export_count
        )

    typer.run(main)
