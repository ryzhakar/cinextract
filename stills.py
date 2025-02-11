import numpy as np
import cv2
import polars as pl
from pathlib import Path
from typing import NamedTuple
import torch
import open_clip
from PIL import Image
import umap
import hdbscan
import urllib.request
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

class Frame(NamedTuple):
    index: int
    timestamp: float
    quality: float

def setup_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"

def ensure_weights() -> Path:
    weights_path = Path("laion_weights.pth")
    if not weights_path.exists():
        url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
        urllib.request.urlretrieve(url, weights_path)
    return weights_path

class FrameExtractor:
    def analyze_frame(self, frame: np.ndarray) -> float:
        frame_size = 320
        frame = cv2.resize(frame, (frame_size, frame_size))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        blur_measure = cv2.Laplacian(gray, cv2.CV_64F).var()
        luma_std = gray.std()
        return float(0.7 * blur_measure + 0.3 * luma_std)

    def extract_best_frames(self, video_path: Path, window_sec: float = 1.0, picks_per_window: int = 1) -> pl.DataFrame:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        window_frames = int(window_sec * fps)

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn()) as progress:
            task = progress.add_task("Extracting frames...", total=frame_count)
            frame_buffer = []
            best_frames = []
            
            for frame_idx in range(frame_count):
                success, frame = capture.read()
                if not success:
                    break

                quality = self.analyze_frame(frame)
                frame_buffer.append(Frame(
                    index=frame_idx,
                    timestamp=frame_idx / fps,
                    quality=quality
                ))

                if len(frame_buffer) >= window_frames:
                    best_frames.extend(
                        sorted(frame_buffer, key=lambda x: x.quality, reverse=True)[:picks_per_window]
                    )
                    frame_buffer.clear()

                progress.advance(task)

            if frame_buffer:
                best_frames.extend(
                    sorted(frame_buffer, key=lambda x: x.quality, reverse=True)[:picks_per_window]
                )

        capture.release()
        
        return pl.DataFrame({
            "frame_idx": [f.index for f in best_frames],
            "timestamp": [f.timestamp for f in best_frames],
            "quality_score": [f.quality for f in best_frames]
        }).with_columns([
            pl.col("frame_idx").cast(pl.Int32),
            pl.col("timestamp").cast(pl.Float32),
            pl.col("quality_score").cast(pl.Float32)
        ])

class EmbeddingGenerator:
    def __init__(self, device: str = setup_device()):
        self.device = torch.device(device)
        model, _, self.preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        self.model = model.to(self.device)

    def extract_frame(self, video_path: Path, frame_idx: int) -> Image.Image:
        capture = cv2.VideoCapture(str(video_path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = capture.read()
        capture.release()
        
        if not success:
            raise ValueError(f"Failed to extract frame {frame_idx}")
            
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def generate_embeddings(self, video_path: Path, frame_indices: list[int], batch_size: int = 32) -> np.ndarray:
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(frame_indices), batch_size):
                batch_indices = frame_indices[i:i + batch_size]
                batch_images = torch.stack([
                    self.preprocess(self.extract_frame(video_path, idx))
                    for idx in batch_indices
                ]).to(self.device)
                
                batch_embeddings = self.model.encode_image(batch_images)
                embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

class SceneAnalyzer:
    def __init__(self):
        self.umap = umap.UMAP(
            n_components=32,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=5,
            min_samples=3,
            cluster_selection_epsilon=0.5,
            prediction_data=True
        )

    def analyze_scenes(self, embeddings: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        temporal_features = np.expand_dims(timestamps, axis=1)
        combined_features = np.hstack([embeddings, temporal_features])
        
        reduced = self.umap.fit_transform(combined_features)
        cluster_labels = self.clusterer.fit(reduced)
        probabilities = hdbscan.all_points_membership_vectors(self.clusterer)
        
        return cluster_labels.labels_, probabilities

    def find_exemplars(self, df: pl.DataFrame, embeddings: np.ndarray) -> pl.DataFrame:
        centroids = {}
        for cluster_id in df['cluster_id'].unique():
            if cluster_id < 0:
                continue
                
            mask = df['cluster_id'] == cluster_id
            cluster_embeddings = embeddings[mask.to_list()]
            centroid = cluster_embeddings.mean(axis=0)
            
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            exemplar_idx = distances.argmin()
            
            centroids[cluster_id] = mask.to_list().index(True) + exemplar_idx
            
        return df.with_columns(
            pl.Series('is_exemplar', [i in centroids.values() for i in range(len(df))])
        )

class AestheticScorer:
    def __init__(self, device: str = "mps" if torch.backends.mps.is_available() else "cpu"):
        self.device = torch.device(device)
        model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        self.model = model.to(self.device)
        self.fc = torch.nn.Linear(768, 1).to(self.device)
        
        weights_path = ensure_weights()
        weights = torch.load(weights_path, map_location=self.device)
        
        # Map weights properly
        for k, v in weights.items():
            if "fc.weight" in k:
                self.fc.weight.data = v
            elif "fc.bias" in k:
                self.fc.bias.data = v

        self.fc.eval()

    def score_exemplars(self, df: pl.DataFrame, exemplar_embeddings: np.ndarray) -> pl.DataFrame:
        with torch.no_grad():
            embeddings = torch.from_numpy(exemplar_embeddings).to(self.device)
            scores = self.fc(embeddings).cpu().numpy().flatten()
            
        exemplar_scores = dict(zip(
            df.filter(pl.col("is_exemplar"))["frame_idx"].to_list(),
            scores
        ))
        
        return df.with_columns(
            pl.Series(
                'aesthetic_score',
                [exemplar_scores.get(idx, 0.0) for idx in df["frame_idx"].to_list()]
            ).cast(pl.Float32)
        )

def process_video(
    video_path: Path,
    window_sec: float = 1.0,
    picks_per_window: int = 1,
    device: str = setup_device()
) -> pl.DataFrame:
    extractor = FrameExtractor()
    frame_df = extractor.extract_best_frames(video_path, window_sec, picks_per_window)
    
    embedding_gen = EmbeddingGenerator(device)
    embeddings = embedding_gen.generate_embeddings(
        video_path, 
        frame_df["frame_idx"].to_list()
    )
    
    analyzer = SceneAnalyzer()
    cluster_labels, probabilities = analyzer.analyze_scenes(
        embeddings,
        frame_df["timestamp"].to_numpy()
    )
    
    # Filter both DataFrame and embeddings together
    scene_df = frame_df.with_columns([
        pl.Series("cluster_id", cluster_labels).cast(pl.Int32),
        pl.Series("cluster_probability", probabilities.max(axis=1)).cast(pl.Float32)
    ])
    mask = scene_df["cluster_id"] >= 0
    scene_df = scene_df.filter(mask)
    embeddings = embeddings[mask.to_list()]
        
    # Now lengths will match
    scene_df = analyzer.find_exemplars(scene_df, embeddings)
    
    scorer = AestheticScorer(device)
    exemplar_mask = scene_df["is_exemplar"].to_list()
    exemplar_embeddings = embeddings[exemplar_mask]
    scene_df = scorer.score_exemplars(scene_df, exemplar_embeddings)
    
    return scene_df

def export_best_frames(
    video_path: Path,
    df: pl.DataFrame,
    output_dir: Path,
    top_percentile: float = 10.0
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exemplar_count = len(df.filter(pl.col("is_exemplar")))
    top_n = int(exemplar_count * top_percentile / 100)
    
    best_frames = (df
        .filter(pl.col("is_exemplar"))
        .sort("aesthetic_score", descending=True)
        .head(top_n)
    )
    
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
        
    for row in best_frames.iter_rows(named=True):
        capture.set(cv2.CAP_PROP_POS_FRAMES, row["frame_idx"])
        success, frame = capture.read()
        if not success:
            continue
            
        output_path = output_dir / f"frame_{row['frame_idx']:06d}_{row['aesthetic_score']:.2f}.jpg"
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
    capture.release()

if __name__ == "__main__":
    import typer
    
    def main(
        video_path: Path = typer.Argument(..., help="Input video file"),
        window_sec: float = typer.Option(1.0, help="Analysis window in seconds"),
        picks_per_window: int = typer.Option(1, help="Frames to pick per window"),
        device: str = typer.Option(setup_device(), help="Device for ML models"),
        top_percentile: float = typer.Option(10.0, help="Top percentage of exemplars to export"),
        output_dir: Path = typer.Option(Path("output"), help="Output directory")
    ) -> None:
        results = process_video(video_path, window_sec, picks_per_window, device)
        export_best_frames(video_path, results, output_dir, top_percentile)
    
    typer.run(main)
