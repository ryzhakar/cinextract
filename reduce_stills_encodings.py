from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
from rich.console import Console
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from umap import UMAP

console = Console()


@dataclass(frozen=True)
class EmbeddingData:
    frame_paths: Sequence[Path]
    embeddings: np.ndarray
    frame_indices: Sequence[int]
    quality_scores: Sequence[float]


@dataclass(frozen=True)
class ReducedEmbedding:
    frame_path: Path
    coordinates: np.ndarray
    frame_index: int
    quality_score: float
    isolation_score: float


class DimensionalityReducer:
    def __init__(
        self,
        n_components: int = 31,
        random_state: int = 42,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        contamination: float = 0.1,
    ) -> None:
        self.outlier_detector = IsolationForest(
            contamination=contamination, random_state=random_state
        )
        self.pipeline = Pipeline(
            [
                (
                    "pca",
                    PCA(
                        n_components=min(256, n_components * 2),
                        random_state=random_state,
                    ),
                ),
                ("scaler", StandardScaler()),
                (
                    "umap",
                    UMAP(
                        n_components=n_components,
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def find_embeddings_file(self, search_dir: Path = Path("output")) -> Path:
        embeddings_path = search_dir / "clip_embeddings.parquet"
        if not embeddings_path.exists():
            raise FileNotFoundError(f"No embeddings file found at {embeddings_path}")
        return embeddings_path

    def prepare_feature_matrix(self, data: EmbeddingData) -> np.ndarray:
        sequence_features = np.array(data.frame_indices).reshape(-1, 1)
        return np.hstack(
            [data.embeddings, StandardScaler().fit_transform(sequence_features)]
        )

    def reduce_dimensions(self) -> list[ReducedEmbedding]:
        embeddings_path = self.find_embeddings_file()
        data = self.load_embeddings(embeddings_path)

        feature_matrix = self.prepare_feature_matrix(data)
        isolation_scores = self.outlier_detector.fit_predict(feature_matrix)
        reduced_coordinates = self.pipeline.fit_transform(feature_matrix)

        return [
            ReducedEmbedding(
                frame_path=frame_path,
                coordinates=coordinates,
                frame_index=frame_index,
                quality_score=quality_score,
                isolation_score=float(isolation_score),
            )
            for frame_path, coordinates, frame_index, quality_score, isolation_score in zip(
                data.frame_paths,
                reduced_coordinates,
                data.frame_indices,
                data.quality_scores,
                isolation_scores,
            )
        ]

    def load_embeddings(self, embeddings_path: Path) -> EmbeddingData:
        df = pl.read_parquet(embeddings_path)
        return EmbeddingData(
            frame_paths=[Path(p) for p in df["frame_path"]],
            embeddings=np.stack(
                [np.frombuffer(e, dtype=np.float32) for e in df["embedding"]]
            ),
            frame_indices=df["frame_index"].to_list(),
            quality_scores=df["quality_score"].to_list(),
        )

    def save_results(self, reduced_embeddings: list[ReducedEmbedding]) -> None:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        df = pl.DataFrame(
            [
                {
                    "frame_path": str(e.frame_path),
                    "coordinates": e.coordinates.tobytes(),
                    "frame_index": e.frame_index,
                    "quality_score": e.quality_score,
                    "isolation_score": e.isolation_score,
                }
                for e in reduced_embeddings
            ]
        )

        df.write_parquet(output_dir / "reduced_embeddings.parquet")


if __name__ == "__main__":
    reducer = DimensionalityReducer()
    reduced = reducer.reduce_dimensions()
    reducer.save_results(reduced)
