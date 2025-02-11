from dataclasses import dataclass
from pathlib import Path

import hdbscan
import numpy as np
import polars as pl
from rich.console import Console

console = Console()


@dataclass(frozen=True)
class SceneClusterer:
    min_cluster_size: int = 5
    min_samples: int = 3
    cluster_selection_epsilon: float = 0.5

    def cluster_scenes(self) -> None:
        embeddings = (
            pl.scan_parquet(Path("output") / "reduced_embeddings.parquet")
            .filter(pl.col("isolation_score") == 1)
            .collect()
            .with_columns(pl.col("quality_score").cast(pl.Float32))
        )

        coordinates = np.stack(
            [np.frombuffer(c, dtype=np.float32) for c in embeddings["coordinates"]]
        )

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            prediction_data=True,
        )

        cluster_labels = clusterer.fit(coordinates)
        probabilities = hdbscan.all_points_membership_vectors(clusterer)

        exemplar_mask = np.zeros_like(cluster_labels.labels_, dtype=bool)
        for label in np.unique(cluster_labels.labels_):
            if label != -1:
                cluster_points = coordinates[cluster_labels.labels_ == label]
                centroid = cluster_points.mean(axis=0)
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                exemplar_idx = np.where(cluster_labels.labels_ == label)[0][
                    distances.argmin()
                ]
                exemplar_mask[exemplar_idx] = True

        clustered_inliers = embeddings.with_columns(
            [
                pl.Series("cluster_id", cluster_labels.labels_).cast(pl.Int64),
                pl.Series(
                    "cluster_probability", probabilities.max(axis=1).astype(np.float32)
                ),
                pl.Series("is_exemplar", exemplar_mask),
            ]
        )

        clustered_outliers = (
            pl.scan_parquet(Path("output") / "reduced_embeddings.parquet")
            .filter(pl.col("isolation_score") != 1)
            .with_columns(
                [
                    pl.lit(-1).cast(pl.Int64).alias("cluster_id"),
                    pl.lit(0.0).cast(pl.Float32).alias("cluster_probability"),
                    pl.lit(False).alias("is_exemplar"),
                    pl.col("quality_score").cast(pl.Float32),
                ]
            )
            .collect()
        )

        (
            pl.concat([clustered_inliers, clustered_outliers])
            .sort("frame_index")
            .write_parquet(Path("output") / "clustered_frames.parquet")
        )


if __name__ == "__main__":
    clusterer = SceneClusterer()
    clusterer.cluster_scenes()
