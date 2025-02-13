import umap
import hdbscan
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import polars as pl


class SceneAnalyzer:
    def __init__(self):
        self.feature_pipeline = Pipeline(
            [
                ("pca", PCA(n_components=32)),
                ("scaler", StandardScaler()),
                (
                    "umap",
                    umap.UMAP(
                        n_components=3,
                        n_neighbors=15,
                        min_dist=0.05,
                        metric="cosine",
                        set_op_mix_ratio=0.618,
                    ),
                ),
            ]
        )

        self.clusterer = hdbscan.HDBSCAN(
            min_samples=5,
            cluster_selection_epsilon=0.618,
            prediction_data=True,
        )

    def analyze_scenes(
        self,
        embeddings: np.ndarray,
        frame_idx: np.ndarray,
    ) -> tuple[pl.Series, pl.Series]:
        reduced = self.feature_pipeline.fit_transform(embeddings)
        temporal_features = np.expand_dims(
            frame_idx / max(len(embeddings), frame_idx.max()), axis=1
        )
        combined_features = np.hstack([reduced, temporal_features])

        cluster_labels = self.clusterer.fit(combined_features)
        probabilities = hdbscan.all_points_membership_vectors(self.clusterer)

        return (
            pl.Series(
                "cluster_id",
                cluster_labels.labels_,
                dtype=pl.Int32,
            ),
            pl.Series(
                "cluster_probability",
                probabilities.max(axis=1),
                dtype=pl.Float32,
            ),
        )
