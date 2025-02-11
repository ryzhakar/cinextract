import umap
import hdbscan
import polars as pl
import numpy as np


class SceneAnalyzer:
    def __init__(self):
        self.umap = umap.UMAP(
            n_components=7,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidian",
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
