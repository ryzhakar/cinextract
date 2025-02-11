from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import open_clip  # Make sure you have open_clip installed
import polars as pl
import torch
from PIL import Image
from rich.console import Console

console = Console()

@dataclass(frozen=True)
class ExemplarScore:
    frame_path: Path
    score: float
    frame_index: int
    cluster_id: int  # Add cluster ID if you have it in your data
    cluster_probability: float  # Add cluster probability if you have it


class AestheticPredictor:
    def __init__(self, device: str = "mps") -> None:
        self.device = torch.device(device)

        self.clip_model = open_clip.create_model(  # Use the same CLIP model as your embeddings
            "ViT-L-14",
            pretrained="openai"
        ).to(self.device)

        weights_path = "ava.pth"  # Or the .pth file you want to use
        self.aesthetic_predictor_weights = torch.load(weights_path, map_location=self.device)

        self.fc = torch.nn.Linear(768, 1).to(self.device)  # Input size MUST match your embeddings (ViT-L-14 outputs 768)

        # Load weights:
        for k, v in self.aesthetic_predictor_weights.items():
            if "fc.weight" in k:
                self.fc.weight.data = v
            elif "fc.bias" in k:
                self.fc.bias.data = v

        self.fc.eval()

    def score_exemplars(self, embeddings_file: Path) -> Iterator[ExemplarScore]:
        df = pl.read_parquet(embeddings_file)  # Use polars to read Parquet

        for row in df.iter_rows(named=True):
            embedding = np.frombuffer(row["embedding"], dtype=np.float32).reshape(768) # Reshape to 768
            embedding = torch.from_numpy(embedding).to(self.device).unsqueeze(0)  # Correct way to create tensor

            with torch.no_grad():
                aesthetic_score = self.fc(embedding).item()

            yield ExemplarScore(
                frame_path=Path(row["frame_path"]),
                score=aesthetic_score,
                frame_index=row["frame_index"],
                cluster_id=row.get("cluster_id"),  # Use .get to handle missing columns
                cluster_probability=row.get("cluster_probability")
            )

    def save_results(self, scores: list[ExemplarScore], output_file: Path) -> None:
        df = pl.DataFrame([{
            "frame_path": str(s.frame_path),
            "aesthetic_score": s.score,
            "frame_index": s.frame_index,
            "cluster_id": s.cluster_id,
            "cluster_probability": s.cluster_probability
        } for s in scores])

        df.write_parquet(output_file)  # Use polars to write Parquet


if __name__ == "__main__":
    console.print("\nLoading Improved Aesthetic Predictor...", style="blue")
    predictor = AestheticPredictor()

    embeddings_file = Path("output/clip_embeddings.parquet")  # Your existing embeddings file
    output_file = Path("output/scored_exemplars.parquet")  # Output for scored exemplars

    console.print("Processing exemplar frames...", style="blue")
    scores = list(predictor.score_exemplars(embeddings_file))

    console.print(f"\nScored {len(scores)} exemplar frames", style="green")
    predictor.save_results(scores, output_file)  # Pass the output file path
    console.print(f"\nResults saved to: {output_file}", style="green")
