import numpy as np
from pathlib import Path
import polars as pl
import torch
import open_clip

import urllib.request
from rich.console import Console

AESTHETIC_PREDICTOR_WEIGHTS_PATH = Path("laion_weights.pth")
AESTHETIC_PREDICTOR_WEIGHTS_URL = (
    "https://github.com/christophschuhmann/"
    "improved-aesthetic-predictor/raw/main/"
    "sac%2Blogos%2Bava1-l14-linearMSE.pth"
)


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


class AestheticScorer:
    def __init__(
        self,
        weights_path: Path,
        device: str,
    ):
        self.device = torch.device(device)
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self.model = model.to(self.device)
        self.fc = torch.nn.Linear(768, 1).to(self.device)

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
