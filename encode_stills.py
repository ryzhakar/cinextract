from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import open_clip
import polars as pl
import torch
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from torch.utils.data import Dataset, DataLoader

console = Console()


@dataclass(frozen=True)
class FrameEmbedding:
    frame_path: Path
    embedding: np.ndarray
    frame_index: int
    quality_score: float


class FrameDataset(Dataset):
    def __init__(self, frames_dir: Path) -> None:
        self.frame_paths = sorted(
            frames_dir.glob("*.jpg"),
            key=lambda p: int(p.stem.split("_")[1])
        )
        self.model_name = "ViT-L-14"
        self.pretrained = "openai"
        _, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained
        )
    
    def __len__(self) -> int:
        return len(self.frame_paths)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        frame_path = self.frame_paths[idx]
        image = Image.open(frame_path).convert("RGB")
        return self.preprocess(image), str(frame_path)


class CLIPEncoder:
    def __init__(self, device: str = "mps") -> None:
        self.device = torch.device(device)
        self.model_name = "ViT-L-14"
        self.pretrained = "openai"
        self.model, _, _ = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device
        )
    
    def encode_frames(self, frames_dir: Path, batch_size: int = 32) -> Iterator[FrameEmbedding]:
        dataset = FrameDataset(frames_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Encoding frames...", total=len(dataset))
            
            with torch.no_grad():
                for batch_images, batch_paths in dataloader:
                    batch_images = batch_images.to(self.device)
                    embeddings = self.model.encode_image(batch_images)
                    
                    for embedding, frame_path in zip(embeddings, batch_paths):
                        frame_path = Path(frame_path)
                        frame_index = int(frame_path.stem.split("_")[1])
                        quality_score = float(frame_path.stem.split("_")[2])
                        
                        yield FrameEmbedding(
                            frame_path=frame_path,
                            embedding=embedding.cpu().numpy(),
                            frame_index=frame_index,
                            quality_score=quality_score
                        )
                    
                    progress.advance(task, batch_size)


def extract_embeddings(frames_dir: Path, output_dir: Path) -> None:
    encoder = CLIPEncoder()
    embeddings = []
    
    for embedding in encoder.encode_frames(frames_dir):
        embeddings.append({
            "frame_path": str(embedding.frame_path),
            "embedding": embedding.embedding.tobytes(),
            "frame_index": embedding.frame_index,
            "quality_score": embedding.quality_score
        })
    
    df = pl.DataFrame(embeddings)
    df.write_parquet(output_dir / "clip_embeddings.parquet")


if __name__ == "__main__":
    import typer
    
    def main(
        frames_dir: Path = typer.Argument(..., help="Directory containing extracted frames"),
        output_dir: Path = typer.Option("output", help="Output directory for embeddings")
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        extract_embeddings(frames_dir, output_dir)
    
    typer.run(main)
