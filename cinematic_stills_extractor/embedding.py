import itertools
from pathlib import Path
from collections.abc import Iterable
from PIL import Image
import open_clip
import cv2
import polars as pl
import torch
import numpy as np
from rich.progress import (
    Progress,
)


def extract_specific_frame_from(
    capture,
    frame_idx: int,
) -> Image.Image:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = capture.read()
    if success:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    raise ValueError(f"Failed to extract frame {frame_idx}")


class EmbeddingGenerator:
    def __init__(self, device: str):
        self.device = torch.device(device)
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self.model = model.to(self.device)

    def generate_embeddings(
        self,
        image_stream: Iterable[Image.Image],
        *,
        total_frames_count: int,
        batch_size: int = 256,
        progress_keeper: Progress,
    ) -> np.ndarray:
        stream = iter(image_stream)
        embeddings = []
        task = progress_keeper.add_task(
            "Generating embeddings...",
            total=total_frames_count,
        )

        def preprocess_batch_from(
            stream: Iterable[Image.Image],
        ) -> torch.Tensor:
            return torch.stack(
                [
                    self.preprocess(frame)
                    for frame in itertools.islice(stream, batch_size)
                ]
            )

        with torch.no_grad():
            for _ in range(total_frames_count // batch_size + 1):
                batch_embeddings = self.model.encode_image(
                    preprocess_batch_from(stream).to(self.device)
                )
                embeddings.append(batch_embeddings.cpu().numpy())

                progress_keeper.advance(task, batch_size)

        return np.vstack(embeddings)


def generate_embeddings_from(
    video_path: Path,
    *,
    frame_df: pl.DataFrame,
    progress_keeper: Progress,
    device: str,
) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    frame_indices = frame_df["frame_idx"].to_list()
    image_stream = iter(
        extract_specific_frame_from(
            capture,
            frame_idx=index,
        )
        for index in frame_indices
    )
    embeddings = EmbeddingGenerator(device).generate_embeddings(
        image_stream,
        total_frames_count=len(frame_indices),
        progress_keeper=progress_keeper,
    )
    capture.release()
    return embeddings
