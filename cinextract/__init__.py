__version__ = "0.1.0"

from cinextract.scene import SceneAnalyzer
from cinextract.still_frames import extract_best_frames_from
from cinextract.embedding import generate_embeddings_from
from cinextract.aesthetic_predictor import AestheticScorer

__all__ = [
    "SceneAnalyzer",
    "extract_best_frames_from",
    "generate_embeddings_from",
    "AestheticScorer",
]
