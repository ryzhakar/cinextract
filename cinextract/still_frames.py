from pathlib import Path
from rich.progress import (
    Progress,
)
import cv2
import polars as pl
from dataclasses import dataclass
import numpy as np
from cinextract.cache import get_cache_path_from

FRAME_SIZE = 320


@dataclass
class FrameSelectionConfig:
    # Window parameters
    local_window_size: int = 30  # Window for local quality comparison
    min_gap: int = 3  # Minimum frames between selections
    max_gap: int = 60  # Maximum frames before forcing selection

    # Quality thresholds
    local_max_ratio: float = 0.95  # Must be this close to local max
    local_mean_ratio: float = 1.2  # Must exceed local mean by this factor

    # Dynamic threshold adjustments
    quality_boost_per_gap: float = 0.01  # How much to lower threshold per frame gap
    max_gap_adjustment: float = 0.3  # Maximum threshold reduction from gap
    quality_penalty_after_selection: float = (
        0.3  # How much to raise threshold after selection
    )
    penalty_decay_rate: float = 0.618  # How quickly the penalty decays per frame
    min_quality_threshold: float = 0.0  # Absolute minimum threshold


def select_frames(
    frames: pl.DataFrame, config: FrameSelectionConfig = FrameSelectionConfig()
) -> pl.DataFrame:
    """
    Select high quality frames while maintaining good temporal distribution.

    Args:
        frames: DataFrame with columns 'frame_idx' and 'quality' (0-1 normalized)
        config: Selection configuration parameters

    Returns:
        DataFrame with selected frames and their selection metadata
    """
    # Compute local quality metrics using rolling windows
    with_metrics = frames.with_columns(
        [
            # Local max quality in window, fill null with current quality
            pl.col("quality")
            .rolling_max(
                window_size=config.local_window_size, center=True, min_samples=1
            )
            .fill_null(pl.col("quality"))
            .alias("local_max_quality"),
            # Local mean quality in window, fill null with current quality
            pl.col("quality")
            .rolling_mean(
                window_size=config.local_window_size, center=True, min_samples=1
            )
            .fill_null(pl.col("quality"))
            .alias("local_mean_quality"),
        ]
    )
    median_quality = frames["quality"].median()

    # Initialize selection tracking
    selected_frames = []
    last_selected_idx = -config.max_gap  # Start with max gap to allow early selection
    current_penalty = 0.0

    # Process frames sequentially
    df_dict = with_metrics.to_dicts()

    for frame in df_dict:
        frame_idx = frame["frame_idx"]
        quality = frame["quality"]
        gap = frame_idx - last_selected_idx

        # Compute dynamic threshold
        gap_adjustment = min(
            gap * config.quality_boost_per_gap, config.max_gap_adjustment
        )
        penalty = current_penalty * (config.penalty_decay_rate**gap)
        threshold = max(
            config.min_quality_threshold, median_quality - gap_adjustment + penalty
        )

        # Determine if frame should be selected
        is_local_max = (
            quality >= frame["local_max_quality"] * config.local_max_ratio
            and quality >= frame["local_mean_quality"] * config.local_mean_ratio
        )

        should_select = gap >= config.min_gap and (
            (is_local_max and quality >= threshold) or gap >= config.max_gap
        )

        if should_select:
            selected_frames.append(
                {
                    "frame_idx": frame_idx,
                    "quality": quality,
                    "threshold": threshold,
                    "gap": gap,
                    "forced": gap >= config.max_gap,
                }
            )
            last_selected_idx = frame_idx
            current_penalty = config.quality_penalty_after_selection
        else:
            current_penalty *= config.penalty_decay_rate

    # Create output DataFrame
    if not selected_frames:
        return pl.DataFrame(
            schema={
                "frame_idx": pl.Int64,
                "quality": pl.Float64,
                "threshold": pl.Float64,
                "gap": pl.Int64,
                "forced": pl.Boolean,
            }
        )

    return pl.DataFrame(selected_frames)


def calculate_color_characteristics(frame: np.ndarray) -> float:
    """
    Assess color characteristics objectively.

    Strategy:
    - Analyze color distribution
    - Detect color variety and contrast

    Args:
        frame (np.ndarray): Input video frame

    Returns:
        float: Color characteristic score between 0 and 1
    """
    # Convert to LAB color space for perceptual color analysis
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Split into L*a*b* channels
    L, a, b = cv2.split(lab_frame)

    # Color diversity via standard deviation of color channels
    color_std_a = np.std(a)
    color_std_b = np.std(b)
    color_diversity_score = np.clip((color_std_a + color_std_b) / 50, 0, 1)

    # Color contrast via luminance variation
    luminance_contrast = np.std(L)
    contrast_score = np.clip(luminance_contrast / 50, 0, 1)

    # Weighted combination of color characteristics
    color_score = 0.6 * color_diversity_score + 0.4 * contrast_score

    return float(color_score)


def frame_quality_from(frame: np.ndarray) -> float:
    """
    Comprehensive frame quality assessment.

    Metrics:
    1. Sharpness
    2. Exposure
    3. Information Density
    4. Color Characteristics

    Args:
        frame (np.ndarray): Input video frame

    Returns:
        float: Quality score between 0 and 1
    """
    # Resize frame consistently
    resized_frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))

    # Convert to grayscale for efficient processing
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # 1. Sharpness: Laplacian variance (measure of high-frequency content)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = np.clip(laplacian_var / 1000, 0, 1)

    # 2. Exposure: Analyze luminance distribution
    mean_luminance = np.mean(gray)

    # Ideal luminance is around 127 (middle of 0-255 range)
    # Penalize extreme brightness or darkness
    luminance_score = 1 - np.abs(mean_luminance - 127) / 127

    # 3. Information Density: Edge detection
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (FRAME_SIZE * FRAME_SIZE)
    edge_score = np.clip(edge_density * 2, 0, 1)

    # 4. Color Characteristics
    color_score = calculate_color_characteristics(resized_frame)

    # Balanced weighting strategy
    quality_score = (
        0.45 * sharpness_score  # Crisp image details
        + 0.25 * edge_score  # Visual information
        + 0.20 * color_score  # Color variety and contrast
        + 0.10 * luminance_score  # Proper exposure
    )

    return float(quality_score)


def process_video_frames(
    video_path: Path,
    progress_keeper: Progress,
) -> pl.DataFrame:
    """Extract and quality-score frames"""
    cache_path = get_cache_path_from(
        video_path,
        step='process_video_frames',
        extension='.parquet'
    )
    if cache_path.exists():
        return pl.read_parquet(cache_path, glob=False)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    task = progress_keeper.add_task("Processing frames...", total=frame_count)

    frames_data = []
    frame_idx = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        quality = frame_quality_from(frame)
        frames_data.append({"frame_idx": frame_idx, "quality": quality})
        frame_idx += 1
        progress_keeper.advance(task, 1)

    capture.release()
    progress_keeper.update(task, completed=True)

    frame_qualities = pl.DataFrame(frames_data)
    frame_qualities.write_parquet(cache_path)
    return frame_qualities


def extract_best_frames_from(
    video_path: Path,
    *,
    window_sec: float,
    patience_factor: int,
    progress_keeper: Progress,
) -> pl.DataFrame:
    """Legacy interface wrapper"""
    cache_path = get_cache_path_from(
        video_path,
        step='extract_best_frames_from',
        window_sec=window_sec,
        patience_factor=patience_factor,
        extension='.parquet'
    )
    if cache_path.exists():
        return pl.read_parquet(cache_path, glob=False)

    PHI = 1.618
    frames_df = process_video_frames(video_path, progress_keeper)

    fps = cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FPS)
    window_size = int(window_sec * fps)
    config = FrameSelectionConfig(
        local_window_size=window_size,
        min_gap=window_size // 10 or 1,
        max_gap=int(window_size * (PHI**patience_factor)),
    )

    best_frames = select_frames(
        frames_df, config
    )[["frame_idx", "quality"]].rename({"quality": "quality_score"})
    best_frames.write_parquet(cache_path)
    return best_frames
