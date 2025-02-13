# cinextract

Extract still frames from videos using scene detection and quality assessment.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

## Quick Start

```bash
# Install
pip install cinextract

# Basic usage
cinextract video.mp4

# Custom output directory
cinextract video.mp4 --output-dir frames/
```

## How It Works

The tool:
1. Analyzes frames for motion blur and visual quality
2. Creates CLIP embeddings to understand visual content
3. Groups similar frames into scenes
4. Selects the best representative frames from each scene
5. Exports frames as JPEGs with quality metadata

## Requirements

- Python 3.10 or higher
- PyTorch with appropriate backend (CPU/CUDA/Metal)
- 8GB+ RAM recommended for HD video
- GPU recommended but not required

## Installation

Standard install:
```bash
pip install cinextract
```

Development install:
```bash
git clone https://github.com/ryzhakar/cinextract.git
cd cinextract
pip install -e ".[dev]"
```

## Command Line Options

```
cinextract VIDEO_PATH [OPTIONS]

Options:
  --output-dir PATH           Output directory [default: output/]
  --device TEXT              Computing device (cpu/cuda/mps) [default: auto]
  --top-percentile FLOAT     Percentage of top frames to export [default: 10.0]
  --min-export-count INT     Minimum frames to export [default: 10]
  --window-sec FLOAT         Analysis window in seconds [default: 2.0]
  --embedding-batch-size INT CLIP batch size [default: 256]
  --patience-factor INT      Frame selection multiplier [default: 4]
```

## Output Structure

The tool creates:
- JPEG frames in the output directory
- Filenames encode frame index, scene ID, and quality score:
  `frame_000123_c5_a7.842.jpg`
  - `000123`: Frame index in video
  - `c5`: Scene/cluster ID
  - `a7.842`: Aesthetic score

## Common Issues

1. Out of Memory
   - Reduce `embedding-batch-size`
   - Process shorter video segments
   
2. Slow Processing
   - Enable GPU acceleration
   - Decrease `window-sec` parameter
   - Increase `patience-factor` for fewer frames

3. Poor Frame Selection
   - Decrease `patience-factor` for more frames
   - Adjust `top-percentile` based on content
   - Ensure input video is of sufficient quality

## License

GNU Affero General Public License v3.0 or later - see [LICENSE](LICENSE)

## Contributing

Issues and pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
