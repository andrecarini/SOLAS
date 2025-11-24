<p align="center">
  <img src="logo.png" alt="SOLAS Logo" width="400">
</p>

# SOLAS - Self-hosted Open-source Lecture Assistant System

SOLAS is an end-to-end pipeline that transforms lecture audio into accessible study materials, including translated transcripts, key-point summaries, and conversational podcasts.

## Features

- **Automatic Speech Recognition (ASR)**: Transcribe lectures using OpenAI Whisper models
- **Translation**: Translate transcripts between 10+ languages using instruction-tuned LLMs
- **Summarization**: Extract key points as bullet-point summaries
- **Podcast Generation**: Create two-host conversational scripts from lecture content
- **Text-to-Speech**: Synthesize podcast audio with customizable voices

## Notebooks

### Interactive Module

The Interactive notebook provides a user-friendly interface for processing lecture audio into study materials.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andrecarini/SOLAS/blob/main/SOLAS_Interactive.ipynb)

See [SOLAS_Interactive_README.md](docs/SOLAS_Interactive_README.md) for detailed usage instructions.

### Evaluation Module

The Evaluation notebook runs controlled experiments to benchmark pipeline performance across different configurations.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andrecarini/SOLAS/blob/main/SOLAS_Evaluation.ipynb)

See [SOLAS_Evaluation_README.md](docs/SOLAS_Evaluation_README.md) for detailed usage instructions.

### Analysis Module

The Analysis notebook visualizes and analyzes evaluation results. **Does not require a GPU.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andrecarini/SOLAS/blob/main/SOLAS_Analysis.ipynb)

See [SOLAS_Analysis_README.md](docs/SOLAS_Analysis_README.md) for detailed usage instructions.

## Quick Start

### Google Colab (Recommended)

1. Click one of the "Open in Colab" buttons above
2. Select **Runtime → Change runtime type → T4 GPU**
3. Run the cells sequentially

### Local Installation

```bash
# Clone the repository
git clone https://github.com/andrecarini/SOLAS.git
cd SOLAS

# Install system dependencies (Ubuntu/Debian)
sudo apt install ffmpeg espeak-ng libsndfile1

# Install Python dependencies
pip install -r library/requirements.txt

# Start Jupyter and select a notebook
jupyter notebook
```

## Supported Models

### ASR Models
- `openai/whisper-tiny` - Fast, lower accuracy
- `openai/whisper-small` - Balanced speed/accuracy
- `openai/whisper-large-v3` - Best accuracy (recommended)

### LLM Models
- `Qwen/Qwen2-0.5B-Instruct` - Smallest, fastest
- `Qwen/Qwen2-1.5B-Instruct` - Small, good quality
- `microsoft/phi-3-mini-4k-instruct` - Medium, excellent quality
- `mistralai/Mistral-7B-Instruct-v0.3` - Largest, best quality

### TTS
- Coqui XTTS v2 with voice cloning support

## Supported Languages

Portuguese, English, Spanish, French, German, Italian, Russian, Chinese, Japanese, Korean

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support
- **VRAM**:
  - 8GB minimum (with 4-bit quantization)
  - 16GB recommended (Google Colab T4)
  - 40GB for full-precision 7B models

## Credits

Sample audio files included in this repository are attributed to their respective sources. See [CREDITS.md](CREDITS.md) for details.

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See [LICENSE](LICENSE.md) for details.
