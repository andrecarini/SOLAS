# SOLAS Interactive Module

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andrecarini/SOLAS/blob/main/SOLAS_Interactive.ipynb)

The Interactive module provides a user-friendly interface for processing lecture audio into study materials, with interactive widgets for configuration and file upload.

## Requirements

### Hardware

The pipeline requires GPU acceleration for reasonable performance.

- **Google Colab (recommended)**: Optimized for Colab's free T4 GPU (16GB VRAM). Verify the runtime shows "T4 GPU" in the top-right corner. If necessary, go to **Runtime → Change runtime type** and select **T4 GPU**.
- **Local execution**: Not officially supported but should work with minor modifications. The notebook displays a warning when running outside Colab.

### Software

#### Google Colab

No manual installation required. The notebook automatically:
- Clones the SOLAS repository from GitHub
- Installs all Python dependencies
- Installs system tools for audio processing
- Verifies GPU availability

#### Local Installation

```bash
# Clone the repository
git clone https://github.com/andrecarini/SOLAS.git
cd SOLAS

# Install system dependencies (Ubuntu/Debian)
sudo apt install ffmpeg espeak-ng libsndfile1

# Install Python dependencies
pip install -r library/requirements.txt

# Run Jupyter
jupyter notebook SOLAS_Interactive.ipynb
```

## Step-by-Step Execution

Run each cell sequentially using the Play button. Some cells display interactive widgets that require your input before proceeding.

### Cell 1: Setup Environment

Prepares the execution environment:
- Downloads or updates the SOLAS repository
- Installs required packages
- Checks GPU availability and displays VRAM information

**Important (Colab only)**: After the first run, Colab may require a runtime restart. If a restart warning appears:
1. Restart the runtime: **Runtime → Restart session** (or press `Ctrl+M` then `.`)
2. After restart, run this Setup Environment cell again to verify installation completed successfully

For local execution, restart the Jupyter kernel if prompted after installing dependencies.

### Cell 2: Configuration

An interactive configuration panel appears with the following options:

- **ASR Model**:
  - `whisper-tiny`
  - `whisper-small`
  - `whisper-large-v3`

- **LLM Model**:
  - `Qwen2-0.5B-Instruct`
  - `Qwen2-1.5B-Instruct`
  - `phi-3-mini-4k-instruct`
  - `Mistral-7B-Instruct-v0.3`

- **Quantization**: `None` or `4-bit`
- **Chunk Size**: `2000` or `8000` characters

- **Repetition Penalty**:
  - `None`
  - `1.2` (default)
  - `1.8` (aggressive)

- **Summary Mode**: `greedy`, `sampled`, or `hybrid`
- **Podcast Temperature**: Slider from 0.1 to 1.0 (default 0.3)

- **Source/Target Language**: Portuguese, English, Spanish, French, German, Italian, Russian, Chinese, Japanese, Korean

After adjusting settings, click **Confirm Configuration & Continue** to lock in your choices.

### Cell 3: Upload Input Audio

Choose your audio source:
- **Sample audio**: Pre-loaded Portuguese lecture excerpts (short, medium, or long) for testing
- **Upload my own audio**: Upload a custom file (WAV, MP3, M4A, FLAC, or OGG)

Select your choice and click **Confirm Audio Selection**. If uploading, a file picker dialog will appear.

### Cell 4: Upload Host Voices

Configure voices for the two podcast hosts:
- **Host A**: Male sample voice or upload custom
- **Host B**: Female sample voice or upload custom

The sample voices work well for most use cases. Click **Confirm Voice Selections** to proceed.

### Cell 5: Run Pipeline

Executes the complete SOLAS pipeline with a visual progress display showing:
1. Loading and preprocessing audio
2. Transcribing audio (ASR)
3. Translating transcript
4. Generating key points summary
5. Creating podcast script
6. Synthesizing podcast audio (TTS)

Each stage shows a progress bar, elapsed time, and VRAM usage.

### Cell 6: View Results

Displays all pipeline outputs in a formatted interface:
- **Performance summary**: Total runtime, audio duration, real-time factor
- **Stage metrics**: Time and VRAM usage per stage
- **Text outputs**: Expandable sections for transcript, translation, summary, and script
- **Audio player**: Embedded player for the generated podcast
- **Download buttons**: Save individual outputs

## Tips

- **Session disconnects**: If Colab disconnects mid-run, restart from Cell 1 (Setup Environment).
- **VRAM errors**: Try a smaller LLM model or enable 4-bit quantization.
- **First run latency**: Initial runs are slower due to model downloads from Hugging Face.
- **Sample audio**: Use the short sample for quick testing before processing your own files.

## Output Files

The pipeline generates the following files in the output directory:

- `original_transcript.txt`: Raw transcription in the source language
- `translated_transcript.txt`: Translated text in the target language
- `key_points.md`: Bullet-point summary of main topics (Markdown format)
- `podcast_script.txt`: Two-host dialogue script
- `podcast.wav`: Synthesized podcast audio file
