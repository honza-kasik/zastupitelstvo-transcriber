# Environment Setup Guide

This document explains how to set up the Python environment for the meeting transcriber pipeline.

## TL;DR - Quick Setup

```bash
# Automated setup (recommended)
./setup_env.sh

# Activate environment
source venv/bin/activate

# Set HuggingFace token (for transcriber)
export HF_TOKEN="your_token_here"
```

## Python Version Compatibility

### Recommended: Python 3.12

The pipeline has been **tested and verified on Python 3.12**. This is the recommended version.

### Supported: Python 3.10-3.12

The pipeline should work on Python 3.10, 3.11, and 3.12.

### Not Supported: Python 3.13+

Python 3.13+ may have compatibility issues with some dependencies (particularly PyTorch and its ecosystem).

### Check Your Version

```bash
python3 --version
# or
python3.12 --version
```

## Installation Options

### Option 1: Automated Setup (Recommended)

The `setup_env.sh` script handles everything automatically:

```bash
./setup_env.sh
```

This will:
1. Detect your Python version
2. Create a virtual environment
3. Ask which components you need
4. Install dependencies
5. Show next steps

### Option 2: Manual Setup - Full Pipeline

For all components (transcriber + analyzer + article generator):

```bash
# Create virtual environment with Python 3.12
python3.12 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

### Option 3: Manual Setup - Specific Components

If you only need certain components, install only what you need:

#### Article Generator Only (~10MB)
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements-article-generator.txt
```

#### Analyzer Only (~500MB)
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements-analyzer.txt
```

#### Transcriber Only (~2GB)
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements-transcriber.txt
```

## Requirements Files Explained

| File | Size | Components | Use Case |
|------|------|------------|----------|
| `requirements.txt` | ~2GB | All | Full pipeline |
| `requirements-transcriber.txt` | ~2GB | Whisper + PyTorch | Audio transcription |
| `requirements-analyzer.txt` | ~500MB | scikit-learn + HDBSCAN | Topic analysis |
| `requirements-article-generator.txt` | <10MB | PyYAML only | Article generation |
| `requirements-core.txt` | <10MB | Base utilities | (auto-included) |
| `requirements-dev.txt` | ~2GB+ | All + dev tools | Development |

## Common Issues and Solutions

### Issue 1: "morphodita not found"

**Problem**: The old `requirements.txt` had `ufal.udpipe` instead of `ufal.morphodita`.

**Solution**: The new requirements files have been fixed. Reinstall:
```bash
pip install -r requirements-analyzer.txt
```

### Issue 2: PyTorch Installation Fails

**Problem**: PyTorch is large (~2GB) and can fail on slow connections.

**Solution**: Install with retry or manually:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

For GPU support (CUDA 11.8):
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue 3: Python 3.13+ Compatibility

**Problem**: PyTorch and some dependencies don't support Python 3.13 yet.

**Solution**: Use Python 3.12 or 3.11:
```bash
# Install Python 3.12 (Fedora/RHEL)
sudo dnf install python3.12

# Create venv with specific version
python3.12 -m venv venv
```

### Issue 4: Out of Disk Space

**Problem**: Full installation requires ~2GB disk space.

**Solution**: Install only needed components:
```bash
# Just article generator (if transcript already analyzed)
pip install -r requirements-article-generator.txt

# Just analyzer (if transcript already created)
pip install -r requirements-analyzer.txt
```

### Issue 5: ufal.morphodita Build Errors

**Problem**: `ufal.morphodita` might fail to build on some systems.

**Solution**: Install build dependencies:
```bash
# Fedora/RHEL
sudo dnf install gcc gcc-c++ python3-devel

# Ubuntu/Debian
sudo apt install build-essential python3-dev
```

Then retry:
```bash
pip install ufal.morphodita
```

## GPU Support

### For Transcription (Whisper)

Install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then use the `--device cuda` flag:
```bash
python process_meeting.py --audio input.opus --date 2025-01-15 --number 23 --device cuda
```

### Check GPU Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## Virtual Environment Management

### Activate Environment

```bash
source venv/bin/activate
```

### Deactivate Environment

```bash
deactivate
```

### Remove Environment

```bash
rm -rf venv
```

### Recreate Environment

```bash
rm -rf venv
./setup_env.sh
```

## Docker Alternative

If you have dependency issues, use Docker for the transcriber:

```bash
cd transcriber/
docker compose build
docker compose up
```

See `transcriber/README.adoc` for details.

## Environment Variables

### Required for Transcriber

```bash
# HuggingFace token for speaker diarization
export HF_TOKEN="your_token_here"
```

Get your token at: https://huggingface.co/settings/tokens

Also accept the pyannote license: https://huggingface.co/pyannote/speaker-diarization

### Optional

```bash
# Override Whisper model size
export WHISPER_MODEL="large-v3"

# Thread count for analysis
export OMP_NUM_THREADS=12
```

## Troubleshooting Checklist

If you have issues, check:

- [ ] Python version is 3.10-3.12 (check: `python3 --version`)
- [ ] Virtual environment is activated (prompt should show `(venv)`)
- [ ] pip is up to date (run: `pip install --upgrade pip`)
- [ ] Sufficient disk space (~3GB free for full install)
- [ ] Build tools installed (gcc, g++, python3-dev)
- [ ] HF_TOKEN is set (for transcriber)
- [ ] Using correct requirements file for your needs

## Quick Reference

### Full Pipeline Setup
```bash
./setup_env.sh
source venv/bin/activate
export HF_TOKEN="your_token"
python process_meeting.py --audio input.opus --date 2025-01-15 --number 23
```

### Minimal Setup (Article Generator Only)
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements-article-generator.txt
python article-generator/generate_meeting_article.py --topics llm_input.json --date 2025-01-15 --number 23
```

### Development Setup
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
pytest  # Run tests
black . # Format code
```

## Need Help?

1. Check this document for your specific issue
2. Check component READMEs:
   - `transcriber/README.adoc`
   - `article-generator/README.adoc`
   - Root `README.adoc`
3. Verify environment with:
   ```bash
   python -c "import sys; print(sys.version)"
   pip list | grep -E "(torch|whisper|pyannote|morphodita)"
   ```
