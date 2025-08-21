---
title: CLIP Demo
---

# CLIP Demo (Vision-Language Model)

This page shows how I ran CLIP locally and in Google Colab to ask “what’s in an image?”

## Local (VS Code) steps

1. Create a virtual env and install deps:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install torch torchvision torchaudio pillow ftfy regex tqdm
   pip install git+https://github.com/openai/CLIP.git