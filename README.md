---
title: AI Image Analyzer
emoji: 🖼️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

# AI Image Analyzer

An intelligent image analysis application that uses local Ollama with the Moondream vision model to describe and analyze uploaded images.

## Features

- **Image Upload**: Easy drag-and-drop or click-to-upload interface
- **AI-Powered Analysis**: Uses Moondream vision model via local Ollama
- **Detailed Elements List**: Identifies and lists objects, people, and elements in the image
- **General Summary**: Provides a comprehensive description of the overall scene
- **Real-time Processing**: Instant analysis as soon as an image is uploaded
- **Local Processing**: No external API calls - everything runs locally

## How to Use

1. Upload an image by dragging and dropping or clicking the upload area
2. The AI will automatically analyze the image and provide:
   - A detailed list of detected elements and objects
   - A general summary of what the image shows
3. Results appear instantly in the interface

## Technology

- **Model**: Moondream vision model (local)
- **API**: Local Ollama server
- **Framework**: Gradio 5.x for the web interface
- **Deployment**: Local

## Model Information

This application uses:
- **Moondream** - A powerful vision language model for image understanding
- **Local Ollama** - No external API dependencies
- **Fast processing** - Direct local model inference

## Setup

### Prerequisites

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull the Moondream model:
   ```bash
   ollama pull moondream:latest
   ```
3. Start Ollama (if not running as service):
   ```bash
   ollama serve
   ```

### Local Development Setup

1. Install dependencies and run:
   ```bash
   pip install -r requirements.txt
   python app.py
   ```

2. Test the setup:
   ```bash
   python test_ollama.py
   ```

## Files Structure

```
HF_Image_Recon/
├── app.py              # Main Gradio application
├── requirements.txt    # Python dependencies
├── test_ollama.py     # Test script for Ollama integration
├── .gitignore         # Git ignore file
└── README.md          # This file
```

## License

MIT License - feel free to use and modify as needed.