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

An intelligent image analysis application that uses state-of-the-art AI models to describe and analyze uploaded images.

## Features

- **Image Upload**: Easy drag-and-drop or click-to-upload interface
- **AI-Powered Analysis**: Uses Salesforce's BLIP (Bootstrapping Language-Image Pre-training) model
- **Detailed Elements List**: Identifies and lists objects, people, and elements in the image
- **General Summary**: Provides a comprehensive description of the overall scene
- **Real-time Processing**: Instant analysis as soon as an image is uploaded

## How to Use

1. Upload an image by dragging and dropping or clicking the upload area
2. The AI will automatically analyze the image and provide:
   - A detailed list of detected elements and objects
   - A general summary of what the image shows
3. Results appear instantly in the interface

## Technology

- **Models**: nlpconnect/vit-gpt2-image-captioning with fallbacks (git-base-coco, vit-gpt2-coco-en)
- **API**: Hugging Face Inference API (no local model downloads)
- **Framework**: Gradio 5.x for the web interface
- **Deployment**: Hugging Face Spaces

## Model Information

This application uses:
- **ViT-GPT2** (Vision Transformer + GPT-2) by nlpconnect for primary image captioning
- **Fallback models** including Microsoft GIT and other ViT-GPT2 variants for reliability
- **Smart model selection** - automatically tries fallback models if primary model fails
- All models accessed via Hugging Face's Inference API for fast, efficient processing

## Setup

### Local Development Setup

1. Get a Hugging Face token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Copy the `.env` file and replace `your_hf_token_here` with your actual token:
   ```
   HUGGINGFACE_TOKEN=hf_your_actual_token_here
   ```
3. Install dependencies and run:
   ```bash
   pip install -r requirements.txt
   python app.py
   ```

### HF Spaces Deployment

For HF Spaces deployment, add your token in the Space settings under "Variables and secrets" as `HUGGINGFACE_TOKEN`

## Files Structure

```
HF_Image_Recon/
├── app.py              # Main Gradio application
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (add your HF token here)
├── .gitignore         # Git ignore file
└── README.md          # This file
```

## License

MIT License - feel free to use and modify as needed.