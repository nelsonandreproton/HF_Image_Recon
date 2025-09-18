import gradio as gr
from PIL import Image
import os
import re
import logging
import traceback
import base64
import io
import requests
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def query_ollama_api(image, model="moondream:latest", ollama_url="http://localhost:11434"):
    """Query local Ollama API for image analysis using Moondream model"""
    logger.info(f"Starting image analysis with Ollama model: {model}")

    try:
        # Convert image to base64
        image_b64 = image_to_base64(image)
        logger.info(f"Image converted to base64, size: {len(image_b64)} characters")

        # Prepare the API request
        api_url = f"{ollama_url}/api/generate"
        payload = {
            "model": model,
            "prompt": "Describe this image in detail. What objects, people, activities, colors, and other elements do you see?",
            "images": [image_b64],
            "stream": False
        }

        logger.info(f"Sending request to Ollama API: {api_url}")

        # Make the API call
        response = requests.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            description = result.get("response", "No description generated")
            logger.info(f"Successfully got result from Ollama: {len(description)} characters")
            return [{"generated_text": description}]
        else:
            error_msg = f"Ollama API error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)

    except requests.exceptions.ConnectionError:
        error_msg = "Cannot connect to Ollama. Make sure Ollama is running on localhost:11434"
        logger.error(error_msg)
        raise Exception(error_msg)
    except requests.exceptions.Timeout:
        error_msg = "Ollama request timed out. The model might be loading or processing is slow."
        logger.error(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        logger.error(f"Ollama API error: {str(e)}")
        raise

def analyze_image(image):
    """Main function to analyze uploaded image"""
    logger.info(f"=== Starting image analysis at {datetime.now()} ===")

    if image is None:
        logger.warning("No image provided to analyze_image function")
        return "Please upload an image first.", ""

    try:
        # Log image input type and details
        logger.info(f"Received image type: {type(image)}")

        # Validate image size to prevent issues with large images
        if hasattr(image, 'size'):
            width, height = image.size
            megapixels = (width * height) / 1_000_000
            logger.info(f"Image dimensions: {width}x{height} ({megapixels:.2f} MP)")

            # Warn if image is very large
            if megapixels > 20:  # More than 20 megapixels
                logger.warning(f"Large image detected: {megapixels:.2f} MP")
                # Optionally resize very large images
                if megapixels > 50:
                    logger.info("Resizing extremely large image")
                    # Resize to a more manageable size while maintaining aspect ratio
                    max_dimension = 5000
                    if width > height:
                        new_width = min(width, max_dimension)
                        new_height = int(height * (new_width / width))
                    else:
                        new_height = min(height, max_dimension)
                        new_width = int(width * (new_height / height))
                    image = image.resize((new_width, new_height), Image.LANCZOS)
                    logger.info(f"Resized image to {new_width}x{new_height}")

        if isinstance(image, str):
            logger.info(f"Loading image from path: {image}")
            image = Image.open(image).convert('RGB')
        elif hasattr(image, 'convert'):
            logger.info(f"Converting existing PIL image to RGB")
            image = image.convert('RGB')
        else:
            logger.error(f"Unsupported image type: {type(image)}")
            return "❌ **Error**: Unsupported image format", ""

        logger.info(f"Final image details: size={image.size}, mode={image.mode}")

        # Get general caption using Ollama
        logger.info("Starting Ollama API query for image captioning")
        result = query_ollama_api(image, model="moondream:latest")
        logger.info(f"Ollama API query completed, result type: {type(result)}")

        if isinstance(result, list) and len(result) > 0:
            general_summary = result[0].get('generated_text', 'Unable to generate caption')
            logger.info(f"Extracted caption from list result: {len(general_summary)} characters")
        elif isinstance(result, dict):
            general_summary = result.get('generated_text', result.get('error', 'Unable to generate caption'))
            logger.info(f"Extracted caption from dict result: {len(general_summary)} characters")
        else:
            logger.warning(f"Unexpected result format: {type(result)}, content: {result}")
            general_summary = "Unable to generate caption"

        # For detailed analysis, just enhance the existing description
        # since many models aren't available via Inference API
        detailed_description = general_summary

        # Extract potential objects from the descriptions
        all_text = f"{general_summary} {detailed_description}".lower()

        visible_objects = []
        common_objects = ["person", "people", "man", "woman", "child", "car", "tree", "building", "house", "sky", "cloud", "water", "grass", "flower", "animal", "dog", "cat", "bird", "table", "chair", "book", "phone", "computer", "food", "plate", "cup", "bottle", "bag", "road", "street", "window", "door", "light", "sign", "bike", "bicycle", "bus", "train", "boat", "airplane", "traffic", "fire", "stop"]

        for obj in common_objects:
            # More precise pattern matching with word boundaries and optional plurals
            pattern = r'\b' + re.escape(obj) + r's?\b'
            if re.search(pattern, all_text):
                visible_objects.append(obj.capitalize())

        # Format the detailed elements
        elements_list = f"**AI Description:**\n{detailed_description}\n\n"

        if visible_objects:
            elements_list += f"**Detected Objects:**\n" + "\n".join([f"• {obj}" for obj in set(visible_objects)])
        else:
            elements_list += "**Note:** Upload an image to see detailed element detection"

        logger.info(f"Analysis completed successfully. Summary length: {len(general_summary)}, Elements length: {len(elements_list)}")
        logger.info("=== Image analysis completed successfully ===")
        return elements_list, general_summary

    except Exception as e:
        logger.error(f"Exception in analyze_image: {type(e).__name__}: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")

        error_msg = str(e)
        # Handle Ollama-specific errors
        if "Cannot connect to Ollama" in error_msg:
            logger.error("Ollama connection error detected")
            return "🔌 **Connection Error**: Cannot connect to Ollama. Make sure Ollama is running on localhost:11434 and the moondream model is available.", ""
        elif "timed out" in error_msg.lower():
            logger.error("Ollama timeout error detected")
            return "⏱️ **Timeout Error**: Ollama request timed out. The model might be loading or processing is slow. Please try again.", ""
        elif "Ollama API error" in error_msg:
            logger.error("Ollama API error detected")
            return f"❌ **Ollama Error**: {error_msg}. Check if the moondream model is properly installed.", ""

        logger.error(f"Unexpected error type: {type(e).__name__}")
        return f"❌ **Unexpected Error**: {error_msg}. Check app.log for details.", ""

with gr.Blocks(title="AI Image Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🖼️ AI Image Analyzer")
    gr.Markdown("Upload an image to get a detailed analysis of its contents and a general summary.")
    gr.Markdown("**Note**: This app uses local Ollama with the Moondream model for image analysis.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            analyze_btn = gr.Button("Analyze Image", variant="primary", size="lg")

        with gr.Column():
            elements_output = gr.Markdown(label="Image Elements", value="")
            summary_output = gr.Textbox(label="General Summary", lines=3, max_lines=5)

    analyze_btn.click(
        fn=analyze_image,
        inputs=[image_input],
        outputs=[elements_output, summary_output]
    )

    # Only auto-analyze on image change, not on manual button click to avoid duplicates
    image_input.upload(
        fn=analyze_image,
        inputs=[image_input],
        outputs=[elements_output, summary_output]
    )

    gr.Markdown("### About")
    gr.Markdown("This app uses local Ollama with the Moondream vision model for image captioning and analysis.")
    gr.Markdown("**Setup:** Make sure Ollama is running locally and the `moondream:latest` model is installed (`ollama pull moondream:latest`)")

if __name__ == "__main__":
    demo.launch()