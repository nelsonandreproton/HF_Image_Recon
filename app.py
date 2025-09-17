import gradio as gr
from PIL import Image
import os
import re
import logging
import traceback
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

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

# Load environment variables from .env file
load_dotenv()

def query_hf_api(image, model_id="nlpconnect/vit-gpt2-image-captioning", fallback_models=None):
    """Query Hugging Face Inference API for image captioning using InferenceClient"""
    logger.info(f"Starting image analysis with primary model: {model_id}")
    
    token = os.getenv('HUGGINGFACE_TOKEN', '').strip()
    if not token or token == 'your_token_here':
        logger.error("Valid HUGGINGFACE_TOKEN not found in environment variables")
        raise ValueError("Please set a valid HUGGINGFACE_TOKEN in your .env file")
    
    # Validate token format
    if not token.startswith('hf_'):
        logger.warning("Token doesn't start with 'hf_' - this might be invalid")
    
    # Initialize InferenceClient
    try:
        client = InferenceClient(token=token)
        logger.info("InferenceClient initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize InferenceClient: {str(e)}")
        raise Exception(f"Failed to initialize InferenceClient: {str(e)}")
    
    # Try the primary model first
    models_to_try = [model_id]
    if fallback_models:
        models_to_try.extend(fallback_models)
    
    logger.info(f"Will try {len(models_to_try)} models: {models_to_try}")
    
    last_error = None
    last_exception = None
    
    for i, model in enumerate(models_to_try):
        try:
            logger.info(f"Attempting model {i+1}/{len(models_to_try)}: {model}")
            
            # Log image info for debugging
            if hasattr(image, 'size'):
                logger.info(f"Image size: {image.size}, mode: {image.mode}")
            
            # Use InferenceClient for image-to-text
            result = client.image_to_text(image, model=model)
            logger.info(f"Successfully got result from model {model}: {type(result)}")
            logger.debug(f"Raw result: {result}")
            
            # InferenceClient returns string directly, wrap in expected format
            if isinstance(result, str):
                logger.info(f"Model {model} returned string result of length {len(result)}")
                return [{"generated_text": result.strip()}]  # Ensure no leading/trailing whitespace
            elif isinstance(result, list) and len(result) > 0:
                logger.info(f"Model {model} returned list with {len(result)} items")
                # If it's already a list, ensure proper format
                return result
            else:
                logger.warning(f"Model {model} returned unexpected format: {type(result)}")
                return [{"generated_text": "Unable to generate caption"}]
                
        except Exception as e:
            last_exception = e
            error_details = {
                'model': model,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }
            logger.error(f"Model {model} failed: {error_details}")
            
            # Log specific error types for better debugging
            if hasattr(e, 'response'):
                logger.error(f"HTTP Response Status: {getattr(e.response, 'status_code', 'Unknown')}")
                logger.error(f"HTTP Response Text: {getattr(e.response, 'text', 'Unknown')}")
            
            last_error = f"Model {model} failed: {str(e)} ({type(e).__name__})"
            continue
    
    # If all models failed, log comprehensive error info and raise
    logger.error(f"All {len(models_to_try)} models failed. Last exception: {last_exception}")
    if last_exception:
        logger.error(f"Final traceback: {traceback.format_exc()}")
    
    raise Exception(f"All models failed. Last error: {last_error}")

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
        
        # Define fallback models in case primary ones fail
        fallback_models = [
            "microsoft/git-base-coco",
            "ydshieh/vit-gpt2-coco-en", 
            "Salesforce/blip-image-captioning-base"
        ]
        
        # Get general caption
        logger.info("Starting API query for image captioning")
        result = query_hf_api(image, fallback_models=fallback_models)
        logger.info(f"API query completed, result type: {type(result)}")
        
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
        
    except ValueError as e:
        logger.error(f"ValueError in analyze_image: {str(e)}")
        if "HUGGINGFACE_TOKEN not found" in str(e):
            return "⚠️ **Setup Required**: Please add your Hugging Face token to the .env file. See README for instructions.", ""
        return f"❌ **Configuration Error**: {str(e)}", ""
    except Exception as e:
        logger.error(f"Exception in analyze_image: {type(e).__name__}: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        error_msg = str(e)
        # Handle common InferenceClient errors
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            logger.error("Authentication error detected")
            return "⚠️ **Authentication Error**: Invalid Hugging Face token. Please verify your token.", ""
        elif "503" in error_msg or "model is currently loading" in error_msg.lower():
            logger.error("Service/loading error detected")
            return "⚠️ **Service Error**: Model is loading. Please try again in a few moments.", ""
        elif "rate limit" in error_msg.lower():
            logger.error("Rate limit error detected")
            return "⚠️ **Rate Limit**: Too many requests. Please wait before trying again.", ""
        elif "all models failed" in error_msg:
            logger.error("All models failed error detected")
            return f"❌ **Model Error**: {error_msg}. Check app.log for detailed error information.", ""
        
        logger.error(f"Unexpected error type: {type(e).__name__}")
        return f"❌ **Unexpected Error**: {error_msg}. Check app.log for details.", ""

with gr.Blocks(title="AI Image Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🖼️ AI Image Analyzer")
    gr.Markdown("Upload an image to get a detailed analysis of its contents and a general summary.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            analyze_btn = gr.Button("Analyze Image", variant="primary", size="lg")
        
        with gr.Column():
            elements_output = gr.Markdown(label="Image Elements", value="")
            summary_output = gr.Textbox(label="General Summary", lines=3, max_lines=5)
            loading_indicator = gr.Markdown(value="⏳ Analyzing image...", visible=False)
    
    def on_image_upload_change(image):
        if image is not None:
            return [gr.update(visible=True), *analyze_image(image)]
        return [gr.update(visible=False), "", ""]
    
    analyze_btn.click(
        fn=lambda image: ["", *analyze_image(image)] if image else ["Please upload an image first.", ""],
        inputs=[image_input],
        outputs=[loading_indicator, elements_output, summary_output]
    )
    
    # Only auto-analyze on image change, not on manual button click to avoid duplicates
    image_input.upload(
        fn=on_image_upload_change,
        inputs=[image_input],
        outputs=[loading_indicator, elements_output, summary_output]
    )
    
    gr.Markdown("### About")
    gr.Markdown("This app uses Hugging Face's Inference API with ViT-GPT2 and other vision models for image captioning and analysis.")
    gr.Markdown("**Setup:** Add your HF token to the `.env` file as `HUGGINGFACE_TOKEN=your_token_here`")

if __name__ == "__main__":
    demo.launch()