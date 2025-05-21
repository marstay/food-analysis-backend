from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import io
import logging
import time
import gc
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure TensorFlow to use less memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logger.warning(f"GPU memory growth setting failed: {e}")

# Limit TensorFlow memory usage
tf.config.set_logical_device_configuration(
    tf.config.list_physical_devices('CPU')[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=512)]  # Limit to 512MB
)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model only once
model = None
class_names = None

def load_model():
    global model, class_names
    if model is None:
        logger.info("Loading model...")
        start_time = time.time()
        try:
            model = hub.load("https://tfhub.dev/google/aiy/vision/classifier/food_V1/1")
            class_names = model.class_names
            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to load model")
    return model, class_names

def preprocess_image(image_data: bytes, max_size: int = 512) -> np.ndarray:  # Reduced max size
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Calculate new size maintaining aspect ratio
        ratio = min(max_size / image.width, max_size / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        
        # Resize image
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0
        
        # Clear memory
        del image
        gc.collect()
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def analyze_image_quality(image: np.ndarray) -> dict:
    try:
        # Convert to grayscale for some metrics
        gray = np.mean(image, axis=-1)
        
        # Calculate color statistics
        color_mean = np.mean(image, axis=(0, 1))
        color_std = np.std(image, axis=(0, 1))
        color_variance = np.var(image, axis=(0, 1))
        
        # Calculate brightness
        brightness = np.mean(gray)
        
        # Calculate saturation
        max_color = np.max(image, axis=-1)
        min_color = np.min(image, axis=-1)
        saturation = np.mean((max_color - min_color) / (max_color + 1e-6))
        
        # Detect dark spots (areas with very low brightness)
        dark_spots = np.mean(gray < 0.2) * 100
        
        # Detect unusual colors (high variance in color channels)
        unusual_colors = np.mean(color_variance > 0.1) * 100
        
        # Clear memory
        del gray, max_color, min_color
        gc.collect()
        
        return {
            "saturation": float(saturation),
            "brightness": float(brightness),
            "color_variance": float(np.mean(color_variance)),
            "dark_spots": float(dark_spots),
            "unusual_colors": float(unusual_colors)
        }
    except Exception as e:
        logger.error(f"Error analyzing image quality: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing image quality: {str(e)}")

def analyze_food_image(image: np.ndarray) -> dict:
    try:
        # Load model if not loaded
        model, class_names = load_model()
        
        # Get predictions
        predictions = model(image)
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        
        # Get top predictions
        top_predictions = [
            {
                "name": class_names[idx].decode('utf-8'),
                "confidence": float(predictions[0][idx] * 100)
            }
            for idx in top_3_idx
        ]
        
        # Clear memory
        del predictions
        gc.collect()
        
        # Get image quality metrics
        quality_metrics = analyze_image_quality(image)
        
        # Determine if food is safe based on quality metrics
        is_safe = (
            quality_metrics["dark_spots"] < 20 and
            quality_metrics["unusual_colors"] < 30 and
            quality_metrics["color_variance"] < 0.2
        )
        
        # Calculate confidence based on quality metrics
        confidence = 100 - (
            quality_metrics["dark_spots"] * 0.4 +
            quality_metrics["unusual_colors"] * 0.4 +
            quality_metrics["color_variance"] * 100 * 0.2
        )
        confidence = max(0, min(100, confidence))
        
        # Generate message and recommendations
        message = "This food appears to be in good condition." if is_safe else "This food shows signs of spoilage."
        recommendations = []
        
        if quality_metrics["dark_spots"] > 10:
            recommendations.append("Check for mold or dark spots")
        if quality_metrics["unusual_colors"] > 20:
            recommendations.append("Unusual discoloration detected")
        if quality_metrics["color_variance"] > 0.15:
            recommendations.append("Color appears inconsistent")
            
        # Clear memory
        del image
        gc.collect()
            
        return {
            "isSafe": is_safe,
            "confidence": float(confidence),
            "message": message,
            "identified_food": top_predictions[0],
            "analysis": {
                "color_analysis": {
                    "saturation": quality_metrics["saturation"],
                    "brightness": quality_metrics["brightness"],
                    "color_variance": quality_metrics["color_variance"]
                },
                "defects": {
                    "dark_spots": quality_metrics["dark_spots"],
                    "unusual_colors": quality_metrics["unusual_colors"]
                },
                "food_identification": {
                    "top_prediction": top_predictions[0],
                    "all_predictions": top_predictions
                }
            },
            "issues": [
                {
                    "severity": "high" if quality_metrics["dark_spots"] > 15 else "medium",
                    "description": "Dark spots detected"
                } if quality_metrics["dark_spots"] > 10 else None,
                {
                    "severity": "high" if quality_metrics["unusual_colors"] > 25 else "medium",
                    "description": "Unusual discoloration"
                } if quality_metrics["unusual_colors"] > 20 else None
            ],
            "recommendations": recommendations
        }
    except Exception as e:
        logger.error(f"Error analyzing food image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing food image: {str(e)}")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        logger.info(f"Received image: {file.filename}")
        start_time = time.time()
        
        # Read image data
        image_data = await file.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="Empty image file")
            
        # Preprocess image
        image_array = preprocess_image(image_data)
        
        # Clear memory
        del image_data
        gc.collect()
        
        # Analyze image
        result = analyze_food_image(image_array)
        
        logger.info(f"Analysis completed in {time.time() - start_time:.2f} seconds")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable (Render sets this)
    port = int(os.getenv("PORT", 8080))
    
    print(f"\nServer starting...")
    print(f"Server will be available at port: {port}")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port) 