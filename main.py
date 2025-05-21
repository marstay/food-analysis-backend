from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model
model = None
class_names = None

def load_model():
    global model, class_names
    if model is None:
        # Load the model from TensorFlow Hub
        model = hub.load('https://tfhub.dev/google/aiy/vision/classifier/food_V1/1')
        # Load class names
        class_names = model.class_names
    return model, class_names

def preprocess_image(image):
    # Resize image to model's expected size
    image = image.resize((224, 224))
    # Convert to numpy array and normalize
    image = np.array(image) / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

def analyze_image_quality(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Calculate color statistics
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    
    # Calculate average saturation and value
    avg_saturation = np.mean(saturation)
    avg_value = np.mean(value)
    
    # Calculate color variance (indicates discoloration)
    color_variance = np.var(hsv[:, :, 0])
    
    # Detect dark spots (potential mold or spoilage)
    dark_spots = cv2.threshold(value, 50, 255, cv2.THRESH_BINARY_INV)[1]
    dark_spot_percentage = (np.sum(dark_spots) / 255) / (image.shape[0] * image.shape[1]) * 100
    
    # Detect unusual color patterns
    unusual_colors = cv2.threshold(saturation, 150, 255, cv2.THRESH_BINARY)[1]
    unusual_color_percentage = (np.sum(unusual_colors) / 255) / (image.shape[0] * image.shape[1]) * 100
    
    return {
        'avg_saturation': avg_saturation,
        'avg_value': avg_value,
        'color_variance': color_variance,
        'dark_spot_percentage': dark_spot_percentage,
        'unusual_color_percentage': unusual_color_percentage
    }

def analyze_food_image(image: np.ndarray) -> dict:
    # Load model and get predictions
    model, class_names = load_model()
    preprocessed_image = preprocess_image(Image.fromarray(image))
    predictions = model(preprocessed_image)[0]
    
    # Get top 3 predictions
    top_3_idx = predictions.numpy().argsort()[-3:][::-1]
    food_predictions = [
        {
            "name": class_names[idx].decode('utf-8'),
            "confidence": float(predictions[idx]) * 100
        }
        for idx in top_3_idx
    ]
    
    # Analyze image quality
    quality_metrics = analyze_image_quality(image)
    
    # Initialize safety assessment
    is_safe = True
    confidence = 0.8
    issues = []
    recommendations = []
    
    # Detailed analysis of visual characteristics
    analysis_details = {
        "color_analysis": {
            "saturation": quality_metrics['avg_saturation'],
            "brightness": quality_metrics['avg_value'],
            "color_variance": quality_metrics['color_variance']
        },
        "defects": {
            "dark_spots": quality_metrics['dark_spot_percentage'],
            "unusual_colors": quality_metrics['unusual_color_percentage']
        },
        "food_identification": {
            "top_prediction": food_predictions[0],
            "all_predictions": food_predictions
        }
    }
    
    # Check for signs of spoilage with more detailed thresholds
    if quality_metrics['dark_spot_percentage'] > 5:
        is_safe = False
        confidence *= 0.7
        issues.append({
            "type": "dark_spots",
            "severity": "high" if quality_metrics['dark_spot_percentage'] > 10 else "medium",
            "description": "Dark spots detected which might indicate mold or bacterial growth",
            "percentage": round(quality_metrics['dark_spot_percentage'], 2)
        })
        recommendations.append("Inspect the food carefully for visible mold or discoloration")
    
    if quality_metrics['unusual_color_percentage'] > 10:
        is_safe = False
        confidence *= 0.8
        issues.append({
            "type": "discoloration",
            "severity": "high" if quality_metrics['unusual_color_percentage'] > 20 else "medium",
            "description": "Unusual discoloration detected",
            "percentage": round(quality_metrics['unusual_color_percentage'], 2)
        })
        recommendations.append("Check if the color is normal for this type of food")
    
    if quality_metrics['color_variance'] > 5000:
        is_safe = False
        confidence *= 0.85
        issues.append({
            "type": "inconsistent_coloring",
            "severity": "medium",
            "description": "Inconsistent coloring detected",
            "variance": round(quality_metrics['color_variance'], 2)
        })
        recommendations.append("Look for any unusual color patterns or spots")
    
    if quality_metrics['avg_saturation'] < 50:
        is_safe = False
        confidence *= 0.9
        issues.append({
            "type": "low_saturation",
            "severity": "low",
            "description": "Low color saturation might indicate the food is past its prime",
            "value": round(quality_metrics['avg_saturation'], 2)
        })
        recommendations.append("Check the expiration date and storage conditions")
    
    # Generate response message
    if is_safe:
        message = f"The {food_predictions[0]['name']} appears to be in good condition."
        details = "No significant signs of spoilage detected. However, always check expiration dates and proper storage conditions."
        recommendations = ["Check the expiration date", "Verify proper storage conditions", "Look for any unusual odors"]
    else:
        message = f"The {food_predictions[0]['name']} shows signs that might indicate spoilage."
        details = "Issues detected:\n" + "\n".join(f"- {issue['description']} (Severity: {issue['severity']})" for issue in issues)
        details += "\n\nRecommendations:\n" + "\n".join(f"- {rec}" for rec in recommendations)
        details += "\n\nNote: This is an automated analysis. When in doubt, it's better to be safe than sorry."
    
    return {
        "isSafe": is_safe,
        "confidence": round(confidence * 100, 2),
        "message": message,
        "details": details,
        "analysis": analysis_details,
        "issues": issues,
        "recommendations": recommendations,
        "identified_food": food_predictions[0]
    }

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Read and validate the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Analyze the image
        result = analyze_food_image(image_array)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
