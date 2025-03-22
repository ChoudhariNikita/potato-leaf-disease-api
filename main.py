from io import BytesIO
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import numpy as np
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all messages, 1 = info, 2 = warnings, 3 = errors only

app = FastAPI()

# CORS configuration
origins = ["http://localhost:8081","http://42.104.228.121:8081", "http://192.168.43.65:8081","exp://192.168.43.65:8081","*"]  # Allow requests from Expo Go app and localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],# Use the updated origins list
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Load the model from the directory containing saved_model.pb
MODEL_PATH = os.getenv("MODEL_PATH", "potatoes.h5")  # Default value if not set
MODEL = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    try:
        # Attempt to open the image
        image = Image.open(BytesIO(data))
        
        # Resize image to match the model's input size (adjust the size as needed)
        image = image.resize((224, 224))  # Adjust the size to match the model's expected input size
        
        # Convert to a NumPy array
        image_array = np.array(image)
        
        return image_array
    except Exception as e:
        print(f"Error reading image: {e}")  # Debug statement
        raise ValueError("Invalid image file") from e

def detect_leaf_with_orb(image_array: np.ndarray) -> bool:
    """
    Detect if the input image contains a leaf using ORB feature matching.
    """
    try:
        # Placeholder logic for ORB-based detection
        # Since no sample image is provided, we cannot perform matching
        # Instead, we can use the number of keypoints as a heuristic
        return True  # Simplified logic
    except Exception as e:
        print(f"Error during ORB feature detection: {e}")
        return False

def predict_with_tensorflow(image_data):
    """
    Perform inference using the TensorFlow model.
    """
    try:
        # Decode and preprocess the image
        image = tf.image.decode_jpeg(image_data, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.expand_dims(image, axis=0)
        
        # Perform prediction
        predictions = MODEL.predict(image)
        
        # Get the predicted class and confidence
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_index]
        confidence_score = float(predictions[0][predicted_class_index])
        
        return predicted_class, confidence_score
    except Exception as e:
        print(f"Error during TensorFlow inference: {e}")
        return None, str(e)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the input image file as binary data
        image_data = await file.read()
        image_array = read_file_as_image(image_data)

        # Detect leaf using ORB feature matching
        if detect_leaf_with_orb(image_array):
            # Perform inference using the primary TensorFlow model
            predicted_class, confidence = predict_with_tensorflow(image_data)

            if predicted_class is None:
                return JSONResponse(content={"error": confidence}, status_code=400)

            return JSONResponse(content={
                "class": predicted_class,
                "confidence": confidence,
                "filename": file.filename,
                "content_type": file.content_type,
                "file_size": len(image_data),
            })

        return JSONResponse(content={"error": "No leaf detected in the image."}, status_code=400)

    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        print(f"Unexpected error: {e}")
        return JSONResponse(content={"error": "An unexpected error occurred."}, status_code=500)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))  # Render assigns a dynamic port
    uvicorn.run(app, host="0.0.0.0", port=port)
