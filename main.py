from io import BytesIO
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import numpy as np
import os
from ultralytics import YOLO
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all messages, 1 = info, 2 = warnings, 3 = errors only

import tensorflow as tf

app = FastAPI()

# CORS configuration
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify your frontend URL)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model from the directory containing saved_model.pb
# MODEL = tf.keras.models.load_model("C:/Users/admin/Desktop/Project/potatoes.h5",compile=False)  # Adjust path to match your directory structure
MODEL_PATH = os.getenv("MODEL_PATH", "potatoes.h5")  # Default value if not set
MODEL = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Load YOLOv8 pre-trained model
YOLO_MODEL = YOLO('yolov8n.pt')  # Ensure the correct YOLOv8 nano model is used

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

# @app.post("/demo")
# async def demo(Name: str = Form(...)):
#     return {"Name": Name}

def read_file_as_image(data) -> np.ndarray:
    try:
        # Attempt to open the image
        image = Image.open(BytesIO(data))
        
        # Resize image to match the model's input size (adjust the size as needed)
        image = image.resize((224, 224))  # Adjust the size to match the model's expected input size
        
        # Convert to a NumPy array
        image_array = np.array(image)
        
        # Normalize the image if the model expects it (e.g., scaling pixel values to [0, 1])
        # image_array = image_array / 255.0  # Uncomment if normalization is required
        
        return image_array
    except Exception as e:
        print(f"Error reading image: {e}")  # Debug statement
        raise ValueError("Invalid image file") from e

def detect_leaf_or_plant(image_array: np.ndarray) -> bool:
    """
    Detects if an image contains a plant-like object using YOLOv8.
    """
    try:
        image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        results = YOLO_MODEL(image)

        print("Available labels:", results[0].names)  # Debug labels
        
        CONFIDENCE_THRESHOLD = 0.3
        PLANT_LIKE_CLASSES = ["potted plant", "vase", "bottle", "bowl"]  # Expanded classes

        for box in results[0].boxes:
            label = results[0].names[int(box.cls)]
            confidence = box.conf

            if confidence > CONFIDENCE_THRESHOLD and label in PLANT_LIKE_CLASSES:
                print(f"Detected: {label} with {confidence:.2f} confidence")
                return True

        return False
    except Exception as e:
        print(f"Error during object detection: {e}")
        return False

def detect_leaf_with_orb(image_array: np.ndarray, sample_image_array: np.ndarray) -> bool:
    """
    Detect if the input image contains a leaf using ORB feature matching.
    """
    try:
        # Convert input and sample images to grayscale
        input_image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        sample_image_gray = cv2.cvtColor(sample_image_array, cv2.COLOR_RGB2GRAY)

        # Initialize ORB detector
        orb = cv2.ORB_create()

        # Detect keypoints and compute descriptors
        kp1, des1 = orb.detectAndCompute(input_image_gray, None)
        kp2, des2 = orb.detectAndCompute(sample_image_gray, None)

        # Debugging: Log the number of keypoints detected
        print(f"Keypoints in input image: {len(kp1) if kp1 else 0}")
        print(f"Keypoints in sample image: {len(kp2) if kp2 else 0}")

        # If no descriptors are found, return False
        if des1 is None or des2 is None:
            print("No descriptors found in one or both images.")
            return False

        # Create BFMatcher and match descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Calculate match score
        match_score = len(matches)
        print(f"Match score: {match_score}")  # Debugging: Log match score

        # Check match distances for quality
        MATCH_THRESHOLD = 3  # Lowered threshold for good matches
        DISTANCE_THRESHOLD = 100  # Increased distance threshold for flexibility
        good_matches = [m for m in matches if m.distance < DISTANCE_THRESHOLD]
        print(f"Good matches: {len(good_matches)}")  # Debugging: Log good matches

        # Return True only if there are enough good matches
        return len(good_matches) >= MATCH_THRESHOLD
    except Exception as e:
        print(f"Error during ORB feature matching: {e}")
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
async def predict(file: UploadFile = File(...), sample_file: UploadFile = None):
    try:
        # Read the input image file as binary data
        image_data = await file.read()
        image_array = read_file_as_image(image_data)

        # Check if sample_file is provided
        if sample_file:
            # Read the sample leaf image file as binary data
            sample_image_data = await sample_file.read()
            sample_image_array = read_file_as_image(sample_image_data)

            # Step 1: Detect leaf using ORB feature matching
            if detect_leaf_with_orb(image_array, sample_image_array):
                # Step 2: Perform inference using the primary TensorFlow model
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

            print("ORB feature matching failed. Falling back to YOLOv8.")
            # Fallback to YOLOv8 for leaf detection
            if detect_leaf_or_plant(image_array):
                return JSONResponse(content={
                    "message": "Leaf detected using YOLOv8 fallback.",
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "file_size": len(image_data),
                })

            return JSONResponse(content={"error": "No leaf detected in the image."}, status_code=400)

        # If sample_file is not provided, return an error
        return JSONResponse(content={"error": "Sample file is required for leaf detection."}, status_code=400)

    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        print(f"Unexpected error: {e}")
        return JSONResponse(content={"error": "An unexpected error occurred."}, status_code=500)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))  # Render assigns a dynamic port
    uvicorn.run(app, host="0.0.0.0", port=port)
