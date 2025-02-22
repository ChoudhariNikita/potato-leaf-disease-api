from io import BytesIO
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import numpy as np
import os

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
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model from the directory containing saved_model.pb
MODEL = tf.keras.models.load_model("C:/Users/admin/Desktop/Project/potatoes.h5",compile=False)  # Adjust path to match your directory structure
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

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


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the file as an image
        image_array = read_file_as_image(await file.read())
        
        # Ensure image_array is not None and contains data
        if image_array is None:
            raise ValueError("Failed to process the image file.")
        
        # Expand dimensions and make prediction
        image_batch = np.expand_dims(image_array, axis=0)
        predictions = MODEL.predict(image_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        return JSONResponse(content={
            "class": predicted_class,
            "confidence": float(confidence),
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": image_array.size,
        })
    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        print(f"Unexpected error: {e}")
        return JSONResponse(content={"error": "An unexpected error occurred."}, status_code=500)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))  # Render assigns a dynamic port
    uvicorn.run(app, host="0.0.0.0", port=port)
