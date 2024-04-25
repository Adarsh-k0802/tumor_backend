from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Load the trained model
model = tf.keras.models.load_model("CN96.42.h5")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Class labels mapping
class_labels = {
    0: "Glioma",
    1: "Meningioma",
    2: "Pituitary"
}

# Function to preprocess the image
def preprocess_image(image_bytes):
    # Open image using PIL
    image = Image.open(io.BytesIO(image_bytes))
    # Convert image to numpy array
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    # You might need to add more preprocessing steps depending on your model
    return image

# Define endpoint to receive MRI image and return score
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image as bytes
        contents = await file.read()
        # Preprocess the image
        processed_image = preprocess_image(contents)
        # Make prediction using the loaded model
        prediction = model.predict(np.expand_dims(processed_image, axis=0))
        # Convert prediction to JSON serializable format
        prediction = prediction.tolist()  # Convert NumPy array to Python list
        # Get the index of the maximum score
        max_index = np.argmax(prediction[0])
        # Print the scores in the terminal
        print("Scores:", prediction)
        # Get the corresponding class label based on the index
        predicted_label = class_labels[max_index]
        # Return the predicted class label
        return {"predicted_label": predicted_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
