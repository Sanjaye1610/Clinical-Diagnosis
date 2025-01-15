from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image, ExifTags
import os

# Flask app setup
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "model/xray_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (update based on your dataset)
CLASS_LABELS = {0: "Normal", 1: "Pneumonia"}  # Example labels

# Helper function to preprocess the image
def preprocess_image(image_path):
    IMAGE_SIZE = (224, 224)
    image = load_img(image_path, target_size=IMAGE_SIZE)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0  # Normalize to [0, 1]
    return image

# Helper function to validate if an image is an X-ray
def is_valid_xray(image_path):
    try:
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        width, height = image.size
        aspect_ratio = width / height

        # Check if the image aspect ratio is reasonable for X-rays
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False

        # Check pixel intensity distribution for X-ray characteristics
        pixel_data = np.array(image)
        mean_pixel_value = np.mean(pixel_data)
        std_pixel_value = np.std(pixel_data)

        # X-ray images tend to have a specific intensity and contrast range
        if mean_pixel_value < 50 or mean_pixel_value > 200:
            return False
        if std_pixel_value < 10 or std_pixel_value > 80:  # Validate contrast range
            return False

        # Further checks: Ensure the image has enough edges or structure
        edges = np.sum(np.abs(np.gradient(pixel_data)))
        if edges < 10000:  # Threshold for detecting structural details in X-rays
            return False

        # Optional: Check for EXIF metadata to reject photos with camera data
        exif_data = image._getexif() if hasattr(image, '_getexif') else None
        if exif_data:
            for tag, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                if tag_name in ["Make", "Model", "Software"]:  # Common camera metadata
                    return False

        return True
    except Exception as e:
        print(f"Image validation error: {e}")
        return False

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("result.html", error="No file uploaded!")

    file = request.files["file"]

    if file.filename == "":
        return render_template("result.html", error="No selected file!")

    # Save the uploaded file temporarily
    file_path = os.path.join("static", file.filename)
    file.save(file_path)

    # Validate the image
    if not is_valid_xray(file_path):
        os.remove(file_path)  # Clean up the file
        return render_template("result.html", error="The uploaded image is not a valid X-ray. Please upload a valid X-ray image!")

    # Preprocess the image
    image = preprocess_image(file_path)

    # Get predictions
    predictions = model.predict(image)
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index]

    result = {
        "class": CLASS_LABELS[class_index],
        "confidence": f"{confidence * 100:.2f}%",
    }

    # Cleanup uploaded file
    os.remove(file_path)

    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
