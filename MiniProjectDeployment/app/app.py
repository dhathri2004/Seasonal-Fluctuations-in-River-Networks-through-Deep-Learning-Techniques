import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request
from PIL import Image
import base64
import requests
import io

# Set TensorFlow logging level to ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)

app = Flask(__name__)

# Define the path to your model
model_path = r"C:\Users\kcrav\trail1\MiniProject1.h5"  # Adjust the path to your model

# Load the model
model = load_model(model_path)

# Define function to calculate water percentage
def calculate_water_percentage(image):
    img = image.resize((128, 128))  # Resize image to match model input shape
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)
    mask = model.predict(img_array)[0]
    water_pixels = np.sum(mask >= 0.5)  # Count white pixels in the mask (representing water)
    total_pixels = mask.shape[0] * mask.shape[1]
    water_percentage = (water_pixels / total_pixels) * 100
    return water_percentage

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


@app.route('/')
def index():
    return render_template("input.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Ensure an image is uploaded
        if 'image' not in request.files:
            return render_template("output.html", error="No image file uploaded")
        
        # Read uploaded image
        image_file = request.files['image']
        
        # Check if the file is not empty
        if image_file.filename == '':
            return render_template("output.html", error="No selected file")

        # Check if the file is allowed
        if image_file and allowed_file(image_file.filename):
            # Open image file and calculate water percentage
            image = Image.open(image_file)
            water_percentage = calculate_water_percentage(image)

            # Determine result based on water percentage
            if water_percentage >= 1.5:
                result = "The image contains water ({}% water)".format(round(water_percentage, 2))
            else:
                result = "The image does not contain enough water ({}% water)".format(round(water_percentage, 2))

            # Render the output.html template with the result
            return render_template("output.html", predict=result)

        else:
            return render_template("output.html", error="Invalid file format. Allowed formats are PNG, JPG, and JPEG")


if __name__ == '__main__':
    app.run(debug=True)
