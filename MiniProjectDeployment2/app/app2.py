import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, session  # Import session
from PIL import Image

# Set TensorFlow logging level to ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Add a secret key for session

# Define the path to your model
model_path = r"C:\Users\kcrav\trail1\MiniProject2.h5"  # Adjust the path to your model

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
    return render_template("input2.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Ensure both images are uploaded
        if 'image1' not in request.files or 'image2' not in request.files:
            return render_template("output2.html", error="Please upload two images")

        # Read uploaded images
        image_file1 = request.files['image1']
        image_file2 = request.files['image2']
        
        # Check if the files are not empty
        if image_file1.filename == '' or image_file2.filename == '':
            return render_template("output2.html", error="Please select two files")

        # Check if the files are allowed
        if image_file1 and allowed_file(image_file1.filename) and image_file2 and allowed_file(image_file2.filename):
            # Open image files and calculate water percentages
            image1 = Image.open(image_file1)
            image2 = Image.open(image_file2)
            water_percentage1 = calculate_water_percentage(image1)
            water_percentage2 = calculate_water_percentage(image2)

            # Determine the situation based on the comparison of water percentages
            if water_percentage1 > water_percentage2:
                situation = "Flood"
            elif water_percentage1 < water_percentage2:
                situation = "Drought"
            else:
                situation = "Normal"

            # Store the result in session
            session['WaterDetectionOutput'] = situation

            # Redirect to the output page
            return render_template("output2.html")

        else:
            return render_template("output2.html", error="Invalid file format. Allowed formats are PNG, JPG, and JPEG")

if __name__ == '__main__':
    app.run(debug=True)
