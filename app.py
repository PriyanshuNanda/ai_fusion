import os
import io
from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
import pickle  # Import pickle
import torch  # If you use PyTorch for your actual model
import onnxruntime  # For ONNX models


app = Flask(__name__)

# Set up upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model file (assuming it's in the same directory)
MODEL_PATH = 'your_model.pkl'  # or 'your_model.onnx' -- REPLACE WITH YOUR ACTUAL FILE NAME AND EXTENSION
# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")


###############################
# Model Loading and Processing
###############################

def load_model():
    """Loads the model from the specified file path (pkl or onnx)."""
    try:
        if MODEL_PATH.endswith('.pkl'): # Pickle file
            print("Loading Pickle model...")
            with open(MODEL_PATH, 'rb') as f:  # Open in binary read mode
                model = pickle.load(f)  # Load the model
            return model


        elif MODEL_PATH.endswith('.onnx'): #ONNX file
            print("Loading ONNX model...")
            model = onnxruntime.InferenceSession(MODEL_PATH)
            return model

        else:
            raise ValueError("Unsupported model file format.  Must be .pkl or .onnx")

    except Exception as e:
         raise RuntimeError(f"Error loading model: {e}")


# Load the model once at startup
try:
    model = load_model()
except RuntimeError as e:
    print(e) # Print to console for debugging
    model = None # Set to None so processing doesn't proceed.


def process_image(image):
    """Processes the image using the loaded model."""
    if model is None:
        raise RuntimeError("Model not loaded.  Cannot process image.")

    # Ensure image is in RGB mode
    image = image.convert("RGB")

    try:
        if MODEL_PATH.endswith('.pkl'):
            # Preprocess the image (resize, normalize, etc.)  Adapt to your model's requirements
            image = image.resize((256, 256))  # Example: resize
            img_array = preprocess_pickle(image) # Example: convert PIL image to numpy array
            # Perform inference
            output = model.predict(img_array) # Replace with your model's prediction method

            # Post-process the output (convert numpy array to PIL image, etc.) Adjust to your model's output
            processed_image = postprocess_pickle(output)  # Convert back to PIL

        elif MODEL_PATH.endswith('.onnx'):

            # Preprocess the image (resize, normalize, etc.) Adapt to your model's input requirements
            image = image.resize((256, 256))  # Example: resize
            img_array = preprocess_onnx(image) # Example: Convert PIL to Numpy array
            # ONNX inference
            input_name = model.get_inputs()[0].name #Get input name from ONNX model
            output = model.run(None, {input_name: img_array})[0] #Run inference

            # Post-process the output (convert numpy array to PIL image, etc.)
            processed_image = postprocess_onnx(output) # Convert back to PIL

        else:
            raise ValueError("Unsupported model file format. Must be .pkl or .onnx.")


    except Exception as e:
        raise RuntimeError(f"Error processing image: {e}")

    return processed_image

###########################
# Helper Functions for Preprocessing and Postprocessing
###########################

def preprocess_pickle(image):
    """Preprocesses a PIL image for Pickle model input."""
    # Example: Resize, convert to numpy array, normalize
    import numpy as np
    img_array = np.array(image).astype(np.float32)
    img_array = img_array / 255.0 # Normalize
    #No batch dimension here as it is a numpy array.
    return img_array


def postprocess_pickle(output_array):
    """Postprocesses the Pickle model output to a PIL image."""
    # Example: Convert back to PIL, scale values, etc.
    import numpy as np
    output_array = np.clip(output_array * 255, 0, 255).astype(np.uint8) # Scale and clip
    return Image.fromarray(output_array) #Convert to PIL


def preprocess_onnx(image):
    """Preprocesses a PIL image for ONNX model input."""
    # Example: Resize, convert to numpy array, normalize
    import numpy as np
    img_array = np.array(image).astype(np.float32)
    img_array = img_array / 255.0 # Normalize
    img_array = np.expand_dims(img_array.transpose((2, 0, 1)), axis=0)  # Add batch dimension and change to channel-first
    return img_array

def postprocess_onnx(output_array):
    """Postprocesses the ONNX model output to a PIL image."""
    # Example: Convert back to PIL, scale values, etc.
    import numpy as np
    output_array = np.squeeze(output_array, axis=0) #Remove batch dimension
    output_array = output_array.transpose((1, 2, 0))  # Change to channel-last
    output_array = np.clip(output_array * 255, 0, 255).astype(np.uint8) # Scale and clip
    return Image.fromarray(output_array) #Convert to PIL



###########################
# Flask Routes
###########################

@app.route('/')
def index():
    # Serve the frontend page
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    # Check if an image file was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file securely
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # Open the image using PIL
        image = Image.open(file_path)
    except Exception as e:
        return jsonify({'error': 'Invalid image file'}), 400

    try:
        # Process the image using the loaded model
        output_image = process_image(image)

        # Save the output image to a BytesIO stream (so we can send it back without saving to disk)
        img_io = io.BytesIO()
        output_image.save(img_io, 'JPEG')
        img_io.seek(0)

        # Return the processed image to the client
        return send_file(img_io, mimetype='image/jpeg')


    except RuntimeError as e:
        # Handle model loading or processing errors gracefully
        return jsonify({'error': str(e)},), 500


if __name__ == '__main__':
    app.run(debug=True)