import os
import io
from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import torch  # Only needed for .ckpt processing if used
import onnxruntime  # Only needed for ONNX models if used

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)

# Set up upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Specify your model file. In this case, we use the FastAI exported model (.pkl).
MODEL_PATH = 'export.pkl'  # Change if your file has a different name or path

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

###############################
# Model Loading and Processing
###############################

def load_model():
    """
    Loads the model from the specified file path.
    Supports FastAI exported model (.pkl), PyTorch checkpoint (.ckpt) and ONNX (.onnx).
    """
    try:
        if MODEL_PATH.endswith('.pkl'):
            print("Loading FastAI exported model (.pkl)...")
            from fastai.learner import load_learner
            learner = load_learner(MODEL_PATH)
            return learner

        elif MODEL_PATH.endswith('.ckpt'):
            # Example code for loading a PyTorch checkpoint
            print("Loading PyTorch checkpoint...")
            # Replace MyModel() with your actual model class definition
            model = MyModel()
            checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            return model

        elif MODEL_PATH.endswith('.onnx'):
            print("Loading ONNX model...")
            model = onnxruntime.InferenceSession(MODEL_PATH)
            return model

        else:
            raise ValueError("Unsupported model file format. Must be .pkl, .ckpt, or .onnx.")

    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

# Load the model once at startup
try:
    model = load_model()
except RuntimeError as e:
    print(e)  # Print to console for debugging
    model = None  # Set to None so processing doesn't proceed.

def overlay_text(image, text):
    """
    Overlays the given text on the image.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    # Draw text at position (10, 10) with red color
    draw.text((10, 10), text, fill="red", font=font)
    return image

def process_image(image):
    """
    Processes the image using the loaded model.
    For FastAI (.pkl) models, it uses the learner's predict() method.
    For PyTorch and ONNX models, it uses custom preprocessing/postprocessing.
    """
    if model is None:
        raise RuntimeError("Model not loaded. Cannot process image.")

    # Ensure image is in RGB mode
    image = image.convert("RGB")

    try:
        if MODEL_PATH.endswith('.pkl'):
            # Using FastAI learner's predict method
            # Typically returns a tuple: (predicted_label, tensor, probabilities)
            pred = model.predict(image)
            pred_label = str(pred[0])
            # For demonstration, overlay the prediction label on the image
            processed_image = overlay_text(image.copy(), f"Prediction: {pred_label}")

        elif MODEL_PATH.endswith('.ckpt'):
            # Preprocess the image for a PyTorch model (adjust dimensions as needed)
            image = image.resize((256, 256))
            img_tensor = preprocess_pytorch(image)
            with torch.no_grad():
                output = model(img_tensor)
            processed_image = postprocess_pytorch(output)

        elif MODEL_PATH.endswith('.onnx'):
            # Preprocess for ONNX
            image = image.resize((256, 256))
            img_array = preprocess_onnx(image)
            input_name = model.get_inputs()[0].name
            output = model.run(None, {input_name: img_array})[0]
            processed_image = postprocess_onnx(output)

        else:
            raise ValueError("Unsupported model file format. Must be .pkl, .ckpt or .onnx.")

    except Exception as e:
        raise RuntimeError(f"Error processing image: {e}")

    return processed_image

###########################
# Helper Functions for PyTorch/ONNX
###########################

def preprocess_pytorch(image):
    """
    Preprocesses a PIL image for PyTorch model input.
    """
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    return img_tensor

def postprocess_pytorch(output_tensor):
    """
    Postprocesses the PyTorch model output to a PIL image.
    """
    from torchvision import transforms
    to_pil = transforms.ToPILImage()
    img = output_tensor.squeeze(0).cpu()  # Remove batch dimension, move to CPU
    img = img.clamp(0, 1)  # Clamp values between 0 and 1
    img = to_pil(img)
    return img

def preprocess_onnx(image):
    """
    Preprocesses a PIL image for ONNX model input.
    """
    import numpy as np
    img_array = np.array(image).astype(np.float32)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array.transpose((2, 0, 1)), axis=0)  # Add batch dimension & make channel-first
    return img_array

def postprocess_onnx(output_array):
    """
    Postprocesses the ONNX model output to a PIL image.
    """
    import numpy as np
    output_array = np.squeeze(output_array, axis=0)  # Remove batch dimension
    output_array = output_array.transpose((1, 2, 0))  # Convert to channel-last
    output_array = np.clip(output_array * 255, 0, 255).astype(np.uint8)  # Scale and clip
    return Image.fromarray(output_array)

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

        # Save the output image to a BytesIO stream (to send it back without saving to disk)
        img_io = io.BytesIO()
        output_image.save(img_io, 'JPEG')
        img_io.seek(0)

        # Return the processed image to the client
        return send_file(img_io, mimetype='image/jpeg')

    except RuntimeError as e:
        # Handle model loading or processing errors gracefully
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
