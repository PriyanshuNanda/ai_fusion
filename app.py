import os
import io
from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
import torch  # If you use PyTorch for your actual model
import onnxruntime  # For ONNX models


app = Flask(__name__)

# Set up upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model file (assuming it's in the same directory)
MODEL_PATH = 'model.pth'  # or 'your_model.onnx' -- REPLACE WITH YOUR ACTUAL FILE NAME AND EXTENSION
# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")


###############################
# Model Loading and Processing
###############################
    

def load_model():
    """Loads the model from the specified file path (pth or onnx)."""
    try:
        if MODEL_PATH.endswith('.pth'):
            print("Loading PyTorch model")
            # Ensure you have CUDA available if your model uses it.
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = torch.load(MODEL_PATH, map_location=device)
            model.eval()  # Set to evaluation mode
            return model


        elif MODEL_PATH.endswith('.onnx'): #ONNX file
            print("Loading ONNX model...")
            model = onnxruntime.InferenceSession(MODEL_PATH)
            return model

        else:
            raise ValueError("Unsupported model file format.  Must be .pth or .onnx")

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
        if MODEL_PATH.endswith('.pth'):
            # Preprocess the image (resize, normalize, etc.)  Adapt to your model's requirements
            image = image.resize((256, 256))  # Example: resize
            img_tensor = preprocess_pytorch(image) # Example: convert PIL image to torch tensor
            # Perform inference
            with torch.no_grad():  # Disable gradient calculation for inference
                output = model(img_tensor)  # Replace with your model's prediction method

            # Post-process the output (convert numpy array to PIL image, etc.) Adjust to your model's output
            processed_image = postprocess_pytorch(output)  # Convert back to PIL

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
            raise ValueError("Unsupported model file format. Must be .pth or .onnx.")


    except Exception as e:
        raise RuntimeError(f"Error processing image: {e}")

    return processed_image

###########################
# Helper Functions for Preprocessing and Postprocessing
###########################

def preprocess_pytorch(image):
    """Preprocesses a PIL image for PyTorch model input."""
    # Example: Resize, convert to torch tensor, normalize
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL Image to tensor (C, H, W) and scales to [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization
    ])

    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension (B, C, H, W)
    return img_tensor


def postprocess_pytorch(output_tensor):
    """Postprocesses the PyTorch model output to a PIL image."""
    # Example: Convert back to PIL, scale values, etc.
    import torchvision.transforms as transforms
    transform = transforms.ToPILImage()  # Converts a torch.*Tensor of range [0, 1] or a NumPy ndarray of dtype uint8 to a PIL Image

    # Inverse Normalize (if you normalized)
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    output_tensor = inv_normalize(output_tensor[0].cpu()).clamp(0, 1) #Bring back to 0 to 1 range

    # Convert to PIL Image
    img = transform(output_tensor)

    return img


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