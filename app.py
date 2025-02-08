import os
import io
from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
import torch  # If you use PyTorch for your actual model

app = Flask(__name__)

# Set up upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

###############################
# Model Loading and Processing
###############################

def load_model():
    """
    Simulate loading a model from a checkpoint file.
    In a real scenario, you might load your PyTorch model as follows:
    
      model = MyModel()
      checkpoint = torch.load('model.ckpt', map_location=torch.device('cpu'))
      model.load_state_dict(checkpoint['state_dict'])
      model.eval()
      return model
    
    For demonstration, we will simply print a message.
    """
    print("Loading model checkpoint...")
    model = None  # Replace with your actual model instance.
    return model

# Load the model once at startup
model = load_model()

def process_image(image):
    """
    Simulate processing the image using the loaded model.
    Replace this function with your model inference code.
    
    For demonstration, we will invert the image colors.
    """
    # Ensure image is in RGB mode
    image = image.convert("RGB")
    # Invert the image colors as a dummy processing step
    processed_image = ImageOps.invert(image)
    return processed_image

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

    # Process the image using our (dummy) model
    output_image = process_image(image)

    # Save the output image to a BytesIO stream (so we can send it back without saving to disk)
    img_io = io.BytesIO()
    output_image.save(img_io, 'JPEG')
    img_io.seek(0)

    # Return the processed image to the client
    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
