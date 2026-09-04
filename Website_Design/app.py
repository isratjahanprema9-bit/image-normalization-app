import os
import cv2
import numpy as np
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def normalize_raster(image_path):
    # Image read (Grayscale / Raster Band)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        return None

    # Min-Max Normalization to 0-255 scale for visualization
    img_float = img.astype(np.float32)
    min_val = np.min(img_float)
    max_val = np.max(img_float)
    
    if max_val - min_val == 0:
        normalized_img = img_float
    else:
        # Normalizing to 0-1 and scaling to 0-255
        normalized_img = ((img_float - min_val) / (max_val - min_val)) * 255.0

    normalized_img = normalized_img.astype(np.uint8)
    
    # Save normalized image
    normalized_filename = 'normalized_' + os.path.basename(image_path)
    normalized_path = os.path.join(UPLOAD_FOLDER, normalized_filename)
    cv2.imwrite(normalized_path, normalized_img)
    
    return normalized_filename, min_val, max_val

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400
        
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Normalize Image
            normalized_file, min_val, max_val = normalize_raster(filepath)
            
            return render_template('index.html', 
                                   original_image=file.filename, 
                                   normalized_image=normalized_file,
                                   min_val=min_val, 
                                   max_val=max_val)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)