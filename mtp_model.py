"""
Flask app for uploading images and running GeoDeep object detection with a modern UI.
Features:
- Upload JPG/PNG/GeoTIFF images.
- Select GeoDeep model (cars, trees, birds, planes, aerovision).
- Displays detections with bounding boxes in the browser.

Requirements:
    pip install flask pillow rasterio geodeep
"""
from flask import Flask, request, send_file, render_template_string
from werkzeug.utils import secure_filename
import os, tempfile, io
from PIL import Image, ImageDraw
import numpy as np
import rasterio
from rasterio.transform import from_origin
from geodeep import detect

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'tif', 'tiff', 'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GeoDeep Object Detection</title>
  <style>
    body { font-family: Poppins, sans-serif; background: #f8fafc; color: #222; text-align: center; }
    .container { width: 90%%; max-width: 700px; margin: 40px auto; background: #fff; padding: 30px; border-radius: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
    input[type=file], select, button { width: 100%%; margin-top: 10px; padding: 12px; border-radius: 10px; border: 1px solid #ccc; font-size: 15px; }
    button { background: #2563eb; color: white; border: none; cursor: pointer; transition: 0.3s; }
    button:hover { background: #1d4ed8; }
    img { margin-top: 25px; max-width: 100%%; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.1); }
  </style>
</head>
<body>
  <div class="container">
    <h1>🛰️ GeoDeep Object Detection</h1>
    <form id="upload-form" enctype="multipart/form-data" method="post" action="/upload">
      <input type="file" name="file" accept=".jpg,.jpeg,.png,.tif,.tiff" required>
      <select name="model">
        <option value="cars">Cars</option>
        <option value="trees">Trees</option>
        <option value="birds">Birds</option>
        <option value="planes">Planes</option>
        <option value="aerovision">AeroVision</option>
      </select>
      <button type="submit">Upload & Detect</button>
    </form>
    <div id="result"></div>
  </div>
  <script>
    const form = document.getElementById('upload-form');
    form.onsubmit = async (e) => {
      e.preventDefault();
      const resultDiv = document.getElementById('result');
      resultDiv.innerHTML = '<p>⏳ Running detection... Please wait.</p>';
      const formData = new FormData(form);
      const response = await fetch('/upload', { method: 'POST', body: formData });
      if (response.ok) {
        const blob = await response.blob();
        const imgUrl = URL.createObjectURL(blob);
        resultDiv.innerHTML = `<h3>Detections:</h3><img src="${imgUrl}" alt="Detections">`;
      } else {
        resultDiv.innerHTML = `<p style='color:red'>Error: ${response.statusText}</p>`;
      }
    };
  </script>
</body>
</html>
'''

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    file = request.files['file']
    if file.filename == '':
        return 'Empty filename', 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        ext = filename.rsplit('.', 1)[1].lower()
        geotiff_path = path if ext in ('tif', 'tiff') else convert_image_to_geotiff(path)
        model = request.form.get('model', 'cars')
        try:
            bboxes, scores, classes = detect(geotiff_path, model)
        except Exception as e:
            return f'GeoDeep Error: {e}', 500

        img_bytes = draw_boxes_on_image(path, bboxes, scores, classes)
        return send_file(io.BytesIO(img_bytes), mimetype='image/png')

    return 'Unsupported file type', 400

def convert_image_to_geotiff(image_path):
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img)
    h, w = arr.shape[:2]
    transform = from_origin(0, 0, 1, 1)
    fd, tif_path = tempfile.mkstemp(suffix='.tif', dir=app.config['UPLOAD_FOLDER'])
    os.close(fd)
    with rasterio.open(
        tif_path, 'w', driver='GTiff', height=h, width=w, count=3,
        dtype=arr.dtype, crs='EPSG:3857', transform=transform
    ) as dst:
        dst.write(np.moveaxis(arr, -1, 0))
    return tif_path

def draw_boxes_on_image(image_path, bboxes, scores, classes, min_score=0.2):
    img = Image.open(image_path).convert('RGBA')
    draw = ImageDraw.Draw(img)
    for bbox, score, cls in zip(bboxes, scores, classes):
        if score < min_score:
            continue
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 200), width=3)
        label = f"{cls[1]} {score:.2f}"
        draw.text((x1 + 4, y1 + 4), label, fill=(255, 255, 255, 255))
    out = io.BytesIO()
    img.save(out, format='PNG')
    out.seek(0)
    return out.read()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
