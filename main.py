import os
from flask import Flask, request, send_file, jsonify
from PIL import Image
import io
from ultralytics import YOLO

app = Flask(__name__)


model_path = "runs/detect/train/weights/best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

model = YOLO(model_path)

@app.route('/annotate', methods=['POST'])
def annotate_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        image = Image.open(image_file)
        results = model(image)
        annotated_image_array = results[0].plot()
        annotated_image = Image.fromarray(annotated_image_array)
        byte_io = io.BytesIO()
        annotated_image.save(byte_io, format='JPEG')
        byte_io.seek(0)
        return send_file(byte_io, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
