from flask import Flask, request, jsonify, render_template, send_file
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLO model
try:
    model = YOLO('./yolov8 model 2.pt')  # Replace with the correct path
except Exception as e:
    app.logger.error(f"Error loading YOLO model: {e}")


def predict_and_save_image(path_test_car, output_image_path):
    try:
        results = model.predict(path_test_car, device='cpu')
        image = cv2.imread(path_test_car)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = box.cls[0].item()
                label = model.names[int(cls)]
                conf = box.conf[0]
                text = f'{label} {conf:.2f}'
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_path, image)
        return output_image_path
    except Exception as e:
        app.logger.error(f"Error processing image: {e}")
        return None


def predict_and_plot_video(video_path, output_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            app.logger.error(f"Error opening video file: {video_path}")
            return None
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb_frame, device='cpu')
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = box.cls[0].item()
                    label = model.names[int(cls)]
                    conf = box.conf[0]
                    text = f'{label} {conf:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            out.write(frame)
        cap.release()
        out.release()
        return output_path
    except Exception as e:
        app.logger.error(f"Error processing video: {e}")
        return None


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        output_path = os.path.join(OUTPUT_FOLDER, f"output_{file.filename}")

        file.save(input_path)

        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
            result_path = predict_and_save_image(input_path, output_path)
        elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
            result_path = predict_and_plot_video(input_path, output_path)
        else:
            return jsonify({"error": f"Unsupported file type: {file_extension}"}), 400

        if result_path:
            return send_file(result_path, as_attachment=True)
        else:
            return jsonify({"error": "Error processing file"}), 500
    return render_template("upload.html")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Use PORT environment variable or default to 10000
    app.run(host="0.0.0.0", port=port)

