import os
from flask import Flask, request, jsonify, send_from_directory
import random
import time

# Create the Flask app
app = Flask(__name__, static_folder='../frontend')

# --- API Endpoints ---

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """
    This is a mock endpoint to simulate the analysis of an uploaded image or video.
    In a real application, this endpoint would:
    1. Receive the file.
    2. Preprocess it.
    3. Pass it to the fine-tuned LLaVA model for inference.
    4. Process the model's output to get detected objects and their bounding boxes.
    5. Calculate evaluation metrics if ground truth is available.
    6. Return the results as JSON.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Simulate model processing time
    time.sleep(2)

    # Simulate model output
    # In a real implementation, you would run your LLaVA model here
    # and generate the actual list of detected objects.
    possible_objects = ["Chat Message", "Button", "Quick Replies", "Carousels", "Typing Indicator", "Accordion"]
    detected_objects = []
    for _ in range(random.randint(2, 6)):
        detected_objects.append({
            "class": random.choice(possible_objects),
            "confidence": random.uniform(0.75, 0.99),
            "bbox": [random.randint(10, 400), random.randint(10, 400), random.randint(50, 200), random.randint(20, 100)]
        })

    # Simulate evaluation metrics
    # In a real scenario, you'd compare model predictions to a ground truth dataset.
    evaluation_metrics = {
        "precision": random.uniform(0.85, 0.95),
        "recall": random.uniform(0.80, 0.92),
        "f1-score": random.uniform(0.82, 0.94)
    }

    return jsonify({
        "detected_objects": detected_objects,
        "evaluation": evaluation_metrics
    })

# --- Frontend Serving ---

@app.route('/')
def index():
    """Serves the main index.html file."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serves other static files like CSS, JS, or images."""
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    # It's recommended to use a production-ready WSGI server like Gunicorn or Waitress
    # instead of Flask's built-in development server for real deployments.
    app.run(debug=True, port=5000)
