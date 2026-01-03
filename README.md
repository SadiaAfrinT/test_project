# Chatbot UI Element Analysis Tool

This project is a web-based tool designed to help researchers analyze chatbot user interfaces. It uses a Vision Language Model (VLM) based on the LLaVA architecture to detect and evaluate UI elements in screenshots and videos.

The application provides a user-friendly interface to upload an image or video, run an analysis, and view the detected UI elements and evaluation metrics (Precision, Recall, F1-Score).

**Note:** The web application currently uses a **simulated model** for analysis. The actual model must be trained separately using the provided training script on a machine with a powerful GPU.

## Project Structure

```
.
├── backend/
│   └── app.py              # Flask backend with the API
├── data/
│   └── result.json         # Your COCO-formatted annotations
├── frontend/
│   ├── index.html          # Main application UI
│   ├── script.js           # Frontend JavaScript logic
│   └── styles.css          # UI styles
├── images/                 # **IMPORTANT**: Your image files go here
├── model/
│   └── train.py            # Script to fine-tune the LLaVA model
├── my_tool.py
├── my_tool_test.py
├── planning.docx
├── ui_design.html
└── requirements.txt        # Python dependencies
```

## Setup and Installation

### 1. Prerequisites
- Python 3.8+
- `pip` for installing Python packages
- A powerful NVIDIA GPU with CUDA installed (for actual model training)

### 2. Install Dependencies
Clone the repository and install the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```
**Note:** The `torch`, `transformers`, and `bitsandbytes` packages are large. The installation may take some time.

## How to Run the Application

### 1. Prepare the Image Data
Your `data/result.json` file contains references to image file paths. You must gather all the corresponding image files and place them into the `images/` directory at the root of the project.

**The `model/train.py` script expects all dataset images to be in this folder.**

### 2. Run the Web Application
The backend is a Flask application that serves the frontend and provides the (simulated) analysis API.

To run the web server, execute the following command from the project's root directory:
```bash
python backend/app.py
```
This will start the development server. You can access the web application by opening your browser and navigating to:
**http://127.0.0.1:5000**

You can now select an image or video file and click "Run Analysis" to see the simulated results.

## Model Training and Fine-Tuning

The core of this project is the fine-tuned LLaVA model. You must train it yourself using your labeled data.

### How to Train the Model
1.  **Ensure Prerequisites:** Make sure you have a machine with a powerful GPU and have installed all dependencies.
2.  **Verify Data:** Confirm that your images are in the `images/` directory and your `data/result.json` annotation file is correct.
3.  **Run the Training Script:** Execute the `train.py` script from the root directory:

    ```bash
    python model/train.py
    ```

**IMPORTANT:**
- Model training is a computationally expensive and time-consuming process.
- The script uses 4-bit quantization (`load_in_4bit=True`) to reduce memory usage, but it will still require a modern GPU.
- The `train.py` script will save the fine-tuned model and processor files into a new directory: `model/llava-finetuned-model/`.

### Integrating the Fine-Tuned Model
Once training is complete, you would modify the `backend/app.py` file to load your fine-tuned model instead of generating mock data. This involves:
1.  Loading the model and processor from `model/llava-finetuned-model/`.
2.  Implementing the image processing and inference logic within the `/api/analyze` endpoint.

## Extensible Pipeline

This project is designed to be easily extensible. To improve your model with more data:
1.  **Add New Images:** Add your new screen captures to the `images/` folder.
2.  **Label Your Data:** Use your labeling tool (e.g., Label Studio) to create new annotations for the new images.
3.  **Update Annotations:** Export the new annotations and merge them with the existing `data/result.json` file, or replace it with a new file containing all annotations.
4.  **Re-run Training:** Execute the `python model/train.py` script again to fine-tune the model on the updated dataset. The trainer will continue from the last checkpoint if available.
