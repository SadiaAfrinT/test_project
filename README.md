# Chatbot UI Element Analysis Tool

This project is a web-based tool designed to help researchers analyze chatbot user interfaces. It uses a Vision Language Model (VLM) based on the LLaVA architecture to detect and evaluate UI elements in screenshots and videos.

The application provides a user-friendly interface to upload an image or video, run an analysis, and view the detected UI elements and evaluation metrics (Precision, Recall, F1-Score).

**Note:** The web application currently uses a **simulated model** for analysis. The actual model must be trained separately using the provided training script on a machine with a powerful GPU.

## Project Structure

```
.
├── annotations/          # **IMPORTANT**: Your YOLO .txt annotation files go here
├── backend/
│   └── app.py              # Flask backend with the API
├── data/                   # Old directory for COCO format, can be removed
│   └── result.json
├── frontend/
│   ├── index.html          # Main application UI
│   ├── script.js           # Frontend JavaScript logic
│   └── styles.css          # UI styles
├── images/                 # **IMPORTANT**: Your image files go here
├── model/
│   └── train.py            # Script to fine-tune the LLaVA model with YOLO data
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
**Note:** The `torch`, `transformers`, `peft` and `bitsandbytes` packages are large. The installation may take some time.

## How to Run the Application

The application's primary purpose is to serve as a platform for training and evaluating your model. The steps below focus on preparing your data for training.

### 1. Prepare Your Data for YOLO Training
The `model/train.py` script now expects your data in **YOLO format**.

1.  **Place Images:** Copy all your image files (`.png`, `.jpg`, etc.) into the `images/` directory.
2.  **Place Annotations:** Copy all your corresponding YOLO annotation files (`.txt`) into the `annotations/` directory. The basename of each `.txt` file must match the basename of its corresponding image file (e.g., `image1.png` and `image1.txt`).
3.  **Update Class Names:** This is a critical step. Open `model/train.py` and find the `CLASS_NAMES` list. You **must** edit this list so that the names of your object classes are in the correct order, matching the integer class IDs in your `.txt` files. For example, if "Cards" is class `2` in your YOLO files, it must be the third item in the list (index 2).

### 2. Run the Web Application
The backend serves the frontend and provides a simulated API. To start it, run the following command from the project's root directory:
```bash
python backend/app.py
```
You can then access the web application at: **http://127.0.0.1:5000**

## Model Training and Fine-Tuning

The core of this project is the fine-tuned LLaVA model. You must train it yourself using your labeled data.

### How to Train the Model
1.  **Ensure Prerequisites:** Make sure you have a machine with a powerful GPU and have installed all dependencies.
2.  **Verify Data:** Confirm that your data is set up correctly as described in the "Prepare Your Data" section above.
3.  **Run the Training Script:** Execute the `train.py` script from the root directory:

    ```bash
    python model/train.py
    ```

**IMPORTANT:**
- Model training is a computationally expensive process.
- The script uses 4-bit quantization and LoRA to reduce memory usage, but it still requires a modern GPU.
- The script will save the fine-tuned model adapters into a new directory: `model/llava-finetuned-yolo-model/`.

### Integrating the Fine-Tuned Model
Once training is complete, you would modify the `backend/app.py` file to load your fine-tuned model instead of generating mock data. This involves:
1.  Loading the base model and attaching the trained LoRA adapters from `model/llava-finetuned-yolo-model/`.
2.  Implementing the image processing and inference logic within the `/api/analyze` endpoint.

## Extensible Pipeline

This project is designed to be easily extensible. To improve your model with more data:
1.  **Add New Images:** Add your new screen captures to the `images/` folder.
2.  **Add New Annotations:** Add the corresponding new YOLO `.txt` files to the `annotations/` folder.
3.  **Re-run Training:** Execute `python model/train.py` again. The script will automatically pick up the new data to fine-tune the model.
