# THIS IS A PLACEHOLDER SCRIPT.
# It demonstrates how you would fine-tune a LLaVA model on your custom YOLO-formatted dataset.
# You will need a machine with a powerful GPU and sufficient memory to run this.

import os
import glob
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model

# --- 1. Configuration ---
MODEL_ID = "llava-hf/llava-1.5-7b-hf"  # Example model
IMAGE_DIR = "../images" 
ANNOTATION_DIR = "../annotations" # Directory for YOLO .txt files
OUTPUT_DIR = "./llava-finetuned-yolo-model"

# =========================================================================================
# IMPORTANT: Please update this list with your class names in the correct order.
# The index of the name in the list should match the class ID in your YOLO .txt files.
# For example, if 'Accordion' is class 0, it should be the first item in the list.
# =========================================================================================
CLASS_NAMES = [
    "Accordion", "Action Controls", "Cards", "Carousels", "Chat Message", 
    "Chatbot Interface", "Information Stamps", "Message Reactions", 
    "Persistent Menu", "Quick Replies", "Typing Indicator", "Window Controls"
]
# =========================================================================================


# --- 2. Custom Dataset for YOLO-formatted data ---
class YoloObjectDetectionDataset(Dataset):
    """
    A custom PyTorch dataset to handle YOLO-formatted object detection data
    for use with a VLM like LLaVA.
    Each line in a YOLO annotation file is treated as a separate data point.
    """
    def __init__(self, image_dir, annotation_dir, class_names, processor):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.class_names = class_names
        self.processor = processor
        self.samples = []

        # Find all annotation files
        annotation_files = glob.glob(os.path.join(self.annotation_dir, '*.txt'))

        for ann_file in annotation_files:
            # Find the corresponding image file (assuming same base name)
            base_name = os.path.basename(ann_file).replace('.txt', '')
            # Try to find image with common extensions
            img_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.webp']:
                potential_path = os.path.join(self.image_dir, base_name + ext)
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break
            
            if not img_path:
                print(f"Warning: No corresponding image found for annotation {os.path.basename(ann_file)}. Skipping.")
                continue

            # Read all annotations in the file
            with open(ann_file, 'r') as f:
                lines = f.readlines()
            
            # Each line is a sample
            for i, line in enumerate(lines):
                self.samples.append({'image_path': img_path, 'annotation_line': line})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample['image_path']
        annotation_line = sample['annotation_line']

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
            img_w, img_h = image.size
        except FileNotFoundError:
            print(f"Warning: Image not found at {image_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))

        # Parse YOLO annotation line
        parts = annotation_line.strip().split()
        class_id = int(parts[0])
        cx_norm, cy_norm, w_norm, h_norm = map(float, parts[1:])

        # Convert normalized YOLO bbox to absolute pixel bbox [x, y, width, height]
        abs_w = w_norm * img_w
        abs_h = h_norm * img_h
        abs_x = (cx_norm * img_w) - (abs_w / 2)
        abs_y = (cy_norm * img_h) - (abs_h / 2)
        
        bbox = [round(abs_x, 2), round(abs_y, 2), round(abs_w, 2), round(abs_h, 2)]

        # Get class name
        category = self.class_names[class_id]

        # Correctly format the conversation for the model
        bbox_str = str(bbox)
        prompt = f"USER: <image>\nPlease provide the bounding box for the {category} in the image."
        conversation = f"{prompt} ASSISTANT: {bbox_str}</s>"

        # Process the full conversation and image
        inputs = self.processor(text=conversation, images=image, return_tensors="pt")
        
        # Create labels and mask the prompt
        labels = inputs.input_ids.clone()
        prompt_only_inputs = self.processor(text=f"{prompt} ASSISTANT:")
        prompt_len = len(prompt_only_inputs.input_ids[0])
        labels[0, :prompt_len] = -100

        inputs['labels'] = labels
        return {key: val.squeeze(0) for key, val in inputs.items()}


# --- 3. Main Training Logic ---
def train():
    print("--- Starting LLaVA Fine-Tuning Script for YOLO data ---")

    # Load the processor and model
    print(f"Loading model and processor from {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )

    # Set up LoRA for PEFT
    print("--- Setting up LoRA for PEFT ---")
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print("LoRA adapters attached.")
    model.print_trainable_parameters()

    # Load the custom YOLO dataset
    print(f"Loading YOLO dataset from images in '{IMAGE_DIR}' and annotations in '{ANNOTATION_DIR}'...")
    train_dataset = YoloObjectDetectionDataset(
        image_dir=IMAGE_DIR,
        annotation_dir=ANNOTATION_DIR,
        class_names=CLASS_NAMES,
        processor=processor
    )
    
    if len(train_dataset) == 0:
        print("Dataset is empty. Please check your image and annotation directories.")
        return

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1.4e-5,
        save_total_limit=3,
        logging_steps=5,
        save_strategy="steps",
        save_steps=20,
        remove_unused_columns=False,
    )

    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Start training
    print("--- Starting Training ---")
    trainer.train()
    print("--- Training Finished ---")

    # Save the final model
    final_model_path = os.path.join(OUTPUT_DIR, "final")
    print(f"Saving final model to {final_model_path}")
    trainer.save_model(final_model_path)
    processor.save_pretrained(final_model_path)


if __name__ == "__main__":
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: The image directory '{IMAGE_DIR}' does not exist.")
    elif not os.path.exists(ANNOTATION_DIR):
        print(f"Error: The annotation directory '{ANNOTATION_DIR}' does not exist.")
    else:
        train()
