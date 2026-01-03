# THIS IS A PLACEHOLDER SCRIPT.
# It demonstrates how you would fine-tune a LLaVA model on your custom COCO-formatted dataset.
# You will need a machine with a powerful GPU and sufficient memory to run this.

import os
import json
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset

# --- 1. Configuration ---
MODEL_ID = "llava-hf/llava-1.5-7b-hf"  # Example model
COCO_ANNOTATIONS_PATH = "../data/result.json"
# The COCO JSON has relative paths, so we need a base path to the images.
# This will depend on where you store the images.
# For this example, we assume they are in a folder 'images/' at the root of the project.
IMAGE_DIR = "../images" # IMPORTANT: You need to create this directory and place your images inside it.
OUTPUT_DIR = "./llava-finetuned-model"

# --- 2. Custom Dataset for COCO-formatted data ---
class CocoObjectDetectionDataset(Dataset):
    """
    A custom PyTorch dataset to handle COCO-formatted object detection data
    for use with a VLM like LLaVA.
    """
    def __init__(self, annotations_file, image_directory, processor):
        self.image_directory = image_directory
        self.processor = processor

        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)

        self.annotations = coco_data['annotations']
        
        # Create mappings for quick lookups
        self.images = {img['id']: img for img in coco_data['images']}
        self.categories = {cat['id']: cat for cat in coco_data['categories']}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        
        # Get image info
        image_info = self.images[image_id]
        # IMPORTANT: The file_name in your COCO file has backwards slashes and relative paths.
        # We need to normalize it and combine it with our IMAGE_DIR.
        # This part might need adjustment based on your exact file paths.
        filename = os.path.basename(image_info['file_name'].replace('\\', '/'))
        image_path = os.path.join(self.image_directory, filename)
        
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image not found at {image_path}. Skipping.")
            # Return the next valid item
            return self.__getitem__((idx + 1) % len(self))

        # Get category name
        category = self.categories[category_id]['name']

        # Correctly format the conversation for the model.
        # The model needs to learn to predict the bounding box after the "ASSISTANT:" token.
        bbox_str = str(annotation['bbox'])
        
        # The conversation template is crucial for LLaVA.
        # It must match the format the model was originally trained on.
        prompt = f"USER: <image>\nPlease provide the bounding box for the {category} in the image."
        conversation = f"{prompt} ASSISTANT: {bbox_str}</s>"

        # Process the full conversation and image
        inputs = self.processor(text=conversation, images=image, return_tensors="pt")
        
        # Create labels, which are a copy of input_ids
        labels = inputs.input_ids.clone()

        # Find where the assistant's response begins to mask the prompt.
        # We tokenize the prompt part separately to find its length.
        prompt_only_inputs = self.processor(text=f"{prompt} ASSISTANT:")
        prompt_len = len(prompt_only_inputs.input_ids[0])

        # Mask the prompt part in the labels by setting it to -100.
        # The loss function will ignore these tokens.
        labels[0, :prompt_len] = -100

        # The trainer expects a dictionary of tensors. Squeeze the batch dimension.
        inputs['labels'] = labels
        
        return {key: val.squeeze(0) for key, val in inputs.items()}


# --- 3. Main Training Logic ---
def train():
    print("--- Starting LLaVA Fine-Tuning Script ---")

    # Load the processor and model
    print(f"Loading model and processor from {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    # For fine-tuning on a custom task, it's good practice to load the model with quantization
    # to save memory, especially on consumer-grade GPUs.
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16,
        load_in_4bit=True, # Use 4-bit quantization
    )

    # Load the custom dataset
    print(f"Loading dataset from {COCO_ANNOTATIONS_PATH}...")
    # NOTE: You might want to split your data into train and validation sets.
    # For simplicity, this script uses the same file for both.
    train_dataset = CocoObjectDetectionDataset(
        annotations_file=COCO_ANNOTATIONS_PATH,
        image_directory=IMAGE_DIR,
        processor=processor
    )
    
    if len(train_dataset) == 0:
        print("Dataset is empty. Please check your annotation paths and image directory.")
        return

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1, # Start with 1 epoch and increase as needed
        per_device_train_batch_size=2, # Adjust based on your GPU memory
        gradient_accumulation_steps=8,
        learning_rate=1.4e-5,
        save_total_limit=3,
        logging_steps=5,
        save_strategy="steps",
        save_steps=20,
        remove_unused_columns=False, # Important for custom datasets
    )

    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # You would also pass a 'eval_dataset' here for evaluation
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
    # This check ensures that the training code only runs when the script is executed directly.
    # IMPORTANT: Before running, ensure you have created an 'images' directory and placed your
    # image files from the dataset into it. The script will fail if it cannot find the images.
    
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: The image directory '{IMAGE_DIR}' does not exist.")
        print("Please create it and copy your dataset images into it before running this script.")
    elif not os.path.exists(COCO_ANNOTATIONS_PATH):
        print(f"Error: The annotations file '{COCO_ANNOTATIONS_PATH}' does not exist.")
    else:
        train()
