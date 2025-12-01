import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPSegProcessor
from PIL import Image
import numpy as np
import os
import argparse
from tqdm import tqdm

from data_loader import DrywallCrackDataset
from model import get_model

DEVICE = "cpu"

def generate_masks(checkpoint_path, output_dir="output/masks"):
    print(f"Generating masks using {checkpoint_path}...")
    os.makedirs(output_dir, exist_ok=True)
    
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    # Use validation set as proxy for test set since we don't have a separate labeled test set loaded
    # Ideally we would use 'test' split if available.
    val_dataset = DrywallCrackDataset(root_dir="data", split="val", processor=processor)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    model = get_model()
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    else:
        print("Checkpoint not found!")
        return

    model.to(DEVICE)
    model.eval()
    
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Inference")):
            pixel_values = batch['pixel_values'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            # Get original image info from dataset directly if possible, or construct filename
            # The dataset loader doesn't return filenames easily in the batch dict unless we modify it.
            # But we can access the dataset via index since shuffle=False.
            sample_info = val_dataset.samples[i]
            image_path = sample_info['image_path']
            prompt = sample_info['prompt']
            
            # Construct filename: {image_id}__{prompt}.png
            # Image ID: basename without extension
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            # Sanitize prompt for filename
            safe_prompt = prompt.replace(" ", "_").replace("/", "-")
            filename = f"{image_id}__{safe_prompt}.png"
            
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            
            # Resize to original image size
            # We need the original size. We can open the image again or store it.
            # Let's open it quickly to get size.
            with Image.open(image_path) as img:
                w, h = img.size
                
            # Interpolate logits to original size
            logits = nn.functional.interpolate(logits.unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze()
            
            probs = torch.sigmoid(logits)
            pred_mask = (probs > 0.5).float().cpu().numpy()
            
            # Convert to {0, 255} uint8
            pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
            
            # Save
            save_path = os.path.join(output_dir, filename)
            Image.fromarray(pred_mask_uint8).save(save_path)
            count += 1
            
    print(f"Generated {count} masks in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="output/best_model.pth", help="Path to model checkpoint")
    args = parser.parse_args()
    
    generate_masks(args.checkpoint)
