import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPSegProcessor
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

from data_loader import DrywallCrackDataset
from model import get_model

DEVICE = "cpu"

def visualize(checkpoint_path, num_samples=5):
    print(f"Generating visualizations using {checkpoint_path}...")
    
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    val_dataset = DrywallCrackDataset(root_dir="data", split="val", processor=processor)
    # Shuffle to get random samples
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    
    model = get_model()
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    else:
        print("Checkpoint not found!")
        return

    model.to(DEVICE)
    model.eval()
    
    output_dir = "output/visuals"
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            if count >= num_samples:
                break
                
            pixel_values = batch['pixel_values'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            
            # Resize
            if logits.shape != labels.shape:
                logits = nn.functional.interpolate(logits.unsqueeze(1), size=labels.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
            
            probs = torch.sigmoid(logits)
            pred_mask = (probs > 0.5).float()
            
            # Convert to numpy for plotting
            # pixel_values is normalized, so we can't just show it directly easily without denormalizing.
            # But we can try to show the mask.
            
            # Let's try to get the original image from the dataset if possible, or just show the masks.
            # The dataset returns processed tensors. 
            # We can visualize the masks mainly.
            
            pred_np = pred_mask[0].cpu().numpy()
            gt_np = labels[0].cpu().numpy()
            original_np = batch['original_image'][0].numpy()
            
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title("Original Image")
            plt.imshow(original_np)
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.title("Ground Truth")
            plt.imshow(gt_np, cmap='gray')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.title("Prediction")
            plt.imshow(pred_np, cmap='gray')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"sample_{count}.png"))
            plt.close()
            
            count += 1
            
    print(f"Saved {count} visualization examples to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="output/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--num", type=int, default=5, help="Number of samples to visualize")
    args = parser.parse_args()
    
    visualize(args.checkpoint, args.num)
