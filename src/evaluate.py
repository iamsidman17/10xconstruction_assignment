import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPSegProcessor
from tqdm import tqdm
import numpy as np
import os
import argparse

from data_loader import DrywallCrackDataset
from model import get_model

DEVICE = "cpu" # Use CPU for evaluation to avoid conflict/memory issues with training
# DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def compute_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (intersection / union).item()

def compute_dice(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    return ((2. * intersection) / (pred.sum() + target.sum() + 1e-6)).item()

def evaluate(checkpoint_path=None):
    print(f"Evaluating on {DEVICE}...")
    
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    val_dataset = DrywallCrackDataset(root_dir="data", split="val", processor=processor)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    model = get_model()
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    else:
        print("No checkpoint found or provided. Evaluating BASE pre-trained model (Zero-Shot).")
    
    model.to(DEVICE)
    model.eval()
    
    total_iou = 0.0
    total_dice = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
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
            
            # Resize if needed
            if logits.shape != labels.shape:
                logits = nn.functional.interpolate(logits.unsqueeze(1), size=labels.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
            
            probs = torch.sigmoid(logits)
            
            # Calculate metrics for each image in batch
            for i in range(probs.shape[0]):
                iou = compute_iou(probs[i], labels[i])
                dice = compute_dice(probs[i], labels[i])
                
                total_iou += iou
                total_dice += dice
                num_samples += 1
    
    avg_iou = total_iou / num_samples
    avg_dice = total_dice / num_samples
    
    print(f"\nResults on Validation Set ({num_samples} images):")
    print(f"mIoU: {avg_iou:.4f}")
    print(f"Dice: {avg_dice:.4f}")
    
    return avg_iou, avg_dice

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="output/best_model.pth", help="Path to model checkpoint")
    args = parser.parse_args()
    
    evaluate(args.checkpoint)
