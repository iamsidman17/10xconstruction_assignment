import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPSegProcessor
from tqdm import tqdm
import numpy as np

from data_loader import DrywallCrackDataset
from model import get_model

# Hyperparameters
BATCH_SIZE = 16 # Adjust based on GPU memory (MPS usually handles 16-32 fine for this size)
LEARNING_RATE = 1e-4 # Slightly lower for fine-tuning
EPOCHS = 10
DEVICE = "cpu"
# DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# if torch.cuda.is_available():
#     DEVICE = "cuda"

import argparse

def train(resume=False):
    print(f"Using device: {DEVICE}")
    
    # Initialize Processor and Dataset
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    
    train_dataset = DrywallCrackDataset(root_dir="data", split="train", processor=processor)
    val_dataset = DrywallCrackDataset(root_dir="data", split="val", processor=processor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize Model
    model = get_model()
    
    # Resume from checkpoint if requested
    if resume and os.path.exists("output/best_model.pth"):
        print("Resuming from checkpoint: output/best_model.pth")
        model.load_state_dict(torch.load("output/best_model.pth", map_location=DEVICE))
    
    model.to(DEVICE)
    
    # Optimizer and Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_loss = float('inf')
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Training Loop
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            pixel_values = batch['pixel_values'].to(DEVICE).contiguous()
            input_ids = batch['input_ids'].to(DEVICE).contiguous()
            attention_mask = batch['attention_mask'].to(DEVICE).contiguous()
            labels = batch['labels'].to(DEVICE).contiguous()
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels # CLIPSeg computes loss internally if labels provided, but let's use ours for control
            )
            
            # CLIPSeg output logits are [batch_size, 352, 352] (or similar)
            logits = outputs.logits
            
            # Resize logits if necessary (though processor/model usually align)
            # If logits shape != labels shape, interpolate
            if logits.shape != labels.shape:
                logits = nn.functional.interpolate(logits.unsqueeze(1), size=labels.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
            
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
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
                if logits.shape != labels.shape:
                    logits = nn.functional.interpolate(logits.unsqueeze(1), size=labels.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print("Saved best model!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from output/best_model.pth")
    args = parser.parse_args()
    
    train(resume=args.resume)
