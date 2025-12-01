import torch
import time
import os
from transformers import CLIPSegProcessor
from model import get_model

DEVICE = "cpu"

def benchmark():
    print(f"Benchmarking on {DEVICE}...")
    
    # 1. Model Size
    model = get_model()
    model.to(DEVICE)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"Model Size: {size_all_mb:.2f} MB")
    
    # 2. Inference Time
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    # Create dummy input
    dummy_image = torch.rand(1, 3, 352, 352).to(DEVICE) # Normalized-ish
    # Processor usually handles raw images, but model takes tensors. 
    # Let's simulate the model forward pass time (excluding preprocessing for now, or include it?)
    # The requirement says "avg inference time/image". Usually implies the whole pipeline or just model.
    # Let's measure model forward pass as it's the core.
    
    # Warmup
    print("Warming up...")
    input_ids = torch.tensor([[1, 2, 3]]).to(DEVICE) # Dummy IDs
    attention_mask = torch.tensor([[1, 1, 1]]).to(DEVICE)
    
    # We need valid input shapes. 
    # CLIPSeg: pixel_values [B, 3, 352, 352], input_ids [B, L], attention_mask [B, L]
    # Let's use the processor to get real shapes
    from PIL import Image
    dummy_pil = Image.new('RGB', (640, 640), color='red')
    inputs = processor(text=["segment crack"], images=[dummy_pil], return_tensors="pt")
    
    pixel_values = inputs['pixel_values'].to(DEVICE)
    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)
    
    with torch.no_grad():
        for _ in range(10):
            _ = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
    
    # Measure
    print("Measuring latency...")
    start_time = time.time()
    num_runs = 50
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    print(f"Average Inference Time (CPU): {avg_time*1000:.2f} ms/image")
    
    return size_all_mb, avg_time

if __name__ == "__main__":
    benchmark()
