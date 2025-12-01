import torch
import torch.nn as nn
from transformers import CLIPSegForImageSegmentation

def get_model(model_name="CIDAS/clipseg-rd64-refined"):
    """
    Loads the CLIPSeg model for fine-tuning.
    """
    model = CLIPSegForImageSegmentation.from_pretrained(model_name)
    return model

if __name__ == "__main__":
    model = get_model()
    print("Model loaded successfully.")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
