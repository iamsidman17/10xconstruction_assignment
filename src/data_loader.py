import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import numpy as np
from transformers import CLIPSegProcessor

class DrywallCrackDataset(Dataset):
    def __init__(self, root_dir, split='train', processor=None):
        """
        Args:
            root_dir (str): Path to the 'data' directory.
            split (str): 'train', 'valid' (or 'val'), or 'test'.
            processor (CLIPSegProcessor): Hugging Face processor.
        """
        self.root_dir = root_dir
        self.split = split
        self.processor = processor
        
        # Map 'val' to 'valid' if necessary, as Roboflow uses 'valid'
        if self.split == 'val':
            self.split_dir = 'valid'
        else:
            self.split_dir = self.split

        self.samples = []
        
        # Load Drywall samples (BBox format)
        drywall_path = os.path.join(root_dir, 'drywall', self.split_dir, 'images')
        if os.path.exists(drywall_path):
            drywall_images = glob.glob(os.path.join(drywall_path, '*'))
            for img_path in drywall_images:
                self.samples.append({
                    'image_path': img_path,
                    'type': 'drywall',
                    'prompt': 'segment taping area' # Or "segment joint/tape"
                })
        
        # Load Cracks samples (Polygon format)
        cracks_path = os.path.join(root_dir, 'cracks', self.split_dir, 'images')
        if os.path.exists(cracks_path):
            cracks_images = glob.glob(os.path.join(cracks_path, '*'))
            for img_path in cracks_images:
                self.samples.append({
                    'image_path': img_path,
                    'type': 'cracks',
                    'prompt': 'segment crack' # Or "segment wall crack"
                })

        print(f"Loaded {len(self.samples)} samples for split '{split}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample['image_path']
        dataset_type = sample['type']
        prompt = sample['prompt']

        # Load Image
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        # Load Label
        # Construct label path: replace 'images' with 'labels' and extension with .txt
        label_path = image_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0])
                coords = parts[1:]

                if dataset_type == 'drywall':
                    # YOLO BBox: x_center, y_center, width, height (normalized)
                    if len(coords) == 4:
                        xc, yc, bw, bh = coords
                        x1 = (xc - bw / 2) * w
                        y1 = (yc - bh / 2) * h
                        x2 = (xc + bw / 2) * w
                        y2 = (yc + bh / 2) * h
                        draw.rectangle([x1, y1, x2, y2], fill=1)
                
                elif dataset_type == 'cracks':
                    # YOLO Polygon: x1, y1, x2, y2, ... (normalized)
                    # Denormalize coordinates
                    poly_coords = []
                    for i in range(0, len(coords), 2):
                        px = coords[i] * w
                        py = coords[i+1] * h
                        poly_coords.append((px, py))
                    
                    if len(poly_coords) > 2:
                        draw.polygon(poly_coords, fill=1)

        # Process with CLIPSegProcessor
        # It handles resizing, normalizing, and text tokenization
        inputs = self.processor(
            text=[prompt],
            images=[image],
            padding="max_length",
            return_tensors="pt"
        )
        
        # Resize mask to match model output (usually 352x352 for CLIPSeg)
        # We can resize it to the input size the processor used.
        # CLIPSeg processor usually resizes images to 352x352.
        # Let's check the processor config or just resize to 352x352 manually for the mask.
        target_size = (352, 352) 
        mask = mask.resize(target_size, resample=Image.NEAREST)
        mask_tensor = torch.tensor(np.array(mask)).float() # 0.0 or 1.0

        # Inputs keys: pixel_values, input_ids, attention_mask
        # We need to squeeze the batch dimension added by processor
        # Resize original image for visualization
        original_image = image.resize((352, 352))
        original_image = np.array(original_image)

        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': mask_tensor,
            'original_image': original_image
        }

if __name__ == "__main__":
    # Test the dataset
    from transformers import CLIPSegProcessor
    
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    dataset = DrywallCrackDataset(root_dir="data", split="train", processor=processor)
    
    if len(dataset) > 0:
        sample = dataset[0]
        print("Sample keys:", sample.keys())
        print("Image shape:", sample['pixel_values'].shape)
        print("Label shape:", sample['labels'].shape)
        print("Label unique values:", torch.unique(sample['labels']))
