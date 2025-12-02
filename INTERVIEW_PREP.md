# Interview Preparation: Drywall QA Segmentation Project

## 1. Project Overview
**Goal**: Build a computer vision model to identify two specific defects in drywall installation:
1.  **Cracks** (Structural defects).
2.  **Taping Areas** (Joints between drywall sheets).

**Core Requirement**: The model must be **text-conditioned**. It should take an image AND a text prompt (e.g., "segment crack") and output a binary mask for that specific feature.

---

## 2. Architecture & Model Choice
### Decision: Why CLIPSeg?
We chose **CLIPSeg** (`CIDAS/clipseg-rd64-refined`) over standard U-Nets or DeepLab.
*   **Reason 1 (Text-Conditioning)**: Standard segmentation models (like U-Net) are class-specific. You train head A for cracks, head B for tape. CLIPSeg uses a **frozen CLIP encoder** to understand *any* text prompt, making it naturally suited for this "promptable" requirement.
*   **Reason 2 (Data Efficiency)**: Because CLIP already "knows" what a crack or a wall looks like from its massive pre-training, we only needed to fine-tune the lightweight **decoder** to adapt to our specific mask format. This allowed us to get good results with just ~1000-6000 images in a few hours.
*   **Reason 3 (Lightweight)**: The model size is ~575MB, which is reasonable for deployment compared to massive multimodal models.

---

## 3. Data Strategy
### The Datasets
We used two distinct datasets from Roboflow:
1.  **Drywall-Join-Detect**: Small (~1000 images), labeled with **Bounding Boxes**.
2.  **Cracks**: Large (~6600 images), labeled with **Polygons**.

### Key Decision: Unifying the Format
*   **Challenge**: The model expects a pixel-wise binary mask (0s and 1s), but we had BBoxes and Polygons.
*   **Solution**: We wrote a custom `DrywallCrackDataset` class (`src/data_loader.py`).
    *   For **BBoxes**: We drew a filled white rectangle on a black canvas.
    *   For **Polygons**: We drew filled polygons using the coordinates.
    *   **Why?**: This standardized the input for the model, allowing us to train on both datasets simultaneously without changing the architecture.

### Key Decision: Augmentation Strategy
*   **Observation**: The "Drywall" dataset was much smaller than "Cracks".
*   **Action**: We applied **3x Augmentation** (Flip, Rotation, Brightness) *only* to the Drywall dataset during download.
*   **Why?**: To balance the class distribution. If we didn't, the model would have been biased towards detecting cracks and might have ignored the taping areas.

---

## 4. Training Process
### Hyperparameters
*   **Optimizer**: `AdamW` (Standard for Transformers, handles weight decay better than Adam).
*   **Learning Rate**: `1e-4` (Typical fine-tuning rate; not too high to destroy pre-trained weights, not too low to be slow).
*   **Loss Function**: `BCEWithLogitsLoss`.
    *   **Why?**: This is a binary classification problem at the pixel level (Is this pixel a crack? Yes/No). BCE is the mathematically correct loss for this.

### The "MPS" Challenge (Mac GPU)
*   **Issue**: We initially tried training on Apple's MPS (Metal Performance Shaders) for speed. However, we encountered persistent `RuntimeError: view size is not compatible with input tensor's size and stride`.
*   **Attempted Fixes**: We tried adding `.contiguous()` to tensors and enabling `PYTORCH_ENABLE_MPS_FALLBACK`.
*   **Final Decision**: We switched to **CPU**.
    *   **Why?**: Stability > Speed. It was better to have a reliable 2-hour training run on CPU than a crashing GPU run. In an interview, this shows **pragmatism**—you chose the tool that got the job done.

---

## 5. Results & Analysis
### Final Metrics (Epoch 10)
*   **Overall mIoU**: 0.59
*   **Overall Dice**: 0.74

### Breakdown
*   **Taping Areas**: mIoU **0.63**, Dice **0.77**.
*   **Cracks**: mIoU **0.45**, Dice **0.62**.

### Why is Taping better than Cracks?
You might be asked this. The answer is in the **Labels**:
*   **Taping labels were Bounding Boxes**: These are coarse rectangles. It's "easy" for the model to get high overlap with a big rectangle.
*   **Crack labels were Polygons**: These are thin, precise lines. If the model predicts a crack 1 pixel to the left, the IoU drops to 0. It is **much harder** to get high IoU on thin objects.

---

## 6. What would you improve? (Future Work)
If they ask "What if you had more time?", say:
1.  **Weighted Loss**: Since cracks are thin (few pixels), the "background" (wall) dominates the loss. I would implement **Focal Loss** or weight the positive pixels higher to force the model to focus on the cracks.
2.  **Test Set**: We used the Validation set for final metrics. Ideally, we should have a completely held-out Test set.
3.  **Post-Processing**: The model sometimes outputs "speckles" (noise). I would add a **Morphological Opening** operation (erosion followed by dilation) to clean up the masks.

---

## 7. Summary for the Interview
"I built a text-conditioned segmentation pipeline using CLIPSeg. I unified two disparate datasets (BBox and Polygon) into binary masks, balanced them via augmentation, and fine-tuned the model using PyTorch. I overcame hardware compatibility issues on Mac by prioritizing stability on CPU, achieving a final Dice score of 0.74, which is strong for a thin-object segmentation task."
