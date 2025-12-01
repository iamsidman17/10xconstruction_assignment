# Prompted Segmentation for Drywall QA

This project implements a text-conditioned segmentation model to identify defects in drywall installation, specifically **cracks** and **taping areas**.

## Goal
To train/fine-tune a model that takes an image and a natural language prompt (e.g., "segment crack") and produces a binary mask highlighting the target area.

## Approach
We utilize **CLIPSeg** (Image Segmentation Using Text and Image Prompts), a model capable of zero-shot and few-shot segmentation based on text queries. We fine-tuned this model on two specific datasets:
1.  **Drywall-Join-Detect**: For detecting taping areas/joints.
2.  **Cracks**: For detecting wall cracks.

## Setup

### 1. Installation
```bash
pip install -r requirements.txt
```
(Main dependencies: `torch`, `transformers`, `roboflow`, `pillow`, `matplotlib`)

### 2. Data Preparation
The datasets are downloaded from Roboflow using the `src/download_data.py` script (requires API key).
```bash
python3 src/download_data.py
```
The data is organized into `data/drywall` and `data/cracks`.

## Usage

### Training
To fine-tune the model:
```bash
python3 src/train.py
```
To resume training from a checkpoint:
```bash
python3 src/train.py --resume
```

### Evaluation
To calculate mIoU and Dice scores on the validation set:
```bash
python3 src/evaluate.py
```

### Visualization
To generate visual examples (Original | GT | Prediction):
```bash
python3 src/visualize.py
```

## Results
See [REPORT.md](REPORT.md) for detailed metrics and visual analysis.
