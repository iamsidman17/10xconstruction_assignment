import os
from roboflow import Roboflow

def download_datasets(api_key):
    rf = Roboflow(api_key=api_key)

    # Dataset 1: Drywall-Join-Detect
    print("Downloading Dataset 1: Drywall-Join-Detect...")
    project1 = rf.workspace("objectdetect-pu6rn").project("drywall-join-detect")
    dataset1 = project1.version(1).download("yolov8") # Downloading in YOLOv8 format (images + labels)
    # Note: We might need to adjust the format depending on what's available, but YOLOv8 usually gives images and txt labels.
    # For segmentation, we might prefer 'coco-segmentation' or 'yolov8-segmentation' if available.
    # The PDF says "Prediction masks: PNG", but the dataset might be object detection or segmentation polygons.
    # Let's check the dataset type. The URL says "objectdetect", so it might be bounding boxes.
    # But the goal is segmentation. If it's bbox only, we might have to treat bboxes as masks (rectangles).
    # However, the prompt says "segment taping area", implying segmentation.
    # Let's try to download as 'yolov8-segmentation' or 'coco-segmentation'.
    
    # Dataset 2: Cracks
    print("Downloading Dataset 2: Cracks...")
    project2 = rf.workspace("fyp-ny1jt").project("cracks-3ii36")
    dataset2 = project2.version(1).download("yolov8-segmentation")

if __name__ == "__main__":
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("Error: ROBOFLOW_API_KEY environment variable not set.")
        exit(1)
    
    download_datasets(api_key)
