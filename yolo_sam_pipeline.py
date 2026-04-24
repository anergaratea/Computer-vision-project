import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import urllib.request
import os

from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO

def download_file(url, path):
    if not os.path.exists(path):
        print(f"Downloading from {url} to {path}...")
        urllib.request.urlretrieve(url, path)
        print("Download complete.")

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def main():
    parser = argparse.ArgumentParser(description="YOLO + SAM Pipeline")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    parser.add_argument("--yolo_model", default="yolov8n.pt", help="YOLOv8 model weights")
    args = parser.parse_args()

    # Model configuration for SAM
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    download_file(url, sam_checkpoint)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    print(f"Using device: {device}")

    print("Loading SAM model...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    print(f"Loading YOLO model {args.yolo_model}...")
    yolo_model = YOLO(args.yolo_model)

    # 1. Read Image
    print(f"Loading image {args.image}...")
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print("Error reading image!")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 2. Run object detection
    print("Running YOLO detection...")
    results = yolo_model(image_bgr)
    
    boxes = []
    if len(results) > 0:
        # Extract bounding boxes from YOLO results
        det_boxes = results[0].boxes
        if det_boxes is not None:
            boxes = det_boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
            print(f"Found {len(boxes)} bounding boxes.")

    # 3. Set image in SAM Predictor
    print("Setting image embeddings for SAM...")
    sam_predictor.set_image(image_rgb)

    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)

    # 4. Predict masks based on YOLO bounding boxes
    if len(boxes) > 0:
        input_boxes = torch.tensor(boxes, device=sam_predictor.device)
        # Transform boxes to SAM format
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_boxes, image_rgb.shape[:2])
        
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        for i, mask in enumerate(masks):
            show_mask(mask.cpu().numpy()[0], plt.gca(), random_color=True)
            show_box(boxes[i], plt.gca())
            
    plt.axis('off')
    output_filename = f"yolo_sam_output_{os.path.basename(args.image)}"
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    print(f"Saved result to {output_filename}")
    plt.show()

if __name__ == "__main__":
    main()
