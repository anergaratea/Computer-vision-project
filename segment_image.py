import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import urllib.request
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def download_checkpoint(url, path):
    if not os.path.exists(path):
        print(f"Downloading checkpoint from {url} to {path}...")
        urllib.request.urlretrieve(url, path)
        print("Download complete.")
    else:
        print(f"Checkpoint already exists at {path}.")

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def main():
    parser = argparse.ArgumentParser(description="Segment Anything Automatic Mask Generation")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    args = parser.parse_args()

    # Model configuration
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

    download_checkpoint(url, sam_checkpoint)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    print(f"Using device: {device}")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    # Read image
    print(f"Loading image {args.image}...")
    image = cv2.imread(args.image)
    if image is None:
        print("Error: Could not read image.")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Generating masks...")
    masks = mask_generator.generate(image)

    # Visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    
    output_path = f"segmented_{os.path.basename(args.image)}"
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    print(f"Saved segmented image to {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
