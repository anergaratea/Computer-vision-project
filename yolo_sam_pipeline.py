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
    # Utilidad para descargar los pesos de los modelos si no se tienen localmente
    if not os.path.exists(path):
        print(f"Downloading from {url} to {path}...")
        urllib.request.urlretrieve(url, path)
        print("Download complete.")

def show_mask(mask, ax, random_color=False):
    # Función para superponer la silueta (máscara) segmentada en la imagen original
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    # Función para dibujar las "cajas delimitadoras" de YOLO (Bounding boxes)
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def main():
    parser = argparse.ArgumentParser(description="YOLO + SAM Pipeline")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    parser.add_argument("--yolo_model", default="yolov8n.pt", help="YOLOv8 model weights")
    args = parser.parse_args()

    # --- PREPARACIÓN DE MODELOS ---
    # 1. Configuración del modelo SAM (Segmentación)
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    download_file(url, sam_checkpoint)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    print(f"Using device: {device}")

    # Inicialización de SAM usando SamPredictor 
    # A diferencia de AutomaticMaskGenerator, SamPredictor está optimizado para recibir "prompts" (pistas) como puntos o cajas.
    print("Loading SAM model...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    # 2. Cargar el modelo YOLO (Detección de objetos inicial)
    # YOLOv8n es la versión 'nano' (muy rápida y ligera)
    print(f"Loading YOLO model {args.yolo_model}...")
    yolo_model = YOLO(args.yolo_model)

    # --- EJECUCIÓN DEL PIPELINE ---
    # Paso 1: Leer y preparar la imagen
    # OpenCV usa BGR. Lo pasamos a RGB para que los modelos interpreten bien los colores.
    print(f"Loading image {args.image}...")
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print("Error reading image!")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Paso 2: Detección de objetos con YOLO
    # YOLO escanea la imagen buscando patrones conocidos (personas, coches, etc.)
    # y devuelve las coordenadas de rectángulos que los encierran (bounding boxes).
    print("Running YOLO detection...")
    results = yolo_model(image_bgr)
    
    boxes = []
    if len(results) > 0:
        det_boxes = results[0].boxes
        if det_boxes is not None:
            # Extraer las coordenadas exactas de las cajas detectadas [x_min, y_min, x_max, y_max]
            boxes = det_boxes.xyxy.cpu().numpy() 
            print(f"Found {len(boxes)} bounding boxes.")

    # Paso 3: Calcular características profundas (Embeddings) de la imagen con SAM
    # Este es el paso más pesado computacionalmente. SAM convierte la imagen a un espacio matemático.
    # Se hace UNA SOLA VEZ por imagen. Después, los prompts (cajas/puntos) se procesan instantáneamente.
    print("Setting image embeddings for SAM...")
    sam_predictor.set_image(image_rgb)

    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)

    # Paso 4: Unir ambos modelos (El "Prompting" visual)
    # Pasamos las cajas de YOLO como pistas a SAM
    if len(boxes) > 0:
        input_boxes = torch.tensor(boxes, device=sam_predictor.device)
        # Adaptamos las coordenadas de las cajas al formato interno que espera SAM
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_boxes, image_rgb.shape[:2])
        
        # Generamos las máscaras
        # Le decimos a SAM: "Dentro de estas cajas de YOLO, encuéntrame la silueta exacta"
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False, # Queremos una sola máscara (la mejor) por cada caja
        )
        
        # Dibujar los resultados visuales superpuestos en la imagen original
        for i, mask in enumerate(masks):
            show_mask(mask.cpu().numpy()[0], plt.gca(), random_color=True) # Silueta de SAM
            show_box(boxes[i], plt.gca()) # Caja de YOLO original
            
    plt.axis('off')
    output_filename = f"yolo_sam_output_{os.path.basename(args.image)}"
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    print(f"Saved result to {output_filename}")
    plt.show()

if __name__ == "__main__":
    main()
