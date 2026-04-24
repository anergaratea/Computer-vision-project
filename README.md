# Computer-vision-project
En CV, el proyecto Segment Anything (SAM) de Meta es un modelo fundamental de segmentación de objetos. Es un framework “promptable” que genera máscaras de alta calidad de cualquier objeto en imágenes (y video) usando puntos, cajas u otros prompts.

## Instalación

1. Crea un entorno virtual e instala las dependencias:
   ```bash
   python -m venv venv
   # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

El archivo `requirements.txt` instalará PyTorch, OpenCV, matplotlib, YOLO (ultralytics), y el propio repositorio de SAM desde GitHub.

## Uso

El proyecto cuenta con dos scripts principales:

### 1. `segment_image.py`
Genera máscaras automáticas en toda la imagen usando SAM.

**Ejecución:**
```bash
python segment_image.py --image path/a/tu/imagen.jpg
```
*Si es la primera vez, el modelo `sam_vit_h_4b8939.pth` (~2.4GB) se descargará automáticamente a la carpeta actual.*

### 2. `yolo_sam_pipeline.py`
Combina un modelo detector (YOLOv8) con SAM. YOLO detecta objetos y genera "cajas delimitadoras" (bounding boxes) que se envían como prompts a SAM para segmentar el objeto preciso.

**Ejecución:**
```bash
python yolo_sam_pipeline.py --image path/a/tu/imagen.jpg
```

---
### Extensiones

- **SAM 2 (Video):** Procesamiento temporal cuadro a cuadro con memoria de secuencia.
- **SAM + OCR:** Traductor visual para extraer texto de regiones (unión de un OCR potente).
- **Webapp:** Usando el modelo ONNX exportado de SAM integrado en React.
