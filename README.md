# Proyecto de Visión por Computadora: Segment Anything & YOLO

Este proyecto es una introducción práctica a dos de las tareas más importantes en Visión por Computadora (CV): la **Detección de Objetos** y la **Segmentación de Imágenes**. 

Para ello, utilizamos dos de los modelos más avanzados de la actualidad:
1. **SAM (Segment Anything Model)** de Meta: Un modelo "fundacional" capaz de recortar (segmentar) cualquier objeto en una imagen con precisión de píxel.
2. **YOLO (You Only Look Once)**: Un modelo ultrarrápido especializado en encontrar objetos y encerrarlos en "cajas delimitadoras" (bounding boxes).

---

## Conceptos Previos: ¿Qué estamos haciendo?

En Visión por Computadora, queremos que la máquina "entienda" una imagen. Hay diferentes niveles de entendimiento:
- **Clasificación:** ¿Qué hay en la imagen? (Ej. "Un perro").
- **Detección de Objetos (YOLO):** ¿Dónde está y qué es? Devuelve una caja cuadrada alrededor del perro.
- **Segmentación (SAM):** ¿Qué píxeles exactos forman al perro? Devuelve la silueta exacta, ignorando el fondo.

Este proyecto te enseña a usar la segmentación pura y también a crear un **Pipeline** (una cadena de trabajo) uniendo la detección de YOLO con la segmentación de SAM.

---

## Instalación

1. Crea un entorno virtual para no interferir con otros proyectos de tu PC e instala las dependencias:
   ```bash
   python -m venv venv
   # En Windows: venv\Scripts\activate
   # En Mac/Linux: source venv/bin/activate
   pip install -r requirements.txt
   ```

El archivo `requirements.txt` instalará bibliotecas clave de Visión por Computadora:
- `torch` (PyTorch): El motor de inteligencia artificial donde corren los modelos.
- `opencv-python` (cv2): Herramienta clásica para leer y manipular matrices de imágenes.
- `matplotlib`: Para dibujar las máscaras y ver los resultados visualmente.
- `ultralytics` (YOLO) y `segment_anything` (SAM).

---

## Uso y Explicación del Código

El proyecto cuenta con dos scripts principales. Cada uno ilustra un concepto diferente.

### 1. `segment_image.py` (Segmentación Automática a ciegas)
Este script genera máscaras automáticas en toda la imagen. El modelo no sabe qué son los objetos, simplemente busca todo lo que parezca separarse del fondo.

**¿Cómo funciona internamente?**
1. **Lectura de imagen:** Usa OpenCV para leer la imagen y la convierte de formato BGR (el estándar por defecto de OpenCV) a RGB (el que esperan los modelos de IA).
2. **Generador Automático:** Utiliza la clase `SamAutomaticMaskGenerator`. Bajo el capó, SAM escanea la imagen usando una cuadrícula de "puntos invisibles". Para cada punto, adivina si ahí hay un objeto y genera su silueta.

**Ejecución:**
```bash
python segment_image.py --image path/a/tu/imagen.jpg
```
*(Nota: La primera vez que lo ejecutes, descargará automáticamente el cerebro del modelo `sam_vit_h_4b8939.pth` que pesa unos ~2.4GB).*

### 2. `yolo_sam_pipeline.py` (Pipeline Inteligente: Detección + Segmentación)
Aquí es donde ocurre la magia. SAM es increíble segmentando, pero por defecto no sabe "qué" está segmentando. Por otro lado, YOLO sabe "qué" y "dónde" están los objetos, pero solo te da una caja cuadrada, no la silueta. **¡Los vamos a combinar!**

**¿Cómo funciona el Pipeline?**
1. **Paso 1: Detección (YOLO):** Le pasamos la imagen a YOLOv8. Este busca objetos conocidos (personas, coches, perros) y nos devuelve las coordenadas de sus cajas delimitadoras (Bounding Boxes).
2. **Paso 2: Embeddings de la imagen (SAM):** Preparamos a SAM con `sam_predictor.set_image(image_rgb)`. Aquí SAM calcula un "mapa de características" gigante de la imagen. Se hace solo una vez y es el paso que más cuesta computacionalmente.
3. **Paso 3: Prompting Visual:** Convertimos las cajas de YOLO en formato Tensor (matemático) y se las pasamos a SAM como "prompts" (pistas). Le estamos diciendo a SAM: *"Mira dentro de esta caja y extráeme la silueta exacta del objeto que hay dentro"*.
4. **Resultado:** Obtenemos una segmentación perfecta guiada por inteligencia artificial.

**Ejecución:**
```bash
python yolo_sam_pipeline.py --image path/a/tu/imagen.jpg
```

---
### Próximos pasos y Extensiones

Si quieres seguir aprendiendo y expandiendo este repositorio, aquí tienes algunas ideas:
- **SAM 2 (Video):** Procesamiento temporal cuadro a cuadro con memoria de secuencia para segmentar objetos en movimiento en un MP4.
- **SAM + OCR:** En lugar de YOLO, usa un modelo OCR (Lector de texto). Encuentra dónde hay texto y usa SAM para recortar los letreros.
- **Webapp:** Exporta el modelo a formato ONNX (más ligero) y crea una interfaz web con React donde el usuario pueda hacer clic en una imagen para segmentar.
