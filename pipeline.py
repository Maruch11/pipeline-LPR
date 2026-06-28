import numpy as np
import cv2
import pytesseract

from ultralytics import YOLO

from dotenv import load_dotenv
import os

load_dotenv()


class LPRPipeline():
    """Coordina la ejecución secuencial de las etapas del pipeline de reconocimiento de placas, 
    pasando la salida de cada etapa como entrada de la siguiente.
    """

    def __init__(self, stages):
        self.stages = stages

    def process(self, input_data):
        for stage in self.stages:
            input_data = stage.process(input_data)
        return input_data

class PipelineStage:
    """Clase base para todas las etapas del pipeline.
    Define el contrato process()
    """

    def process(self, input_data):
        raise NotImplementedError(
            "Las clases derivadas deben implementar el metodo process()."
            ) 

class PlateDetector(PipelineStage):
    """Localiza la placa y obtiene la región de interés (ROI)."""
    PADDING_RATIO = 0.10

    def __init__(self):
        model_path = os.getenv("YOLO_MODEL")

        if not model_path:
            raise ValueError("La variable YOLO_MODEL no está definida.")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No se encontró el modelo YOLO: {model_path}"
            )
        self.model = YOLO(model_path)

    def process(self, image):
        """Ejecuta la detección completa de la placa."""

        results = self._detect_plate(image)
        detection = self._select_detection(results)

        print("Detection (archivo pipeline.py): ", detection)

        if detection is None:
            return None
        
        roi = self._crop_roi(image, detection)
        return roi

    def _detect_plate(self, image):
        """Detecta placas en la imagen.
        Devuelve los resultados de la inferencia de YOLO: lista de objetos Results.
        """
        results = self.model(image)
        return results

    def _select_detection(self, results):
        """Selecciona la mejor detección.
        Devuelve una única detección bounding box o None.
        """
        print("Cantidad de Results (archivo pipeline.py): ", len(results))

        if not results:
            print("Sin results (archivo pipeline.py)")
            return None
        
        image_result = results[0]

        print("Cantidad de boxes (archivo pipeline.py): ", len(image_result.boxes))

        if len(image_result.boxes) == 0:
            print("Sin boxes (archivo pipeline.py)")
            return None
        
        print("Box seleccionada (archivo pipeline.py)")
        return image_result.boxes[0]

    def _crop_roi(self, image, detection):
        """Recorta la región de interés (ROI) correspondiente a la placa, aplicando un margen proporcional al bounding box.
        Devuelve imagen recortada (numpy.ndarray) o None.
        """
        x1, y1, x2, y2 = detection.xyxy[0]

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        box_width = x2 - x1
        box_height = y2 - y1

        # calcular padding
        padding_x = int(box_width * self.PADDING_RATIO)
        padding_y = int(box_height * self.PADDING_RATIO)

        # modificar x1, y1, x2, y2
        x1 -= padding_x
        x2 += padding_x
        y1 -= padding_y
        y2 += padding_y

        height, width = image.shape[:2]

        # limitar a los bordes de la imagen
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        roi = image[y1:y2, x1:x2]

        print(
            f"ROI con padding: ({x2 - x1}x{y2 - y1})"
            f"(+{padding_x}px, +{padding_y}px)"
        )
        return roi
    
class ROINormalizer(PipelineStage):
    """Prepara la ROI para OCR."""
    TARGET_WIDTH = 640
    CLAHE_CLIP_LIMIT = 5
    MIN_WIDTH = 80
    MAX_WIDTH = 110
    MIN_HEIGHT = 25
    MAX_HEIGHT = 52
    ASPECT_RATIO = 3.07692307692
    ASPECT_RATIO_TOLERANCE = 0.7

    def process(self, image):
        """Normaliza la ROI para mejorar el reconocimiento OCR."""
        image = self._resize(image)
        image = self._grayscale(image)

        threshold = self._threshold(image)
        contours = self._find_contours(threshold)
        candidates = self._filter_candidates(contours)

        print(f"Candidatos encontrados (archivo pipeline.py): {len(candidates)}")

        candidate = self._select_candidate(candidates)

        image = self._crop_refined_roi(image, candidate)

        image = self._enhance_contrast(image)
        image = self._binarize(image)

        return image
            
    def _resize(self, image):
        """Redimensiona la ROI manteniendo la relación de aspecto, escalando la imagen hasta un ancho objetivo."""
        height, width = image.shape[:2]
        scale = self.TARGET_WIDTH / width
        new_height = int(height * scale)
        image = cv2.resize(
                        image,
                        (self.TARGET_WIDTH, new_height),
                        interpolation=cv2.INTER_CUBIC,
                    )
        
        print(f"Resize (archvio pipeline.py): {width}x{height} -> {self.TARGET_WIDTH}x{new_height}")
        print(f"ROI original (archvio pipeline.py): ({height}, {width})")
        print(f"ROI redimensionada (archvio pipeline.py): {image.shape[:2]}")
        return image

    def _grayscale(self, image):
        """Convierte la ROI a escala de grises."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _enhance_contrast(self, image):
        """Mejora el contraste mediante CLAHE."""
        clahe = cv2.createCLAHE(clipLimit=self.CLAHE_CLIP_LIMIT)
        image = clahe.apply(image)
        return image
    
    def _binarize(self, image):
        """Binariza la ROI mediante el umbral automático de Otsu."""
        _, image = cv2.threshold(
            image,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return image
    
    def _threshold(self, image):
        """Aplica un umbral fijo para la búsqueda de contornos."""
        _, image = cv2.threshold(
            image,
            170,
            255,
            cv2.THRESH_BINARY_INV,
        )
        return image
    
    def _find_contours(self, image):
        """Obtiene los contornos de la imagen binarizada."""
        contours, _ = cv2.findContours(
            image,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        return contours

    def _filter_candidates(self, contours):
        """Filtra los contornos compatibles con una patente."""
        candidates = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            aspect_ratio = w / h

            print("archivo pipeline.py antes del for de _filter_candidates:")
            print(
                f"w={w:3d} h={h:3d} ratio={aspect_ratio:.2f}"
            )

            if (
                np.isclose(
                    aspect_ratio,
                    self.ASPECT_RATIO,
                    atol=self.ASPECT_RATIO_TOLERANCE,
                )
                and self.MIN_WIDTH < w < self.MAX_WIDTH
                and self.MIN_HEIGHT < h < self.MAX_HEIGHT
            ):
                candidates.append(contour)

        return candidates

    def _select_candidate(self, candidates):
        """Selecciona el candidato más probable."""

        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        return max(
            candidates,
            key=lambda contour: cv2.boundingRect(contour)[1],
        )

    def _crop_refined_roi(self, image, contour):
        """Recorta la ROI refinada a partir del contorno seleccionado."""

        if contour is None:
            return image

        x, y, w, h = cv2.boundingRect(contour)

        return image[y:y + h, x:x + w]

class TesseractOCR(PipelineStage):
    """Reconoce el texto de la patente mediante Tesseract."""

    def __init__(self):
        tesseract_cmd = os.getenv("TESSERACT_CMD")

        if not tesseract_cmd:
            raise ValueError(
                "La variable de entorno TESSERACT_CMD no está definida."
            )

        if not os.path.exists(tesseract_cmd):
            raise FileNotFoundError(
                f"No se encontró el ejecutable de Tesseract: {tesseract_cmd}"
            )

        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
               
        self.config =(
            "--oem 3 "
            "--psm 7 "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

    def process(self, image):
        text = pytesseract.image_to_string(
            image,
            config = self.config,
            lang='eng'
        )

        print("(archivo pipeline.py)")
        print(repr(text))
        return text

class PostProcessor(PipelineStage):
    """Corrige, valida y calcula el score final."""

    def process(self, image):
        pass

class Evaluator:
    """Calcula las métricas del sistema."""
    pass