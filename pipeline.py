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

        print("Detection: ", detection)

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
        print("Cantidad de Results:", len(results))

        if not results:
            print("Sin results")
            return None
        
        image_result = results[0]

        print("Cantidad de boxes: ", len(image_result.boxes))

        if len(image_result.boxes) == 0:
            print("Sin boxes")
            return None
        
        print("Box seleccionada")
        return image_result.boxes[0]

    def _crop_roi(self, image, detection):
        """Recorta la región de interés (ROI) correspondiente a la placa.
        Devuelve imagen recortada (numpy.ndarray) o None.
        """
        x1, y1, x2, y2 = detection.xyxy[0]

        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)

        roi = image[y1:y2, x1:x2]

        return roi
    
class ROINormalizer(PipelineStage):
    """Prepara la ROI para OCR."""
    ROI_WIDTH = 640
    ROI_HEIGHT = 480
    CLAHE_CLIP_LIMIT = 5

    def process(self, image):
        """Normaliza la ROI para mejorar el reconocimiento OCR."""
        image = self._resize(image)
        image = self._grayscale(image)
        image = self._enhance_contrast(image)
        return image
        
    def _resize(self, image):
        """Redimensiona la ROI a un tamaño fijo."""
        image = cv2.resize(image, (self.ROI_WIDTH, self.ROI_HEIGHT))
        return image

    def _grayscale(self, image):
        """Convierte la ROI a escala de grises."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _enhance_contrast(self, image):
        """Mejora el contraste mediante CLAHE."""
        clahe = cv2.createCLAHE(clipLimit=self.CLAHE_CLIP_LIMIT)
        image = clahe.apply(image)
        return image

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
            # config = self.config
            lang='eng'
        )
        return text

class PostProcessor(PipelineStage):
    """Corrige, valida y calcula el score final."""

    def process(self, image):
        pass

class Evaluator:
    """Calcula las métricas del sistema."""
    pass

