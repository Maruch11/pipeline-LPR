import cv2

from ultralytics import YOLO

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

class Preprocessor(PipelineStage):
    """Mejora la imagen"""

    def process(self, image):
        """Método publico orquestador"""
        image = self._grayscale(image)
        image = self._resize(image)
        image = self._enhance_contrast(image)
        return image
    
    def _grayscale(self, image):
        """Método interno, nomenclatura consistente con PEP 8, 
        convierte la imagen en escala de grises."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def _resize(self, image):    
        image = cv2.resize(image, (640, 480))
        return image

    def _enhance_contrast(self, image):
        clahe = cv2.createCLAHE(clipLimit=5)
        image = clahe.apply(image)
        return image

class PlateDetector(PipelineStage):
    """Localiza la placa y obtiene la región de interés (ROI)."""
    
    def __init__(self):
        self.model = YOLO("models/plate_detector.pt")

    def process(self, image):
        """Ejecuta la detección completa de la placa."""

        results = self._detect_plate(image)
        detection = self._select_detection(results)

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
        if not results:
            return None
        
        image_result = results[0]

        if len(image_result.boxes) == 0:
            return None
        
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

    def process(self, image):
        pass

class TesseractOCR(PipelineStage):
    """Reconoce el texto."""

    def process(self, image):
        pass

class PostProcessor(PipelineStage):
    """Corrige, valida y calcula el score final."""

    def process(self, image):
        pass

class Evaluator:
    """Calcula las métricas del sistema."""
    pass

