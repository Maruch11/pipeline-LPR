import cv2

class LPRPipeline():
    """Orquesta el flujo"""

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
    """Localiza la placa y obtiene la ROI."""

    def process(self, image):
        """Ejecuta la detección completa de la placa."""
        contours = self._find_contours(image)
        candidates = self._filter_candidates(contours)
        candidate = self._select_candidate(candidates)
        roi = self._crop_roi(image, candidate)
        return roi

    def _find_contours(self, image):
        """Obtiene los contornos presentes en la imagen. 
        Devuelve lista de contornos.
        """
        pass

    def _filter_candidates(self, contours):
        """Descarta contornos que no pueden corresponder a una placa. 
        Devuelve lista de contornos válidos.
        """
        pass

    def _select_candidate(self, candidates):
        """Selecciona el mejor candidato entre los contornos restantes. 
        Devuelve un único contorno (numpy.ndarray) o None si no encuentra ninguno.
        """
        pass

    def _crop_roi(self, image, candidate):
        """Recorta la región de interés (ROI) correspondiente a la placa.
        Devuelve imagen recortada (numpy.ndarray) o None.
        """
        pass

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

