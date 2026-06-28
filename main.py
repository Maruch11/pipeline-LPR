"""Punto de entrada de la aplicación. Coordina la ejecución."""
import cv2

from pipeline import (
    LPRPipeline,
    PlateDetector,
    ROINormalizer,
    TesseractOCR,
    PostProcessor
)

# pipeline = LPRPipeline(
#     [
#         PlateDetector(),
#         ROINormalizer(),
#         TesseractOCR(),
#         PostProcessor(),
#     ]
# )

IMAGE_NAME = "001"

ROI_PATH = f"imgs/roi/{IMAGE_NAME}_padding10.png"
NORMALIZED_PATH = f"imgs/roi_normalizadas/{IMAGE_NAME}_padding10.png"
DEBUG_PATH = f"imgs/debug/{IMAGE_NAME}_refined_roi_padding10.png"

image = cv2.imread(f"imgs/originales/{IMAGE_NAME}.png")

detector = PlateDetector()
normalizer = ROINormalizer()
ocr = TesseractOCR()

print("Etapa 1: PlateDetector")

roi = detector.process(image)

if roi is None:
    print("No se detectó placa")
    exit()

print(f"ROI detectada: {roi.shape}")


print("Etapa 2: ROINormalizer")

normalized_roi = normalizer.process(roi)

print(f"ROI normalizada: {normalized_roi.shape}")


print("Etapa 3: TesseractOCR")

text = ocr.process(normalized_roi)

print(repr(text))

cv2.imshow("ROI", roi)
cv2.imshow("ROI Normalizada", normalized_roi)

cv2.imwrite(ROI_PATH, roi)
cv2.imwrite(NORMALIZED_PATH, normalized_roi)

if cv2.imwrite(DEBUG_PATH, normalized_roi):
    print(f"ROI guardada en: {ROI_PATH}")
    print(f"ROI normalizada guardada en: {NORMALIZED_PATH}")
    print(f"Debug guardada en: {DEBUG_PATH}")
else:
    print("No se pudo guardar la imagen de debug.")

cv2.waitKey(0)
cv2.destroyAllWindows()

