"""Punto de entrada de la aplicación. Coordina la ejecución."""
import cv2

from pipeline import LPRPipeline, PlateDetector, ROINormalizer, TesseractOCR, PostProcessor

# pipeline = LPRPipeline(
#     [
#         PlateDetector(),
#         ROINormalizer(),
#         TesseractOCR(),
#         PostProcessor(),
#     ]
# )

image = cv2.imread("imgs/originales/001.png")

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

cv2.imwrite("imgs/roi/001.png", roi)
cv2.imwrite("imgs/roi_normalizadas/001.png", normalized_roi)
cv2.imwrite("imgs/debug/001.png", normalized_roi)

if not cv2.imwrite("imgs/debug/001.png", normalized_roi):
    print("No se pudo guardar la imagen.")

print("ROI guardada en: imgs/roi/001.png")
print("ROI normalizada guardada en: imgs/roi_normalizadas/001.png")

if not cv2.imwrite("imgs/debug/001.png", normalized_roi):
    print("No se pudo guardar la imagen.")
else:
    print("Text Debug guardada en: imgs/debug/001.png")

cv2.waitKey(0)
cv2.destroyAllWindows()

