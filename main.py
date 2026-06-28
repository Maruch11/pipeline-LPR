"""Punto de entrada de la aplicación. Coordina la ejecución."""
import cv2

from pathlib import Path

from pipeline import (
    # LPRPipeline,
    PlateDetector,
    ROINormalizer,
    # TesseractOCR,
    EasyOCR,
    # PostProcessor
)

# pipeline = LPRPipeline(
#     [
#         PlateDetector(),
#         ROINormalizer(),
#         TesseractOCR(),
#         PostProcessor(),
#     ]
# )


DATASET_PATH = Path("imgs/originales")

# IMAGE_NAME = "001"
# EXPERIMENT = "easyocr"

# ROI_PATH = f"imgs/roi/{IMAGE_NAME}_{EXPERIMENT}.png"
# NORMALIZED_PATH = f"imgs/roi_normalizadas/{IMAGE_NAME}_{EXPERIMENT}.png"
# DEBUG_PATH = f"imgs/debug/{IMAGE_NAME}_{EXPERIMENT}.png"

# image = cv2.imread(f"imgs/originales/{IMAGE_NAME}.png")

# detector = PlateDetector()
# normalizer = ROINormalizer()

# # ocr = TesseractOCR()
# ocr = EasyOCR()

# # postprocessor = PostProcessor()

# print("Etapa 1: PlateDetector")

# roi = detector.process(image)

# if roi is None:
#     print("No se detectó placa")
#     raise SystemExit(1)

# # print(f"ROI detectada: {roi.shape}")


# print("Etapa 2: ROINormalizer")

# normalized_roi = normalizer.process(roi)

# print(f"ROI normalizada: {normalized_roi.shape}")


# print(f"Etapa 3: {ocr.__class__.__name__}")

# text = ocr.process(normalized_roi)

# print(f"Texto reconocido: {repr(text)}")

# print("Guardando imagenes...")

# roi_saved = cv2.imwrite(ROI_PATH, roi)
# normalized_saved = cv2.imwrite(NORMALIZED_PATH, normalized_roi)
# debug_saved = cv2.imwrite(DEBUG_PATH, normalized_roi)

# print(f"ROI guardada: {roi_saved} -> {ROI_PATH}")
# print(f"ROI normalizada guardada: {normalized_saved} -> {NORMALIZED_PATH}")
# print(f"Debug guardada: {debug_saved} -> {DEBUG_PATH}")


# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

detector = PlateDetector()
normalizer = ROINormalizer()
ocr = EasyOCR()

results = []

for image_path in sorted(DATASET_PATH.glob("*.png"))[:10]:

    image_name = image_path.stem
    
    roi_path = f"imgs/roi/{image_name}_easyocr.png"
    normalized_path = f"imgs/roi_normalizadas/{image_name}_easyocr.png"
    debug_path = f"imgs/debug/{image_name}_easyocr.png"

    print("\n" + "=" * 60)
    print(f"Imagen: {image_path.stem}")
    print("=" * 60)

    image = cv2.imread(str(image_path))

    roi = detector.process(image)

    if roi is None:
        print("No se detectó placa.")
        continue

    normalized_roi = normalizer.process(roi)

    text = ocr.process(normalized_roi)

    print(f"{ocr.__class__.__name__}: {repr(text)}")

    results.append((image_name, text))

    cv2.imwrite(roi_path, roi)
    cv2.imwrite(normalized_path, normalized_roi)
    cv2.imwrite(debug_path, normalized_roi)

print("\nResumen")
print("-" * 40)

for image_name, text in results:
    print(f"{image_name:>3} -> {text}")