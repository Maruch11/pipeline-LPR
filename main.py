"""Punto de entrada de la aplicación. Coordina la ejecución."""
import cv2
import pandas as pd

from pathlib import Path

from pipeline import (
    PlateDetector,
    ROINormalizer,
    EasyOCR,
)

GROUND_TRUTH_FILE = Path("data/ground_truth.csv")
DATASET_PATH = Path("imgs/originales")

detector = PlateDetector()
normalizer = ROINormalizer()
ocr = EasyOCR()

ground_truth = pd.read_csv(GROUND_TRUTH_FILE)

labels = {
    row["archivo"]: row["patente"].upper()
    for _, row in ground_truth.iterrows()
}

results = []

for image_path in sorted(DATASET_PATH.glob("*.png")):

    image_name = image_path.stem
    
    roi_path = f"imgs/roi/{image_name}_easyocr.png"
    normalized_path = f"imgs/roi_normalizadas/{image_name}_easyocr.png"
    debug_path = f"imgs/debug/{image_name}_easyocr.png"

    print("\n" + "=" * 60)
    print(f"Imagen: {image_path.stem}")
    print("=" * 60)

    image = cv2.imread(str(image_path))

    print("\nEtapa 1: PlateDetector")
    roi = detector.process(image)

    if roi is None:
        print("No se detectó placa.")

        results.append(
            {
                "image": image_path.name,
                "ground_truth": labels.get(image_path.name, ""),
                "prediction": "",
                "correct": False,
            }
        )
        continue
    
    print("\nEtapa 2: ROINormalizer")
    normalized_roi = normalizer.process(roi)
    
    print(f"\nEtapa 3: {ocr.__class__.__name__}")
    text = ocr.process(normalized_roi)

    print(f"{ocr.__class__.__name__}: {repr(text)}")

    expected = labels.get(image_path.name, "")

    prediction = text.upper()

    is_correct = prediction == expected

    results.append(
        {
            "image": image_path.name,
            "ground_truth": expected,
            "prediction": prediction,
            "correct": is_correct,
        }
    )

    cv2.imwrite(roi_path, roi)
    cv2.imwrite(normalized_path, normalized_roi)
    cv2.imwrite(debug_path, normalized_roi)

print("\nResumen")
print("-" * 70)

correct_predictions = 0

for result in results:

    status = "OK" if result["correct"] else "FAIL"

    if result["correct"]:
        correct_predictions += 1

    print(
        f"{result['image']:10}"
        f"GT={result['ground_truth']:8}"
        f"PRED={result['prediction']:12}"
        f"{status}"
    )

accuracy = (
    correct_predictions / len(results) * 100
    if results
    else 0
)

print("-" * 70)
print(f"Imágenes procesadas : {len(results)}")
print(f"Aciertos exactos    : {correct_predictions}")
print(f"Accuracy            : {accuracy:.2f}%")

results_df = pd.DataFrame(results)

OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "predictions.csv"

OUTPUT_DIR.mkdir(exist_ok=True)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(f"Resultados guardados en: {OUTPUT_FILE}")