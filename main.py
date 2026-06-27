"""Punto de entrada de la aplicación. Coordina la ejecución."""
import cv2

from pipeline import Preprocessor, PlateDetector

# Prueba `Preprocessor`

pre = Preprocessor()
image = cv2.imread("imgs/originales/001.png")
resultado = pre.process(image)

cv2.imshow("Original", image)
cv2.imshow("Preprocesada", resultado)
cv2.imwrite("imgs/preprocesadas/001.png", resultado)
cv2.waitKey(0) # Espera una tecla
cv2.destroyAllWindows() # Cierra las ventanas

# Prueba `PlateDetector`

detector = PlateDetector()
roi = detector.process(resultado)

if roi is not None:
    print("ROI:", roi.shape)
    ok = cv2.imwrite("imgs/roi/001.png", roi)
    print("Guardada:", ok)

    cv2.imshow("ROI", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
else:
    print("No se detectó ninguna placa.")

print("ROI:", roi.shape)