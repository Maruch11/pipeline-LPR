"""Punto de entrada de la aplicación. Coordina la ejecución."""
import cv2

from pipeline import Preprocessor

# Prueba `Preprocessor`

pre = Preprocessor()
image = cv2.imread("imgs/originales/001.png")
resultado = pre.process(image)

cv2.imshow("Original", image)
cv2.imshow("Preprocesada", resultado)
cv2.imwrite("imgs/preprocesadas/001.png", resultado)
cv2.waitKey(0) # Espera una tecla
cv2.destroyAllWindows() # Cierra las ventanas