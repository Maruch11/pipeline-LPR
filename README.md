# Proyecto pipeline LPR (License Plate Recognition)
Estado: PROYECTO EN CONSTRUCCION


El pipeline tiene etapas claramente separadas:

- Detección
- Normalización de la ROI
- OCR
- Postprocesamiento
- Evaluación

Cada una recibe una entrada y produce una salida.

```
LPRPipeline
|
├── PlateDetector
├── ROINormalizer
├── TesseractOCR
├── PostProcessor
└── Evaluator
```

## Flujo
```
Imagen original
   ↓
    
PlateDetector (YOLO)
   ↓
 
Recorte (ROI)
   ↓
   
Normalización
   ↓
   
OCR (Tesseract)
   ↓
   
Post-procesado
   ↓
   
Scoring
   ↓
   
Evaluación (Precision / Recall)
```

---

## Módulos
```
lpr-pipeline/
|
├── data/
|   └── ground_truth.csv
|
├── images/
|
├── src/
      └── pipeline.py
|     └── main.py
|
├── tests/
|
├── README.md
|
└── requirements.txt
```
---

## Modelacion del dominio

Implica diseñar el modelo antes que el algoritmo.

Preguntas:

- ¿Qué es una etapa del pipeline?
- ¿Qué información circula entre etapas?
- ¿Qué representa una detección?
- ¿Qué representa un resultado OCR?
- ¿Qué representa una evaluación?
- cómo representar los datos que pasan entre etapas?
- ¿Qué devuelve exactamente cada process()?
- ¿Cómo integrar el código del notebook povisto por la cátedra dentro de cada clase?
- ¿Conviene incorporar dataclass?
- ¿Cómo modelar el resultado del OCR o de la detección?

---

## Diseño POO

Competencias
- encapsulamiento;
- herencia;
- polimorfismo;
- separación de responsabilidades;
- modularidad;
- facilidad para reemplazar implementaciones.

El requerimiento menciona que el detector puede ser HOG+SVM o YOLO, y que EasyOCR puede actuar como mecanismo de respaldo para Tesseract.

Con una arquitectura orientada a objetos, cambiar una implementación deja de implicar modificar todo el pipeline.

Cada etapa encapsulada en una clase, con responsabilidades definidas.

Diseño alto nivel:

```
LPRPipeline
|
├── PipelineStage (base)
|   |
|   ├── PlateDetector
|   ├── ROINormalizer
|   ├── TesseractOCR
|   └── PostProcessor
|
└── Evaluator
```
Responsabilidades:

- LPRPipeline: orquesta el flujo.
- PipelineStage: define el contrato process().
- PlateDetector: localiza la placa y obtiene la ROI.
- ROINormalizer: prepara la ROI para OCR.
- TesseractOCR: reconoce el texto.
- PostProcessor: corrige, valida y calcula el score final.
- Evaluator: calcula las métricas del sistema.

### Clases

De lo general a lo particular:

- Clase principal.
- Clase base.
- Clses hijas.
- Clase de evaluación.

### Clase LPRPipeline

- recibe la imagen original;
- crea o utiliza las instancias de las demás clases;
- llama a sus métodos en el orden correcto;
- pasa la salida de una etapa como entrada de la siguiente;
- devuelve el resultado final.

### Clase base
```
- `PipelineStage`
        └──process()
```

### CLases hijas

- `PlateDetector`
- `ROINormalizer`
- `TesseractOCR`
- `PostProcessor`

Comportamiento externo

```
entrada
   ↓
process()
   ↓
salida
```

### Clase Evaluator

```
texto_detectado
+
ground_truth
↓
calcular métricas
↓
Accuracy, TP, FP, FN, Precision, Recall, F1-score
```

---

## Módulos

### `pipeline.py`

Se aplica polimorfismo.

#### `LPRPipeline` 

El `LPRPipeline` podrá ejecutar todas las etapas sin conocer cuál es cada una:

```
for stage in stages:
    salida = stage.process(entrada)
    entrada = salida
```

Su responsabilidad es orquestar las etapas. 

No hace procesamiento de imágenes.

Solo mantiene el orden:

```
image
   ↓
PlateDetector
   ↓
ROINormalizer
   ↓
TesseractOCR
   ↓
PostProcessor
   ↓
Result
```

Internamente tiene un atributo:

`self.etapas`

que almacena la secuencia de objetos del pipeline. Esto hace que el diseño sea desacoplado y extensible.

Es una colección de objetos que representan las etapas del pipeline.

Su responsabilidad será recorrer esa colección ejecutando process() sobre cada objeto.

Depende por uso de PipelineStage, no por herencia ya que `LPRPipeline` necesita trabajar con objetos que sean etapas del pipeline.

Conceptualmente hace:

```
LPRPipeline
    |
    ├── PlateDetector
    ├── ROINormalizer
    ├── TesseractOCR
    └── PostProcessor
```

Cada uno de esos objetos hereda de PipelineStage.

Por eso `LPRPipeline` usa `PipelineStage`: espera recibir objetos que cumplan ese contrato (process()). LPRPipeline tiene varias PipelineStage (composición/agregación).

#### `class PipelineStage`:

Clase base para todas las etapas del pipeline.

Responsabilidad:

Definir el contrato común de todas las etapas.

No debe conocer nada de OCR, OpenCV ni métricas.

A partir de ella heredan*:

- `PlateDetector`
- `ROINormalizer`
- `TesseractOCR`
- `PostProcessor`

*`Evaluator` queda fuera de esta jerarquía porque no transforma datos del pipeline; evalúa resultados.

`PipelineStage.método`:

`process(entrada)`

Todas las clases hijas lo implementan.

#### `PlateDetector`

Responsabilidad: 

Localizar la patente.

Recibe:

imagen

Devuelve:

ROI

Internamente utiliza YOLO para seleccionar la mejor detección y recortar la región de interés (ROI).

La primera versión del proyecto utilizaba un detector basado en Canny y contornos. Debido a la elevada cantidad de falsos positivos producidos por el filtrado basado en área y relación de aspecto, dicha implementación fue reemplazada por un detector basado en YOLOv8, que proporciona una localización más robusta de la placa mediante un modelo de aprendizaje profundo.

### Modelo de detección de placas

La detección de placas se implementa mediante un modelo **YOLOv8** preentrenado, sin realizar entrenamiento adicional ni *fine-tuning*, utilizando la biblioteca **Ultralytics**.

##### Instalación

Con el entorno virtual activado, instalar la dependencia:

```bash
pip install ultralytics
```

#### Descarga del modelo

Se utilizó el modelo **YOLOv8 | Plate Detection Model & Finetuned weights**, disponible en Kaggle.

1. Acceder al dataset en Kaggle.
2. Descargar el archivo comprimido.
3. Extraer el archivo `best.pt`.

#### Integración en el proyecto

1. Crear el directorio `models/` en la raíz del proyecto.
2. Copiar `best.pt` dentro de dicho directorio.
3. Renombrar el archivo a `plate_detector.pt`.

```text
pipeline-LPR/
├── models/
│   └── plate_detector.pt
.
.
.
```

El modelo se carga al instanciar la clase `PlateDetector` mediante la biblioteca Ultralytics:

```python
from ultralytics import YOLO

self.model = YOLO("models/plate_detector.pt")
```

Durante la ejecución, YOLO realiza la inferencia sobre la imagen de entrada y devuelve las detecciones de placas con sus correspondientes coordenadas (*bounding box*), puntaje de confianza y clase detectada.

---

#### `ROINormalizer`

Responsabilidad: 

Normalizar la ROI para el OCR.

Recibe:

ROI

Devuelve:

ROI normalizada

Aquí se relaizan las operaciones de redimensionado, escala de grises, binarización y mejora de contraste. No reconoce texto; solo optimiza la imagen para la siguiente etapa.
```
ROINormalizer
│
├── process()
├── _resize()
├── _grayscale()
└── _enhance_contrast()
```

#### `TesseractOCR`

Responsabilidad: 

Reconocer los caracteres de la patente.

Recibe:

ROI normalizada

Devuelve:

texto reconocido + score OCR

Su única tarea es extraer el texto mediante Tesseract. No valida el formato ni corrige errores.

Configuración inicial: 

`--oem 3`
Utiliza automáticamente el mejor motor OCR disponible (LSTM cuando está presente).
Es la configuración recomendada en versiones actuales de Tesseract.

`--psm 7`
Le indica a Tesseract que la imagen contiene una única línea de texto.
Una patente normalmente es exactamente eso, por lo que suele dar mejores resultados que otros modos.

`tessedit_char_whitelist`
Limita los caracteres posibles a:
A-Z
0-9
Reduce errores como interpretar símbolos o letras que nunca aparecerán en una patente.

#### `PostProcessor`

Responsabilidad

Mejorar y validar el resultado del OCR.

Recibe:

texto reconocido + score OCR

Devuelve:

texto corregido + score final

Aquí se realizan correcciones (por ejemplo 0↔O, 1↔I), validación mediante `Regex` y cálculo del score final. No ejecuta OCR; trabaja únicamente sobre el resultado obtenido.

#### `Evaluator`

Responsabilidad 

Medir el desempeño del pipeline.

Recibe:

resultado del pipeline + ground truth

Devuelve:

Accuracy, TP, FP, FN, Precision, Recall, F1-score

Se ejecuta al finalizar el pipeline para comparar los resultados obtenidos con los esperados y generar las métricas del sistema. No forma parte del flujo de procesamiento de imágenes.

### main.py

```
main.py
│
├── cargar imagen
├── crear LPRPipeline
├── ejecutar pipeline
├── mostrar resultado
└── evaluar
```

---

## Orden de implementación de proyecto
- Diseño y Estructura del proyecto.
- Definición de las clases.
- Clase base del pipeline.
- Dataset.
- Detector.
- Normalización.
- OCR.
- Postprocesado.
- Pipeline completo.
- Métricas.
- Tests.
- Documentación.

---

## Habilidades

- Diseño orientado a objetos.
- Funcionamiento del algoritmo.
- Arquitectura.
- POO.
- Modularidad.
- Buenas prácticas.
- Documentación.
- Tests.
- Evolución incremental.

---

## Autor

Mariana Emilia Mazzoccoli (2026)

# Apendice: El motor OCR (Tesseract OCR)
Es un programa independiente que realiza el reconocimiento de texto.

`pytesseract` es solo un "puente" entre Python y ese programa.

El flujo queda así:
```
image
   ↓

PlateDetector (YOLO)
   ↓

ROINormalizer
(OpenCV)
   ↓

pytesseract
(interfaz Python)
   ↓
   
Tesseract OCR
(reconoce el texto)
   ↓
   
"AB123CD"
```

En Windows se instala mediante un instalador (.exe):

https://github.com/UB-Mannheim/tesseract/wiki

Instalar Tesseract OCR.

Después, en el código, normalmente se indica dónde está instalado:

`import pytesseract`

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

El proyecto incluiye OpenCV + OCR + un pipeline LPR, por tanto esa es la configuración estándar y la que más se suele utilizar en proyectos académicos y de prototipado. El diseño escalable permite reemplazar Tesseract por un OCR basado en redes neuronales (como EasyOCR o PaddleOCR), en cuyo caso la arquitectura del pipeline prácticamente no cambia; solo cambia la etapa de OCR.