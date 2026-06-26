# Documento Técnico — Billco Detector

**Sistema de Detección de Moneda Boliviana mediante Visión por Computadora**

Trabajo de Maestría — Inteligencia Artificial aplicada

---

## 1. Introducción y objetivo

Bolivia utiliza nueve denominaciones de moneda corriente: cuatro monedas (Bs 0.50, Bs 1, Bs 2, Bs 5) y cinco billetes (Bs 10, Bs 20, Bs 50, Bs 100, Bs 200). La identificación visual automática de estas denominaciones tiene aplicaciones en sistemas de cobro asistido, accesibilidad para personas con discapacidad visual y automatización de procesos comerciales.

El presente proyecto desarrolla un sistema completo de detección en tiempo real que combina un modelo de detección de objetos entrenado a medida con una aplicación de escritorio multiplataforma. El objetivo central es que, dado un flujo de video procedente de una cámara web, el sistema identifique cada denominación visible, la muestre con un cuadro delimitador etiquetado y calcule el monto total en bolivianos.

El sistema está compuesto por tres módulos: preparación del dataset, entrenamiento y evaluación del modelo, y aplicación de escritorio con interfaz gráfica PyQt6.

---

## 2. Construcción del dataset

### 2.1 Criterios de recolección de imágenes

Un modelo de detección de objetos solo aprende lo que ve durante el entrenamiento. Por eso la diversidad del dataset es determinante para la robustez del sistema en condiciones reales. Se establecieron los siguientes criterios:

- **Cantidad mínima por clase:** entre 150 y 300 imágenes. Con menos de 100 imágenes el modelo tiende a sobreajustarse; superar las 500 por clase en una primera versión no aporta proporcionalidad respecto al esfuerzo de etiquetado.
- **Iluminación:** luz natural de ventana, luz artificial de oficina, iluminación lateral, contraluz suave y sombras parciales. La iluminación es la fuente de variación más crítica porque cambia drásticamente la apariencia del metal y del papel.
- **Fondos:** superficies de distintos colores y texturas (madera oscura, madera clara, tela de distintos colores, papel blanco, escritorio gris, suelo) para evitar que el modelo memorice el fondo.
- **Rotaciones:** 0°, 45°, 90°, 135°, 180° y ángulos intermedios. Las monedas son simétricas pero los billetes tienen orientación; ambos pueden estar en cualquier posición sobre una superficie.
- **Distancias:** primer plano (moneda ocupa más del 50 % del encuadre), distancia media y plano general con varias denominaciones.
- **Perspectivas:** vista frontal cenital, inclinación lateral de 20°–45°, vista oblicua.
- **Oclusión:** una moneda o billete parcialmente tapado por otro objeto o denominación. El modelo debe aprender a detectar instancias parcialmente visibles.
- **Escenas mixtas:** varias monedas y distintos billetes en el mismo encuadre, que es el caso de uso más habitual.
- **Escenarios reales:** mostrador de tienda, billetera abierta, escritorio de trabajo, bandeja de caja registradora.

### 2.2 División del dataset — train / val / test (70 / 20 / 10)

El dataset total se divide en tres subconjuntos con proporciones 70 % entrenamiento, 20 % validación y 10 % prueba, usando semilla aleatoria 42 para reproducibilidad.

**Por qué 70 / 20 / 10:**

- El **conjunto de entrenamiento** (70 %) maximiza la información disponible para que el modelo ajuste sus pesos. Con menos del 60 % el modelo tiene menos capacidad de generalización; superar el 80 % deja poco margen para la validación.
- El **conjunto de validación** (20 %) se usa durante el entrenamiento para medir el desempeño en datos no vistos en cada época, lo que permite aplicar early stopping y seleccionar el mejor checkpoint. Necesita ser lo suficientemente grande para que las métricas sean estables.
- El **conjunto de prueba** (10 %) se reserva estrictamente para la evaluación final y nunca influye en el entrenamiento ni en la selección de hiperparámetros. Aunque es el subconjunto más pequeño, basta para obtener una estimación honesta del rendimiento real si el dataset total es suficientemente grande.

La división se realiza con el script `src/dataset/dividir_dataset.py`, que mezcla aleatoriamente los pares imagen-etiqueta antes de separar para evitar sesgos por orden de captura.

---

## 3. Formato YOLO de anotaciones

### 3.1 Estructura del archivo de etiqueta

YOLO usa un archivo de texto plano (`.txt`) por imagen, con el mismo nombre base que el archivo de imagen. Cada línea representa un objeto detectado y sigue el formato:

```
<id_clase> <cx> <cy> <ancho> <alto>
```

Todos los valores son números en punto flotante. Las coordenadas y dimensiones están **normalizadas** entre 0 y 1 respecto al tamaño de la imagen (es decir, divididas entre el ancho y el alto en píxeles). Esto hace que las anotaciones sean independientes de la resolución.

**Ejemplo de línea real:**

```
0 0.512 0.488 0.210 0.305
```

Interpretación: clase `0` (moneda_050), centro en x=51.2 %, centro en y=48.8 %, ancho=21.0 % del encuadre, alto=30.5 % del encuadre.

Si la imagen contiene tres objetos, el archivo tendrá tres líneas, una por objeto.

### 3.2 Tabla de IDs de clase

| ID | Nombre de clase | Valor |
|----|-----------------|-------|
| 0  | moneda_050      | Bs 0.50 |
| 1  | moneda_1        | Bs 1  |
| 2  | moneda_2        | Bs 2  |
| 3  | moneda_5        | Bs 5  |
| 4  | billete_10      | Bs 10 |
| 5  | billete_20      | Bs 20 |
| 6  | billete_50      | Bs 50 |
| 7  | billete_100     | Bs 100 |
| 8  | billete_200     | Bs 200 |

### 3.3 Archivo dataset.yaml

El archivo `dataset.yaml` describe la ubicación del dataset y el listado de clases. Es la pieza de configuración que YOLO/Ultralytics utiliza para encontrar las imágenes y etiquetas durante el entrenamiento.

```yaml
path: /ruta/absoluta/al/proyecto/dataset
train: train/images
val: val/images
test: test/images
nc: 9
names:
  0: moneda_050
  1: moneda_1
  2: moneda_2
  3: moneda_5
  4: billete_10
  5: billete_20
  6: billete_50
  7: billete_100
  8: billete_200
```

El campo `nc` indica el número de clases (9 en este proyecto). `path` es la ruta base; `train`, `val` y `test` son rutas relativas a esa base que apuntan a las carpetas de imágenes. Ultralytics asume que los archivos de etiquetas se encuentran en la carpeta hermana `labels/` (es decir, reemplazando `images` por `labels` en la ruta).

El script `python -m src.dataset.generar_yaml` genera este archivo automáticamente con la ruta absoluta correcta del sistema en ejecución.

---

## 4. Etiquetado: proceso y buenas prácticas

### 4.1 Proceso con LabelImg

1. Instalar LabelImg (`pip install labelImg`) y abrirlo con el comando `labelImg`.
2. En la interfaz, seleccionar **Open Dir** y apuntar a la carpeta `images/` del dataset.
3. En **Change Save Dir**, seleccionar la carpeta `labels/`.
4. Cambiar el formato de guardado a **YOLO** (menú desplegable en la barra lateral).
5. Para cada imagen, pulsar **Create RectBox** (o la tecla W) y trazar el rectángulo alrededor de cada denominación.
6. Asignar el nombre de clase correcto en el cuadro de diálogo que aparece.
7. Guardar con Ctrl+S antes de avanzar a la siguiente imagen (tecla D).
8. Repetir hasta completar todas las imágenes.

Alternativa en línea: Roboflow permite subir imágenes, etiquetar en el navegador, y exportar directamente en formato YOLO con división automática del dataset.

### 4.2 Buenas prácticas

- **Cajas ajustadas:** el rectángulo debe rodear lo más ceñidamente posible el objeto visible. Un margen excesivo introduce ruido de fondo en el parche que el modelo aprende; un recorte demasiado justo puede cortar información de borde del objeto.
- **Etiquetar todas las instancias:** si hay cinco monedas en la imagen, deben anotarse las cinco. Dejar instancias sin etiquetar introduce falsos negativos en el entrenamiento, penalizando al modelo por detectar objetos que en realidad sí estaban presentes.
- **Consistencia de clases:** utilizar siempre exactamente los mismos nombres de clase (mayúsculas, guiones bajos, etc.). Una inconsistencia como `Moneda_050` vs `moneda_050` crea una décima clase fantasma que confunde al modelo.
- **Oclusión parcial:** etiquetar un objeto parcialmente oculto si puede identificarse su denominación. Si más del 50 % está tapado y la clase no es identificable con seguridad, es preferible omitirlo.
- **Control de calidad:** tras completar el etiquetado, ejecutar `python -m src.dataset.verificar_dataset` para detectar archivos de etiqueta con errores de formato, coordenadas fuera de rango o referencias a IDs de clase inexistentes.

---

## 5. Entrenamiento del modelo

### 5.1 Elección de YOLOv11 nano (yolo11n)

La familia YOLO (You Only Look Once) es el estándar de facto en detección de objetos en tiempo real por su balance entre velocidad y precisión. Se eligió la variante **yolo11n** (nano) por las siguientes razones:

- Es la variante más ligera de la familia YOLO11, con el menor número de parámetros. Esto permite ejecutar inferencia en tiempo real incluso en equipos sin GPU dedicada, que es el escenario más probable en el entorno de uso del proyecto.
- En datasets pequeños (pocos miles de imágenes), las variantes nano y small generalizan mejor que las variantes large y xlarge, que tienden a sobreajustarse sin suficientes datos.
- El modelo base `yolo11n.pt` está preentrenado sobre COCO (dataset de 80 clases de objetos cotidianos), lo que permite aprovechar los pesos aprendidos en millones de imágenes mediante fine-tuning, acortando drásticamente el tiempo de entrenamiento.

### 5.2 Parámetros de entrenamiento

Los parámetros están centralizados en `config/config.py`:

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| MODELO_BASE | yolo11n.pt | Pesos preentrenados en COCO, tamaño mínimo |
| IMGSZ | 640 | Resolución estándar YOLO; equilibra detalle y velocidad |
| EPOCHS | 100 | Máximo de épocas; el early stopping detiene antes si converge |
| BATCH | 16 | Funciona en GPU con 4 GB VRAM; reducir a 8 si hay problemas de memoria |
| PATIENCE | 20 | Early stopping: detiene si no mejora mAP50 en 20 épocas consecutivas |
| LR0 | 0.01 | Tasa de aprendizaje inicial; Ultralytics ajusta con scheduler coseno |
| CONF_MIN | 0.5 | Umbral de confianza mínima en inferencia |

**Data augmentation:** Ultralytics aplica aumentaciones automáticas durante el entrenamiento (mosaico, flip horizontal, cambios de brillo/contraste, recorte aleatorio, mezcla de imágenes). Esto amplifica efectivamente el tamaño del dataset y mejora la robustez sin necesidad de generar imágenes aumentadas manualmente.

**Early stopping con `patience`:** si la métrica mAP50 en validación no mejora durante 20 épocas consecutivas, el entrenamiento se detiene anticipadamente. Esto previene el sobreajuste y ahorra tiempo de cómputo. Los pesos del mejor epoch (según mAP50 en validación) se guardan como `best.pt`.

### 5.3 Comando de entrenamiento

```bash
python -m src.entrenamiento.entrenar
# Con parámetros personalizados:
python -m src.entrenamiento.entrenar --epochs 50 --batch 8
```

Al terminar, el script copia automáticamente `best.pt` a `modelos/best.pt`.

---

## 6. Métricas de evaluación

### 6.1 Definiciones

**Precision (Precisión)**

Fracción de detecciones positivas que son correctas:

```
Precision = VP / (VP + FP)
```

Donde VP son verdaderos positivos (detecciones correctas) y FP son falsos positivos (detecciones incorrectas). Una precisión alta significa que cuando el modelo dice que hay una moneda, casi siempre tiene razón.

**Recall (Exhaustividad)**

Fracción de objetos reales que el modelo detectó:

```
Recall = VP / (VP + FN)
```

Donde FN son falsos negativos (objetos que el modelo no detectó). Un recall alto significa que el modelo raramente omite objetos reales.

**mAP50 (mean Average Precision al 50 % de IoU)**

Promedio de la precisión media (AP) de cada clase, calculado cuando se considera una detección como correcta si el solapamiento entre la caja predicha y la caja real (IoU, Intersection over Union) es mayor o igual al 50 %. Es la métrica principal para comparar modelos de detección de objetos.

**mAP50-95 (mean Average Precision de 50 % a 95 % de IoU)**

Igual que mAP50 pero promediado sobre diez umbrales de IoU (0.50, 0.55, 0.60, …, 0.95). Es una métrica más exigente que penaliza cajas poco ajustadas. Un modelo puede tener mAP50 alto y mAP50-95 bajo si sus cajas son imprecisas.

**Loss (función de pérdida)**

Durante el entrenamiento se optimizan tres pérdidas complementarias:

- `box_loss`: error en las coordenadas y dimensiones de la caja predicha.
- `cls_loss`: error en la clasificación de la clase.
- `dfl_loss`: error de distribución focal, relacionado con la calidad del ajuste de la caja.

El valor de loss en validación debe decrecer con las épocas y estabilizarse. Si la loss de validación empieza a crecer mientras la de entrenamiento sigue bajando, hay sobreajuste.

### 6.2 Cómo interpretar las métricas

- **mAP50 > 0.90:** rendimiento excelente para una aplicación real.
- **mAP50 entre 0.70 y 0.90:** rendimiento aceptable; revisar qué clases tienen AP más bajo y añadir más datos de esas clases.
- **mAP50 < 0.70:** el modelo no es suficientemente fiable; ampliar el dataset o revisar la calidad del etiquetado.
- **Precision alta, Recall bajo:** el modelo es conservador; sube el umbral de confianza o el modelo está perdiendo muchas detecciones.
- **Recall alto, Precision bajo:** el modelo detecta demasiado; reducir el umbral de confianza o mejorar la separación entre clases.

Para evaluar sobre el conjunto de prueba:

```bash
python -m src.entrenamiento.evaluar
python -m src.entrenamiento.evaluar --modelo modelos/best.pt
```

---

## 7. Interpretación de gráficas de Ultralytics

Ultralytics genera automáticamente varias gráficas en la carpeta `runs/detect/train/` tras el entrenamiento.

### 7.1 results.png

Muestra la evolución por época de las métricas y las pérdidas. Las subgráficas clave son:

- **train/box_loss, train/cls_loss, train/dfl_loss:** deben decrecer monótonamente. Si oscilan mucho, la tasa de aprendizaje puede ser demasiado alta.
- **val/box_loss, val/cls_loss, val/dfl_loss:** deben descender en paralelo con las pérdidas de entrenamiento. Divergencia indica sobreajuste.
- **metrics/precision(B), metrics/recall(B):** deben crecer y estabilizarse. Si el recall converge antes que la precisión, el modelo tiene dificultad para reducir falsos positivos.
- **metrics/mAP50(B), metrics/mAP50-95(B):** la curva de mAP50 debe subir rápidamente en las primeras épocas y luego aplanarse. El punto donde se estabiliza indica la época del mejor checkpoint.

### 7.2 Matriz de confusión (confusion_matrix.png)

Muestra cuántos objetos de cada clase verdadera fueron predichos como cada clase posible. La diagonal principal representa las predicciones correctas. Las celdas fuera de la diagonal revelan confusiones sistemáticas entre clases.

Para este proyecto, las confusiones más esperables son:
- Entre monedas de denominaciones similares (Bs 1 y Bs 2 tienen tamaños parecidos).
- Entre billetes del mismo color base (Bs 10 y Bs 20 comparten familia de diseño).

Si la matriz muestra una clase con muchos errores, se debe aumentar el número de imágenes de esa clase o mejorar la calidad del etiquetado.

### 7.3 Curvas Precision-Recall (PR_curve.png)

Muestra el trade-off entre precisión y recall a distintos umbrales de confianza para cada clase. El área bajo cada curva es el AP de esa clase. Una curva ideal se acerca al ángulo superior derecho (precision=1, recall=1). La línea horizontal promediada es el mAP50 reportado.

Si una clase tiene una curva que cae rápidamente, significa que para obtener un recall alto esa clase necesita un umbral de confianza muy bajo, lo que puede introducir falsos positivos.

### 7.4 Obtención de best.pt

Ultralytics guarda automáticamente los pesos del epoch con el mejor mAP50 en validación en:

```
runs/detect/train/weights/best.pt
```

El script de entrenamiento del proyecto copia este archivo a `modelos/best.pt` al finalizar. Si se necesita comparar varios experimentos, los pesos de cada run quedan preservados en su carpeta `runs/detect/trainN/weights/best.pt`.

---

## 8. Arquitectura de la aplicación

### 8.1 Separación de responsabilidades

La aplicación sigue el principio de separación de responsabilidades. El código está organizado en cinco capas con dependencias unidireccionales:

```
config/
└── config.py          ← Única fuente de verdad (clases, valores, rutas, parámetros)

src/utils/
└── denominaciones.py  ← Lógica de negocio (mapeo ID→nombre, cálculo de monto)

src/vision/
├── detector.py        ← Inferencia YOLO (carga, detección, dibujo de cajas)
└── camara.py          ← Captura de video en hilo separado (QThread)

src/app/
├── ventana_principal.py  ← Orquestador: conecta cámara, detector y panel
└── widgets.py            ← Componentes de UI reutilizables (PanelDetecciones)
```

- **config** no importa nada del proyecto.
- **utils** solo importa de `config`.
- **vision** importa de `config` y `utils`.
- **app** importa de `vision`, `utils` y `config`.
- Ninguna capa importa de una capa superior a ella (no hay dependencias circulares).

### 8.2 Flujo cámara → detector → panel

```
[Cámara física]
      |
      v
[HiloCamara (QThread)]          -- captura frames en segundo plano
      |  frame_listo (señal Qt)
      v
[VentanaPrincipal._procesar_frame()]
      |
      +---> [Detector.detectar(frame)]    -- inferencia YOLO
      |           |
      |           v
      |       Lista[Deteccion]            -- clave, confianza, caja
      |
      +---> [Detector.dibujar(frame, detecciones)]  -- dibuja cajas y etiquetas
      |
      +---> [PanelDetecciones.actualizar(detecciones, fps, cargado)]
                  |
                  v
            [Interfaz Qt]                 -- lista, monto total, FPS, estado, conteo
```

El hilo de cámara (`HiloCamara`) emite la señal `frame_listo` con cada nuevo frame capturado. La ventana principal recibe la señal en el hilo principal de Qt (lo que garantiza la seguridad de la interfaz gráfica), ejecuta la inferencia, dibuja el resultado y actualiza el panel lateral.

La clase `Detector` encapsula el modelo YOLO y expone una interfaz simple: `cargar(ruta)`, `detectar(frame)` y `dibujar(frame, detecciones)`. Si el modelo no está cargado, `detectar` devuelve una lista vacía sin lanzar errores, lo que permite que la aplicación funcione en modo degradado sin `best.pt`.

### 8.3 Modelo de datos

La unidad de información de una detección es la dataclass `Deteccion`:

```python
@dataclass
class Deteccion:
    clave: str      # nombre de clase (ej. "billete_50")
    confianza: float  # valor entre 0 y 1
    caja: tuple[int, int, int, int]  # (x1, y1, x2, y2) en píxeles
```

El módulo `denominaciones.py` transforma estas detecciones en información de negocio: convierte `clave` en nombre legible (ej. "Bs 50") y calcula el monto total sumando `VALORES_BS[clave]` por cada detección.

---

## 9. Conclusiones

El proyecto demuestra la viabilidad de construir un sistema completo de detección de moneda boliviana con herramientas de código abierto (YOLO11, PyQt6, OpenCV) y un dataset propio de tamaño moderado.

Los factores clave para obtener un modelo robusto son la diversidad del dataset (iluminación, fondos, perspectivas, oclusión) y la calidad del etiquetado (cajas ajustadas, todas las instancias anotadas, consistencia de clases). La elección de `yolo11n` como arquitectura base permite inferencia en tiempo real incluso en equipos de gama media sin GPU dedicada.

La arquitectura de la aplicación, con separación clara entre capas y comunicación basada en señales Qt, facilita el mantenimiento y la extensión del sistema. Agregar nuevas denominaciones requiere únicamente actualizar `config/config.py` y reentrenar el modelo con ejemplos de las nuevas clases.

Las métricas de evaluación (mAP50, mAP50-95, Precision, Recall) y las gráficas generadas por Ultralytics proporcionan retroalimentación objetiva sobre el rendimiento del modelo y orientan las decisiones de mejora iterativa del dataset y del proceso de entrenamiento.

---

*Documento generado para el trabajo de maestría. Billco Detector — Sistema de detección de moneda boliviana.*
