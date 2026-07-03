# Billco Detector

Sistema de detección de moneda boliviana en tiempo real mediante visión por computadora. Utiliza un modelo YOLO11 entrenado a medida para identificar 9 denominaciones de monedas y billetes del Banco Central de Bolivia y calcular el monto total visible en cámara.

Proyecto desarrollado como trabajo de maestría. El código está organizado en tres módulos independientes: preparación del dataset, entrenamiento del modelo y aplicación de escritorio.

---

## Clases detectadas

El modelo reconoce las siguientes 9 denominaciones (el índice es el ID de clase YOLO):

| ID | Clave          | Denominación |
|----|----------------|--------------|
| 0  | moneda_050     | Bs 0.50      |
| 1  | moneda_1       | Bs 1         |
| 2  | moneda_2       | Bs 2         |
| 3  | moneda_5       | Bs 5         |
| 4  | billete_10     | Bs 10        |
| 5  | billete_20     | Bs 20        |
| 6  | billete_50     | Bs 50        |
| 7  | billete_100    | Bs 100       |
| 8  | billete_200    | Bs 200       |

---

## Requisitos del sistema

- Python 3.10 o superior
- Cámara web (para la aplicación de escritorio)
- GPU opcional (el entrenamiento puede ejecutarse en CPU, aunque es más lento)

Dependencias principales: `ultralytics`, `PyQt6`, `opencv-python`, `numpy`, `PyYAML`, `pytest`.

---

## Instalación

```bash
# 1. Crear entorno virtual
python -m venv venv

# 2. Activar el entorno
# macOS / Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Módulo 1 — Dataset

### Estructura esperada

El directorio de origen que se pase a `dividir_dataset` debe tener la siguiente estructura:

```
<origen>/
    images/
        imagen_001.jpg
        imagen_002.jpg
        ...
    labels/
        imagen_001.txt
        imagen_002.txt
        ...
```

Los archivos de etiqueta deben seguir el formato YOLO: una línea por objeto con `<id_clase> <cx> <cy> <w> <h>` (valores normalizados de 0 a 1).

### Recomendaciones de captura

Para obtener un dataset robusto se recomienda:

- **Cantidad:** entre 150 y 300 imágenes por clase como mínimo.
- **Iluminación:** variar entre luz natural, artificial, directa y difusa; incluir sombras parciales.
- **Fondos:** capturar sobre superficies de distintos colores, texturas y patrones (madera, tela, papel, escritorios).
- **Rotaciones:** fotografiar las denominaciones en distintos ángulos (0°, 45°, 90°, 180°, etc.).
- **Distancias:** cerca (primer plano), distancia media y plano alejado.
- **Perspectivas:** frontal, ligera inclinación lateral, vista cenital.
- **Oclusión:** incluir imágenes donde una moneda o billete tape parcialmente a otro.
- **Escenas mixtas:** varias monedas y distintas denominaciones en el mismo encuadre.
- **Escenarios reales:** mostrador, billetera abierta, escritorio de trabajo.

### Guía de etiquetado con LabelImg

1. Instalar LabelImg: `pip install labelImg` y ejecutar `labelImg`.
2. Abrir la carpeta `images/` como directorio de trabajo.
3. Configurar el directorio de salida en `labels/` y seleccionar formato **YOLO**.
4. Para cada imagen, dibujar un rectángulo ajustado alrededor de cada denominación y asignar la clase correcta.
5. Guardar antes de pasar a la siguiente imagen (Ctrl+S).

**Buenas prácticas de etiquetado:**
- Las cajas deben estar bien ajustadas al contorno visible de cada moneda o billete; evitar márgenes excesivos.
- Etiquetar **todas** las instancias presentes en la imagen, incluso las parcialmente ocultas.
- Mantener consistencia en el nombre de clase (respetar la lista exacta del proyecto).
- Si una denominación está más del 50 % tapada y no es identificable con certeza, se puede omitir.

### Comandos del módulo 1

```bash
# Dividir dataset en train/val/test (70 % / 20 % / 10 %, semilla 42)
# <origen> debe contener subcarpetas images/ y labels/
python -m src.dataset.dividir_dataset <origen>

# Generar el archivo dataset.yaml necesario para el entrenamiento
python -m src.dataset.generar_yaml

# Verificar que las etiquetas tienen formato YOLO válido
python -m src.dataset.verificar_dataset
```

El script `dividir_dataset` acepta un segundo argumento opcional `[destino]` para indicar dónde crear las carpetas `dataset/train`, `dataset/val` y `dataset/test`. Si se omite, utiliza `dataset/` en la raíz del proyecto.

---

## Módulo 2 — Entrenamiento

El entrenamiento parte del modelo base `yolo11n.pt` (YOLOv11 nano) con fine-tuning sobre el dataset boliviano. Los pesos resultantes quedan en `modelos/best.pt`.

### Flujo de entrenamiento de principio a fin

El entrenamiento **necesita un dataset construido**: el repositorio no incluye imágenes ni un `best.pt` ya entrenado (debes generarlos con tus propias fotos). Sigue estos pasos en orden:

```bash
# 1. Reúne tus fotos etiquetadas (formato YOLO) en una carpeta con
#    subcarpetas images/ y labels/ (ver "Módulo 1 — Dataset"):
#    mis_fotos/
#    ├── images/   <- tus .jpg / .png
#    └── labels/   <- un .txt por imagen, en formato YOLO

# 2. Divide el dataset en train/val/test (crea la carpeta dataset/)
python -m src.dataset.dividir_dataset mis_fotos

# 3. Verifica que las etiquetas tengan formato YOLO válido
python -m src.dataset.verificar_dataset

# 4. Entrena el modelo (ahora sí encontrará las imágenes)
python -m src.entrenamiento.entrenar
```

> **Nota:** si ejecutas el entrenamiento sin haber construido el dataset, el
> programa no muestra un error críptico: imprime un mensaje claro indicando qué
> carpeta falta y los pasos anteriores para resolverlo.

El comando `dividir_dataset` ya genera la estructura que espera Ultralytics; `dataset.yaml` viene incluido en el repositorio, pero puedes regenerarlo en cualquier momento con `python -m src.dataset.generar_yaml`.

Parámetros de entrenamiento configurados en `config/config.py`:

| Parámetro | Valor |
|-----------|-------|
| MODELO_BASE | yolo11n.pt |
| IMGSZ | 640 |
| EPOCHS | 100 |
| BATCH | 16 |
| PATIENCE | 20 |
| LR0 | 0.01 |
| CONF_MIN | 0.5 |

### Comandos del módulo 2

```bash
# Entrenar el modelo (usa los parámetros de config/config.py)
python -m src.entrenamiento.entrenar

# Entrenar con parámetros personalizados
python -m src.entrenamiento.entrenar --epochs 50 --batch 8

# Evaluar el modelo sobre el conjunto de prueba
python -m src.entrenamiento.evaluar

# Evaluar con un modelo específico
python -m src.entrenamiento.evaluar --modelo modelos/best.pt
```

Al finalizar el entrenamiento, `best.pt` se copia automáticamente a `modelos/best.pt`.

---

## Módulo 3 — Aplicación de escritorio

```bash
python main.py
```

La aplicación abre correctamente aunque no exista `modelos/best.pt`; en ese caso muestra **"Modelo: no cargado"** y permite cargar un archivo `.pt` manualmente con el botón correspondiente.

### Botones

| Botón | Acción |
|-------|--------|
| Iniciar cámara | Arranca la captura de video en tiempo real |
| Detener cámara | Congela la imagen y detiene el hilo de captura |
| Capturar imagen | Guarda el frame actual en memoria para su revisión |
| Guardar captura | Exporta la captura a disco (`.jpg` / `.png`) |
| Cargar modelo… | Abre un diálogo para seleccionar un archivo `.pt` |
| Acerca de | Muestra nombre, versión, autor y descripción de la aplicación |
| Salir | Cierra la aplicación (pide confirmación) |

### Panel lateral

El panel lateral muestra en tiempo real:

- Lista de detecciones con clase y confianza por objeto detectado.
- Monto total acumulado en bolivianos (Bs).
- FPS actuales del flujo de video.
- Estado del modelo (`cargado` / `no cargado`).
- Cantidad de objetos detectados en el frame.

---

## Estructura de carpetas

```
billco-detector/
├── config/
│   └── config.py          # Configuración central (clases, valores, rutas, parámetros)
├── dataset/               # Generado por dividir_dataset
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── docs/
│   └── documento_tecnico.md
├── modelos/
│   └── best.pt            # Pesos del modelo entrenado
├── capturas/              # Capturas guardadas desde la app
├── src/
│   ├── app/
│   │   ├── ventana_principal.py
│   │   └── widgets.py
│   ├── dataset/
│   │   ├── dividir_dataset.py
│   │   ├── generar_yaml.py
│   │   └── verificar_dataset.py
│   ├── entrenamiento/
│   │   ├── entrenar.py
│   │   └── evaluar.py
│   ├── utils/
│   │   └── denominaciones.py
│   └── vision/
│       ├── camara.py
│       └── detector.py
├── tests/
├── dataset.yaml           # Generado por generar_yaml
├── main.py                # Punto de entrada de la aplicación
└── requirements.txt
```

---

## Versionado y releases

El proyecto usa versionado semántico (`MAYOR.MENOR.PARCHE`). La versión es única y vive en `config/config.py` (`VERSION`), y se muestra en el título de la ventana y en el diálogo **Acerca de**.

Los releases se publican de forma **manual**: al liberar una versión se sube `VERSION` (PARCHE para correcciones, MENOR para nuevas funciones, MAYOR para cambios incompatibles), se registra el cambio en [CHANGELOG.md](CHANGELOG.md) y se crea la etiqueta `vX.Y.Z` y el release en GitHub con su descripción.

## Licencia

Proyecto académico — Maestría en Inteligencia Artificial. Uso educativo.
