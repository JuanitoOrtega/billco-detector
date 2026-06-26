# Especificación de diseño — BilLco Detector

**Fecha:** 2026-06-26
**Proyecto:** Sistema Inteligente para la Detección y Cálculo Automático de Monedas y Billetes Bolivianos (YOLO11 + PyQt6)
**Contexto:** Trabajo final de Maestría en Visión Artificial.

---

## 1. Objetivo

Construir un proyecto académico completo que demuestre el flujo end-to-end de un sistema de
Visión Artificial: construcción de dataset, entrenamiento de un modelo YOLO propio y una
aplicación de escritorio que detecta monedas y billetes bolivianos en tiempo real y calcula el
monto total acumulado.

El proyecto **no** depende de modelos preentrenados que ya reconozcan dinero boliviano. El modelo
se entrena específicamente para esta tarea.

## 2. Decisiones tomadas

- **Idioma:** documentación, comentarios, docstrings e interfaz en **español**.
- **Alcance de la sesión:** proyecto completo (código de los 3 módulos + documentación).
- **App sin modelo entrenado:** la aplicación abre y funciona aunque no exista `best.pt`. Muestra
  estado "modelo no cargado", permite cargar pesos manualmente y la cámara funciona igual.
- **Framework de modelo:** Ultralytics YOLO11 (modelo base `yolo11n`).
- **GUI:** PyQt6.

## 3. Clases del modelo (9)

| id | clave            | denominación | valor (Bs) |
|----|------------------|--------------|------------|
| 0  | `moneda_050`     | Moneda 0.50  | 0.50       |
| 1  | `moneda_1`       | Moneda 1     | 1.00       |
| 2  | `moneda_2`       | Moneda 2     | 2.00       |
| 3  | `moneda_5`       | Moneda 5     | 5.00       |
| 4  | `billete_10`     | Billete 10   | 10.00      |
| 5  | `billete_20`     | Billete 20   | 20.00      |
| 6  | `billete_50`     | Billete 50   | 50.00      |
| 7  | `billete_100`    | Billete 100  | 100.00     |
| 8  | `billete_200`    | Billete 200  | 200.00     |

Cada clase es independiente en el modelo YOLO.

## 4. Arquitectura

```
billco-detector/
├── README.md
├── requirements.txt
├── dataset.yaml                   # generado por script
├── main.py                        # punto de entrada de la app
├── docs/
│   └── documento_tecnico.md
├── config/
│   └── config.py                  # rutas, clases, parámetros (única fuente de verdad)
└── src/
    ├── dataset/                   # MÓDULO 1
    │   ├── dividir_dataset.py
    │   ├── generar_yaml.py
    │   └── verificar_dataset.py
    ├── entrenamiento/             # MÓDULO 2
    │   ├── entrenar.py
    │   └── evaluar.py
    ├── vision/                    # núcleo de visión
    │   ├── detector.py
    │   └── camara.py
    ├── app/                       # MÓDULO 3 (PyQt6)
    │   ├── ventana_principal.py
    │   └── widgets.py
    └── utils/
        └── denominaciones.py
```

**Separación de responsabilidades (requerida por el prompt):**

- **Interfaz:** `src/app/`
- **Visión Artificial:** `src/vision/`
- **Modelo YOLO:** `src/entrenamiento/` + `src/vision/detector.py`
- **Utilidades:** `src/utils/`
- **Configuración:** `config/`

## 5. Módulo 1 — Construcción del dataset

**Scripts (ejecutables):**

- `dividir_dataset.py`: divide un directorio de imágenes + etiquetas en `train/val/test`
  (70/20/10 por defecto, configurable), preservando el emparejamiento imagen↔etiqueta.
- `generar_yaml.py`: produce `dataset.yaml` con rutas y las 9 clases.
- `verificar_dataset.py`: valida que cada `.txt` cumpla el formato YOLO (clase + 4 valores
  normalizados en [0,1]), detecta imágenes sin etiqueta y clases fuera de rango.

**Documentación (README + documento técnico):** recomendaciones de cantidad de imágenes por clase
(~150–300), variaciones de iluminación, fondos, rotaciones, distancias, perspectivas, oclusión
parcial, varias monedas juntas, varias denominaciones y escenarios reales. Explicación del formato
YOLO. Guía de etiquetado paso a paso (LabelImg / Roboflow) con buenas prácticas y ejemplos.

## 6. Módulo 2 — Entrenamiento

- `entrenar.py`: entrena `yolo11n` vía Ultralytics con parámetros recomendados (epochs, batch,
  `imgsz=640`, learning rate, data augmentation, early stopping vía `patience`). Genera `best.pt`.
- `evaluar.py`: valida sobre el conjunto de test y reporta Precision, Recall, mAP50, mAP50-95 y
  loss. La documentación explica cómo interpretar las gráficas de Ultralytics (`results.png`,
  matriz de confusión, curvas P/R) y cómo obtener `best.pt`.

## 7. Módulo 3 — Aplicación PyQt6

**Funciones visibles:**

- Vista de cámara en tiempo real con bounding boxes, nombre de denominación y confianza.
- Panel lateral: lista de objetos detectados, **monto total (Bs)**, FPS, estado del modelo,
  cantidad de objetos detectados.
- Botones: Iniciar cámara · Detener cámara · Capturar imagen · Guardar captura · Salir.

**Diseño técnico:**

- `camara.py`: captura de webcam en un hilo separado (`QThread`) para no congelar la UI; emite
  frames por señal.
- `detector.py`: carga el modelo, ejecuta inferencia sobre cada frame, devuelve detecciones
  (clase, confianza, bbox) y dibuja las cajas.
- `denominaciones.py`: mapea clase → valor en Bs y calcula el monto total acumulado de las
  detecciones del frame actual.
- **Robustez sin modelo:** si no existe `best.pt`, la ventana abre con estado "modelo no cargado";
  un botón/diálogo permite cargar pesos manualmente; la cámara funciona sin pesos.

## 8. Calidad del código

PEP8, clases, type hints, docstrings, comentarios donde aporten, sin duplicación. La configuración
(rutas, clases, valores, parámetros) vive en `config/` como única fuente de verdad.

## 9. Documentación

- `README.md`: instalación, uso de cada módulo, cómo entrenar, cómo correr la app.
- `docs/documento_tecnico.md`: proceso completo (dataset → etiquetado → entrenamiento → métricas →
  app) explicado para presentación académica.

## 10. Fuera de alcance (honesto)

- No se incluye un dataset real de imágenes ni pesos `best.pt` entrenados: requieren captura de
  fotos y GPU del autor. Los scripts de dataset y entrenamiento quedan listos y correctos para que
  el autor los ejecute con sus imágenes.
- La app es funcional hoy con webcam; detectará dinero boliviano una vez se entrene y se cargue
  `best.pt`.
- Sin empaquetado/instalador (`.exe`/`.app`), sin tests automatizados extensos más allá de las
  verificaciones de dataset, sin CI.
```