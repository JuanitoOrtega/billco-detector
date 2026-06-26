# Proyecto de Maestría
## Sistema Inteligente para la Detección y Cálculo Automático de Monedas y Billetes Bolivianos utilizando YOLO y PyQt6

Actúa como un Ingeniero Senior en Inteligencia Artificial, Computer Vision y desarrollo de aplicaciones de escritorio con Python.

Quiero desarrollar un proyecto académico completo para una materia de Maestría en Visión Artificial.

El proyecto consiste en desarrollar un sistema capaz de detectar automáticamente monedas y billetes del Estado Plurinacional de Bolivia utilizando un modelo YOLO entrenado específicamente para esta tarea.

El proyecto NO debe depender de modelos preentrenados que ya reconozcan dinero boliviano. Debe incluir el proceso completo de construcción del dataset, entrenamiento del modelo e integración con una aplicación de escritorio desarrollada en PyQt6.

El objetivo es demostrar todo el flujo de un proyecto de Visión Artificial.

---

# Objetivos

El proyecto debe incluir tres módulos principales:

## Módulo 1
Construcción del Dataset

## Módulo 2
Entrenamiento del modelo YOLO

## Módulo 3
Aplicación de escritorio

Cada módulo debe estar completamente documentado.

---

# Monedas y billetes a detectar

El sistema deberá reconocer las siguientes denominaciones de Bolivia:

Moneda Bs 0.50

Moneda Bs 1

Moneda Bs 2

Moneda Bs 5

Billete Bs 10

Billete Bs 20

Billete Bs 50

Billete Bs 100

Billete Bs 200

Cada una será una clase independiente del modelo YOLO.

---

# Dataset

Explicar detalladamente cómo construir un dataset de calidad.

Incluir recomendaciones sobre:

Cantidad de imágenes por clase.

Variaciones de iluminación.

Fondos diferentes.

Rotaciones.

Distancias.

Perspectivas.

Objetos parcialmente ocultos.

Varias monedas juntas.

Varias denominaciones.

Escenarios reales.

Explicar cómo dividir el dataset en:

Train

Validation

Test

Explicar el formato YOLO.

Generar el archivo dataset.yaml.

---

# Etiquetado

Explicar paso a paso cómo etiquetar las imágenes.

Mostrar ejemplos.

Explicar buenas prácticas.

---

# Entrenamiento

Crear un módulo completo dedicado al entrenamiento.

Explicar:

Selección del modelo base (por ejemplo, yolo11n).

Parámetros recomendados.

Epochs.

Batch size.

Image Size.

Learning Rate.

Data Augmentation.

Early Stopping.

Explicar las métricas obtenidas:

Precision

Recall

mAP50

mAP50-95

Loss

Explicar cómo interpretar las gráficas generadas durante el entrenamiento.

Mostrar cómo obtener el archivo final best.pt.

---

# Aplicación PyQt6

Crear una aplicación profesional con una interfaz moderna.

Debe incluir:

Vista de cámara en tiempo real.

Bounding Boxes.

Nombre de la denominación.

Confianza.

Lista de objetos detectados.

Monto total acumulado.

FPS.

Estado del modelo.

Cantidad de objetos detectados.

Botones:

Iniciar cámara

Detener cámara

Capturar imagen

Guardar captura

Salir

---

# Arquitectura

El proyecto debe estar organizado en módulos independientes siguiendo buenas prácticas de ingeniería de software.

Separar claramente:

Interfaz

Visión Artificial

Modelo YOLO

Utilidades

Configuración

---

# Calidad del código

Seguir PEP8.

Utilizar clases.

Type Hints.

Docstrings.

Comentarios donde sean necesarios.

Evitar duplicación de código.

Arquitectura limpia.

---

# Documentación

Generar un README completo.

Además generar un documento técnico que explique el proceso completo de entrenamiento y funcionamiento del sistema.

---

# Resultado esperado

Quiero un proyecto profesional, bien estructurado y listo para ser presentado como trabajo final de una materia de Maestría en Visión Artificial.

No omitas ningún archivo.

Genera el proyecto paso a paso, comenzando por el diseño de la arquitectura antes de escribir código.