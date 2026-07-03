# Changelog

Todas las versiones notables de este proyecto se documentan en este archivo.
El formato sigue [Keep a Changelog](https://keepachangelog.com/es/1.1.0/) y el
proyecto usa versionado semántico (`MAYOR.MENOR.PARCHE`).

Las notas de cada release de GitHub se toman de la sección correspondiente de
este archivo, así que **cada cambio de `VERSION` debe añadir aquí su entrada**.

## [1.1.4] - 2026-06-30
### Corregido
- Se elimina la lentitud de la cámara con control de flujo (backpressure): el
  hilo procesa y emite un frame nuevo solo cuando la interfaz mostró el
  anterior, descartando los intermedios. Antes, el hilo emitía frames sin
  límite y la cola de eventos de la interfaz se acumulaba, aumentando la
  latencia hasta volver la aplicación inusable. Incluye telemetría opcional
  (variable de entorno BILLCO_DEBUG) que reporta los FPS procesados y los
  frames descartados.

## [1.1.3] - 2026-06-30
### Corregido
- La aplicación ya no se ralentiza al iniciar la cámara. La inferencia del
  modelo y el dibujo se movieron al hilo de la cámara; antes se ejecutaban en
  el hilo de la interfaz y la bloqueaban. La interfaz ahora solo muestra el
  frame ya procesado.

## [1.1.2] - 2026-06-26
### Cambiado
- Las notas de cada release ahora se generan a partir de `CHANGELOG.md` en
  lugar de notas automáticas, para tener descripciones claras de cada versión.

## [1.1.1] - 2026-06-26
### Cambiado
- Simplifica el texto del diálogo "Acerca de".

## [1.1.0] - 2026-06-26
### Añadido
- Botón y diálogo "Acerca de" con el nombre de la app, la versión, el autor y
  una breve descripción.

## [1.0.0] - 2026-06-26
### Añadido
- Primera versión estable: detección y suma automática de monedas y billetes
  bolivianos (9 clases) con YOLO11 y una app de escritorio en PyQt6.
- Módulo de dataset (dividir, verificar, generar `dataset.yaml`).
- Módulo de entrenamiento y evaluación con YOLO11n.
- Aplicación PyQt6 con cámara en tiempo real, detecciones, monto total en Bs,
  FPS, estado del modelo, icono propio y confirmación al salir.
- Documentación completa (README + documento técnico) en español.
