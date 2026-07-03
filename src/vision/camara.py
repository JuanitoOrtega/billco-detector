"""Captura de cámara e inferencia en un hilo separado.

La lectura de la cámara, la inferencia y el dibujo se ejecutan en este hilo,
de modo que el hilo de la interfaz solo tiene que mostrar el frame ya
procesado (una operación barata). Así el trabajo pesado (la inferencia YOLO)
nunca bloquea la interfaz.

Control de flujo (backpressure): el hilo procesa y emite un frame nuevo solo
cuando la interfaz ha mostrado el anterior. De este modo hay a lo sumo un
frame «en vuelo», la cola de eventos de la interfaz nunca crece sin límite y
la latencia cámara → pantalla se mantiene baja aunque la interfaz sea más
lenta que la cámara. Los frames intermedios se descartan (se muestra siempre
el más reciente).
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from src.vision.detector import Deteccion, Detector

DEBUG = bool(os.environ.get("BILLCO_DEBUG"))


def procesar(
    detector: Detector, frame: np.ndarray, t_anterior: float
) -> tuple[np.ndarray, list[Deteccion], float, float]:
    """Ejecuta la detección y el dibujo sobre un frame.

    Devuelve el frame anotado, las detecciones, los FPS instantáneos y el
    instante actual (para calcular los FPS del siguiente frame).
    """
    detecciones = detector.detectar(frame)
    frame = detector.dibujar(frame, detecciones)
    ahora = time.perf_counter()
    fps = 1.0 / max(ahora - t_anterior, 1e-6)
    return frame, detecciones, fps, ahora


class HiloCamara(QThread):
    """Lee la cámara, ejecuta la inferencia y emite el frame ya procesado."""

    frame_listo = pyqtSignal(np.ndarray, list, float)
    error = pyqtSignal(str)

    def __init__(self, detector: Detector, indice_camara: int = 0) -> None:
        super().__init__()
        self._detector = detector
        self._indice = indice_camara
        self._corriendo = False
        # Contrapresión: True mientras la interfaz muestra el frame anterior.
        self._gui_ocupada = False

    def frame_consumido(self) -> None:
        """La interfaz llama a este método tras mostrar un frame.

        Libera la contrapresión para que el hilo procese el siguiente.
        """
        self._gui_ocupada = False

    def run(self) -> None:
        """Bucle de captura + inferencia; emite cada frame ya procesado."""
        captura = cv2.VideoCapture(self._indice)
        if not captura.isOpened():
            self.error.emit("No se pudo abrir la cámara.")
            return
        try:
            # Mantener solo el frame más reciente evita acumular latencia
            # cuando la inferencia es más lenta que la cámara.
            captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass
        self._corriendo = True
        t_anterior = time.perf_counter()
        procesados = descartados = 0
        t_reporte = time.perf_counter()
        while self._corriendo:
            ok, frame = captura.read()
            if not ok:
                self.error.emit("Fallo al leer un frame de la cámara.")
                break
            if self._gui_ocupada:
                # La interfaz aún no terminó con el frame anterior: descartamos
                # este para no acumular latencia ni inundar su cola de eventos.
                descartados += 1
                continue
            frame, detecciones, fps, t_anterior = procesar(
                self._detector, frame, t_anterior
            )
            self._gui_ocupada = True
            self.frame_listo.emit(frame, detecciones, fps)
            procesados += 1

            if DEBUG and time.perf_counter() - t_reporte >= 2.0:
                lapso = time.perf_counter() - t_reporte
                print(
                    f"[camara] procesados={procesados / lapso:.1f} FPS  "
                    f"descartados={descartados / lapso:.1f}/s  "
                    f"(descartados altos = la interfaz no alcanza)",
                    flush=True,
                )
                procesados = descartados = 0
                t_reporte = time.perf_counter()
        captura.release()

    def detener(self) -> None:
        """Detiene el bucle de captura y espera a que el hilo termine."""
        self._corriendo = False
        self.wait()
