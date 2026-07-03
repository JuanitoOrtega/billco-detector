"""Captura de cámara e inferencia en un hilo separado.

La lectura de la cámara, la inferencia y el dibujo se ejecutan en este hilo,
de modo que el hilo de la interfaz solo tiene que mostrar el frame ya
procesado (una operación barata). Así el trabajo pesado (la inferencia YOLO)
nunca bloquea la interfaz.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from src.vision.detector import Deteccion, Detector


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
        while self._corriendo:
            ok, frame = captura.read()
            if not ok:
                self.error.emit("Fallo al leer un frame de la cámara.")
                break
            frame, detecciones, fps, t_anterior = procesar(
                self._detector, frame, t_anterior
            )
            self.frame_listo.emit(frame, detecciones, fps)
        captura.release()

    def detener(self) -> None:
        """Detiene el bucle de captura y espera a que el hilo termine."""
        self._corriendo = False
        self.wait()
