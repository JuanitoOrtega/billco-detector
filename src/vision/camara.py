"""Captura de cámara en un hilo separado para no congelar la interfaz."""
from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class HiloCamara(QThread):
    """Lee frames de la webcam en segundo plano y los emite por señal."""

    frame_listo = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, indice_camara: int = 0) -> None:
        super().__init__()
        self._indice = indice_camara
        self._corriendo = False

    def run(self) -> None:
        """Bucle de captura; emite cada frame leído."""
        captura = cv2.VideoCapture(self._indice)
        if not captura.isOpened():
            self.error.emit("No se pudo abrir la cámara.")
            return
        self._corriendo = True
        while self._corriendo:
            ok, frame = captura.read()
            if not ok:
                self.error.emit("Fallo al leer un frame de la cámara.")
                break
            self.frame_listo.emit(frame)
        captura.release()

    def detener(self) -> None:
        """Detiene el bucle de captura y espera a que el hilo termine."""
        self._corriendo = False
        self.wait()
