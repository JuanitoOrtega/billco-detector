"""Tests del pipeline de procesamiento del hilo de cámara.

Verifican que la inferencia y el dibujo ocurren dentro del hilo de captura
(función ``procesar``), no en el hilo de la interfaz.
"""
import time

import numpy as np

from src.vision import camara


class _DetectorFalso:
    """Detector de prueba que registra cuántas veces se le llama."""

    def __init__(self) -> None:
        self.detectar_llamadas = 0
        self.dibujar_llamadas = 0

    def detectar(self, frame):
        self.detectar_llamadas += 1
        return ["moneda_1", "billete_10"]

    def dibujar(self, frame, detecciones):
        self.dibujar_llamadas += 1
        return frame


def test_procesar_ejecuta_inferencia_y_dibujo():
    det = _DetectorFalso()
    frame = np.zeros((4, 4, 3), dtype="uint8")
    salida, detecciones, fps, _ = camara.procesar(
        det, frame, time.perf_counter() - 0.05
    )
    assert det.detectar_llamadas == 1
    assert det.dibujar_llamadas == 1
    assert detecciones == ["moneda_1", "billete_10"]
    assert fps > 0
    assert salida is frame


def test_procesar_fps_finito_si_el_tiempo_no_avanza():
    det = _DetectorFalso()
    frame = np.zeros((4, 4, 3), dtype="uint8")
    _, _, fps, _ = camara.procesar(det, frame, time.perf_counter())
    assert fps > 0  # no divide por cero
