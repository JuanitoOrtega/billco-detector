"""Wrapper de inferencia YOLO para detección de denominaciones."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import config
from src.utils import denominaciones


@dataclass
class Deteccion:
    """Una detección individual en un frame."""

    clave: str
    confianza: float
    caja: tuple[int, int, int, int]  # x1, y1, x2, y2


class Detector:
    """Carga un modelo YOLO y ejecuta inferencia sobre frames de imagen."""

    def __init__(self, ruta_modelo: Path = config.RUTA_MODELO) -> None:
        self._modelo = None
        self.ruta_modelo: Path | None = None
        if ruta_modelo.exists():
            self.cargar(ruta_modelo)

    @property
    def cargado(self) -> bool:
        """Indica si hay un modelo cargado y listo para inferir."""
        return self._modelo is not None

    def cargar(self, ruta: Path) -> bool:
        """Carga un modelo desde la ruta dada. Devuelve True si tuvo éxito."""
        from ultralytics import YOLO  # import perezoso: la app abre sin Ultralytics

        if not ruta.exists():
            return False
        self._modelo = YOLO(str(ruta))
        self.ruta_modelo = ruta
        return True

    def detectar(self, frame: np.ndarray) -> list[Deteccion]:
        """Ejecuta inferencia sobre un frame BGR y devuelve las detecciones."""
        if not self.cargado:
            return []
        resultados = self._modelo(frame, conf=config.CONF_MIN, verbose=False)[0]
        detecciones: list[Deteccion] = []
        for caja in resultados.boxes:
            id_clase = int(caja.cls[0])
            x1, y1, x2, y2 = (int(v) for v in caja.xyxy[0])
            detecciones.append(
                Deteccion(
                    clave=denominaciones.clave_de_id(id_clase),
                    confianza=float(caja.conf[0]),
                    caja=(x1, y1, x2, y2),
                )
            )
        return detecciones

    def dibujar(self, frame: np.ndarray, detecciones: list[Deteccion]) -> np.ndarray:
        """Dibuja las cajas, el nombre de la denominación y la confianza."""
        for det in detecciones:
            x1, y1, x2, y2 = det.caja
            etiqueta = f"{denominaciones.nombre_de(det.clave)} {det.confianza:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(
                frame, etiqueta, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2,
            )
        return frame
