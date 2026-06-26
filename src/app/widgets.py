"""Componentes reutilizables de la interfaz."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGroupBox, QLabel, QListWidget, QVBoxLayout, QWidget

from src.utils import denominaciones
from src.vision.detector import Deteccion


class PanelDetecciones(QWidget):
    """Panel lateral con la información de detección en tiempo real."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        self.lbl_estado = QLabel("Modelo: no cargado")
        self.lbl_estado.setStyleSheet("font-weight: bold; color: #b00;")
        self.lbl_fps = QLabel("FPS: 0.0")
        self.lbl_conteo = QLabel("Objetos detectados: 0")
        self.lbl_total = QLabel("Total: Bs 0.00")
        self.lbl_total.setStyleSheet("font-size: 20px; font-weight: bold; color: #060;")

        grupo = QGroupBox("Detecciones")
        gl = QVBoxLayout(grupo)
        self.lista = QListWidget()
        gl.addWidget(self.lista)

        for w in (self.lbl_estado, self.lbl_fps, self.lbl_conteo, grupo, self.lbl_total):
            layout.addWidget(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    def actualizar(
        self, detecciones: list[Deteccion], fps: float, modelo_cargado: bool
    ) -> None:
        """Refresca todos los indicadores con el estado del frame actual."""
        if modelo_cargado:
            self.lbl_estado.setText("Modelo: cargado")
            self.lbl_estado.setStyleSheet("font-weight: bold; color: #060;")
        else:
            self.lbl_estado.setText("Modelo: no cargado")
            self.lbl_estado.setStyleSheet("font-weight: bold; color: #b00;")

        self.lbl_fps.setText(f"FPS: {fps:.1f}")
        self.lbl_conteo.setText(f"Objetos detectados: {len(detecciones)}")

        claves = [d.clave for d in detecciones]
        self.lbl_total.setText(f"Total: Bs {denominaciones.monto_total(claves):.2f}")

        self.lista.clear()
        for det in detecciones:
            self.lista.addItem(
                f"{denominaciones.nombre_de(det.clave)}  —  {det.confianza:.0%}"
            )
