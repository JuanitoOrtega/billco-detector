"""Ventana principal de la aplicación Billco Detector."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from config import config
from src.app.recursos import crear_icono
from src.app.widgets import PanelDetecciones
from src.vision.camara import HiloCamara
from src.vision.detector import Detector


class VentanaPrincipal(QMainWindow):
    """Orquesta la cámara, el detector y la interfaz."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(
            f"Billco Detector v{config.VERSION} — Detección de moneda boliviana"
        )
        self.setWindowIcon(crear_icono())
        self.resize(1100, 700)

        self.detector = Detector()
        self.hilo: HiloCamara | None = None
        self._ultimo_frame: np.ndarray | None = None
        self._t_anterior = time.time()

        self._construir_ui()

    def _construir_ui(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)

        # Vista de cámara
        self.vista = QLabel("Cámara detenida")
        self.vista.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.vista.setMinimumSize(720, 540)
        self.vista.setStyleSheet("background:#111; color:#888;")
        layout.addWidget(self.vista, stretch=3)

        # Panel lateral + botones
        lateral = QVBoxLayout()
        self.panel = PanelDetecciones()
        lateral.addWidget(self.panel)

        self.btn_iniciar = QPushButton("Iniciar cámara")
        self.btn_detener = QPushButton("Detener cámara")
        self.btn_capturar = QPushButton("Capturar imagen")
        self.btn_guardar = QPushButton("Guardar captura")
        self.btn_modelo = QPushButton("Cargar modelo…")
        self.btn_salir = QPushButton("Salir")

        self.btn_iniciar.clicked.connect(self.iniciar_camara)
        self.btn_detener.clicked.connect(self.detener_camara)
        self.btn_capturar.clicked.connect(self.capturar)
        self.btn_guardar.clicked.connect(self.guardar_captura)
        self.btn_modelo.clicked.connect(self.cargar_modelo)
        self.btn_salir.clicked.connect(self.close)

        for b in (
            self.btn_iniciar, self.btn_detener, self.btn_capturar,
            self.btn_guardar, self.btn_modelo, self.btn_salir,
        ):
            lateral.addWidget(b)

        contenedor = QWidget()
        contenedor.setLayout(lateral)
        layout.addWidget(contenedor, stretch=1)

        self.setCentralWidget(central)
        self.panel.actualizar([], 0.0, self.detector.cargado)

    def iniciar_camara(self) -> None:
        """Arranca el hilo de captura."""
        if self.hilo and self.hilo.isRunning():
            return
        self.hilo = HiloCamara()
        self.hilo.frame_listo.connect(self._procesar_frame)
        self.hilo.error.connect(lambda m: QMessageBox.warning(self, "Cámara", m))
        self.hilo.start()

    def detener_camara(self) -> None:
        """Detiene el hilo de captura."""
        if self.hilo:
            self.hilo.detener()
            self.hilo = None
        self.vista.setText("Cámara detenida")

    def _procesar_frame(self, frame: np.ndarray) -> None:
        """Recibe un frame, ejecuta detección, dibuja y actualiza la UI."""
        detecciones = self.detector.detectar(frame)
        frame = self.detector.dibujar(frame, detecciones)
        self._ultimo_frame = frame.copy()

        ahora = time.time()
        fps = 1.0 / max(ahora - self._t_anterior, 1e-6)
        self._t_anterior = ahora

        self.panel.actualizar(detecciones, fps, self.detector.cargado)
        self._mostrar(frame)

    def _mostrar(self, frame: np.ndarray) -> None:
        """Convierte un frame BGR a QPixmap y lo muestra escalado."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        imagen = QImage(
            rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888
        ).copy()
        pix = QPixmap.fromImage(imagen).scaled(
            self.vista.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.vista.setPixmap(pix)

    def capturar(self) -> None:
        """Congela el último frame mostrándolo como captura."""
        if self._ultimo_frame is None:
            QMessageBox.information(self, "Capturar", "No hay frame disponible.")
            return
        self.detener_camara()
        self._mostrar(self._ultimo_frame)

    def guardar_captura(self) -> None:
        """Guarda el último frame en disco."""
        if self._ultimo_frame is None:
            QMessageBox.information(self, "Guardar", "No hay captura para guardar.")
            return
        config.DIR_CAPTURAS.mkdir(parents=True, exist_ok=True)
        sugerida = str(config.DIR_CAPTURAS / "captura.jpg")
        ruta, _ = QFileDialog.getSaveFileName(
            self, "Guardar captura", sugerida, "Imágenes (*.jpg *.png)"
        )
        if ruta:
            cv2.imwrite(ruta, self._ultimo_frame)
            QMessageBox.information(self, "Guardar", f"Captura guardada en:\n{ruta}")

    def cargar_modelo(self) -> None:
        """Permite seleccionar un best.pt manualmente."""
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Cargar modelo", str(config.RAIZ), "Modelos (*.pt)"
        )
        if ruta and self.detector.cargar(Path(ruta)):
            QMessageBox.information(self, "Modelo", "Modelo cargado correctamente.")
            self.panel.actualizar([], 0.0, True)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (firma de Qt)
        """Pide confirmación y detiene la cámara antes de cerrar la ventana.

        Cubre tanto el botón "Salir" como el botón de cierre del sistema.
        """
        cuadro = QMessageBox(self)
        cuadro.setWindowTitle("Salir")
        cuadro.setIcon(QMessageBox.Icon.Question)
        cuadro.setText("¿Seguro que deseas salir de Billco Detector?")
        btn_salir = cuadro.addButton(
            "Salir", QMessageBox.ButtonRole.AcceptRole
        )
        cuadro.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole)
        cuadro.setDefaultButton(btn_salir)
        cuadro.exec()

        if cuadro.clickedButton() is not btn_salir:
            event.ignore()
            return

        self.detener_camara()
        super().closeEvent(event)
