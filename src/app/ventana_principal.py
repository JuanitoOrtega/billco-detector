"""Ventana principal de la aplicación Billco Detector."""
from __future__ import annotations

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
        self.btn_acerca = QPushButton("Acerca de")
        self.btn_salir = QPushButton("Salir")

        self.btn_iniciar.clicked.connect(self.iniciar_camara)
        self.btn_detener.clicked.connect(self.detener_camara)
        self.btn_capturar.clicked.connect(self.capturar)
        self.btn_guardar.clicked.connect(self.guardar_captura)
        self.btn_modelo.clicked.connect(self.cargar_modelo)
        self.btn_acerca.clicked.connect(self.acerca_de)
        self.btn_salir.clicked.connect(self.close)

        for b in (
            self.btn_iniciar, self.btn_detener, self.btn_capturar,
            self.btn_guardar, self.btn_modelo, self.btn_acerca, self.btn_salir,
        ):
            lateral.addWidget(b)

        contenedor = QWidget()
        contenedor.setLayout(lateral)
        layout.addWidget(contenedor, stretch=1)

        self.setCentralWidget(central)
        self.panel.actualizar([], 0.0, self.detector.cargado)

    def iniciar_camara(self) -> None:
        """Arranca el hilo de captura e inferencia."""
        if self.hilo and self.hilo.isRunning():
            return
        # El hilo hace la lectura, la inferencia y el dibujo; la GUI solo muestra.
        self.hilo = HiloCamara(self.detector)
        self.hilo.frame_listo.connect(self._al_recibir_frame)
        self.hilo.error.connect(lambda m: QMessageBox.warning(self, "Cámara", m))
        self.hilo.start()

    def detener_camara(self) -> None:
        """Detiene el hilo de captura."""
        if self.hilo:
            self.hilo.detener()
            self.hilo = None
        self.vista.setText("Cámara detenida")

    def _al_recibir_frame(
        self, frame: np.ndarray, detecciones: list, fps: float
    ) -> None:
        """Muestra el frame ya procesado por el hilo y actualiza el panel.

        Esta ranura corre en el hilo de la interfaz, pero solo hace trabajo
        ligero (mostrar la imagen); la inferencia ya se hizo en el hilo. Al
        terminar avisa al hilo (contrapresión) para que procese el siguiente
        frame, evitando que la cola de eventos se acumule.
        """
        self._ultimo_frame = frame
        self.panel.actualizar(detecciones, fps, self.detector.cargado)
        self._mostrar(frame)
        if self.hilo is not None:
            self.hilo.frame_consumido()

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

    def acerca_de(self) -> None:
        """Muestra una ventana con el nombre, versión, autor y descripción."""
        texto = (
            f"<b>Billco Detector</b><br>"
            f"Versión {config.VERSION}<br><br>"
            "Detecta monedas y billetes bolivianos en tiempo real con un modelo "
            "YOLO11 y calcula automáticamente el monto total visible en la "
            "cámara.<br><br>"
            f"<b>Autor:</b> {config.AUTOR}"
        )
        QMessageBox.about(self, "Acerca de Billco Detector", texto)

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
