"""Recursos gráficos de la aplicación (icono)."""
from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPen, QPixmap

from config import config


def crear_icono(tam: int = 256) -> QIcon:
    """Devuelve el icono de la aplicación.

    Usa el archivo ``assets/icono.png`` si existe; de lo contrario lo dibuja
    en tiempo de ejecución. Esto permite reemplazar el icono dejando un PNG
    en esa ruta, sin tocar el código.
    """
    if config.RUTA_ICONO.exists():
        return QIcon(str(config.RUTA_ICONO))
    return QIcon(dibujar_icono(tam))


def dibujar_icono(tam: int = 256) -> QPixmap:
    """Dibuja un icono de moneda dorada con el texto 'Bs'."""
    pixmap = QPixmap(tam, tam)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    margen = tam * 0.08
    circulo = QRectF(margen, margen, tam - 2 * margen, tam - 2 * margen)

    # Moneda: relleno dorado con borde más oscuro.
    painter.setBrush(QBrush(QColor("#D4AF37")))
    painter.setPen(QPen(QColor("#7A5C12"), tam * 0.04))
    painter.drawEllipse(circulo)

    # Texto 'Bs' centrado.
    fuente = QFont("Arial", int(tam * 0.34), QFont.Weight.Bold)
    painter.setFont(fuente)
    painter.setPen(QPen(QColor("#3A2C08")))
    painter.drawText(circulo, Qt.AlignmentFlag.AlignCenter, "Bs")

    painter.end()
    return pixmap
