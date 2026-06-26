"""Punto de entrada de la aplicación Billco Detector."""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from src.app.ventana_principal import VentanaPrincipal


def main() -> None:
    """Lanza la aplicación de escritorio."""
    app = QApplication(sys.argv)
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
