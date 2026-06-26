"""Entrenamiento del modelo YOLO11 para denominaciones bolivianas.

Ejecutable como CLI:  python -m src.entrenamiento.entrenar
Requiere: dataset.yaml válido y dependencias de Ultralytics instaladas.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from config import config


def entrenar(
    modelo_base: str = config.MODELO_BASE,
    datos: Path = config.RUTA_DATASET_YAML,
    epochs: int = config.EPOCHS,
    batch: int = config.BATCH,
    imgsz: int = config.IMGSZ,
    patience: int = config.PATIENCE,
    lr0: float = config.LR0,
) -> Path:
    """Entrena el modelo y copia best.pt a la carpeta de modelos del proyecto.

    Devuelve la ruta final de best.pt.
    """
    modelo = YOLO(modelo_base)
    resultados = modelo.train(
        data=str(datos),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=patience,
        lr0=lr0,
        name="billco",
    )
    origen_best = Path(resultados.save_dir) / "weights" / "best.pt"
    config.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origen_best, config.RUTA_MODELO)
    return config.RUTA_MODELO


def main() -> None:
    """Punto de entrada CLI."""
    parser = argparse.ArgumentParser(description="Entrena YOLO11 para denominaciones bolivianas.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch", type=int, default=config.BATCH)
    args = parser.parse_args()
    ruta = entrenar(epochs=args.epochs, batch=args.batch)
    print(f"Entrenamiento finalizado. Modelo guardado en: {ruta}")


if __name__ == "__main__":
    main()
