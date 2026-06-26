"""Evaluación del modelo entrenado sobre el conjunto de test.

Ejecutable como CLI:  python -m src.entrenamiento.evaluar
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from config import config


def evaluar(ruta_modelo: Path = config.RUTA_MODELO) -> dict[str, float]:
    """Evalúa el modelo sobre el split de test y devuelve métricas clave."""
    modelo = YOLO(str(ruta_modelo))
    metricas = modelo.val(data=str(config.RUTA_DATASET_YAML), split="test")
    return {
        "precision": float(metricas.box.mp),
        "recall": float(metricas.box.mr),
        "mAP50": float(metricas.box.map50),
        "mAP50_95": float(metricas.box.map),
    }


def main() -> None:
    """Punto de entrada CLI."""
    parser = argparse.ArgumentParser(description="Evalúa el modelo sobre test.")
    parser.add_argument("--modelo", type=Path, default=config.RUTA_MODELO)
    args = parser.parse_args()
    m = evaluar(args.modelo)
    print("Métricas sobre test:")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
