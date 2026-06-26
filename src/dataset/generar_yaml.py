"""Generación del archivo dataset.yaml para entrenamiento YOLO.

Ejecutable como CLI:  python -m src.dataset.generar_yaml
"""
from __future__ import annotations

from pathlib import Path

import yaml

from config import config


def generar(ruta_yaml: Path = config.RUTA_DATASET_YAML) -> Path:
    """Escribe dataset.yaml con rutas y clases, y devuelve la ruta escrita."""
    contenido = {
        "path": str(config.DIR_DATASET),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(config.CLASES),
        "names": {i: nombre for i, nombre in enumerate(config.CLASES)},
    }
    ruta_yaml.write_text(yaml.safe_dump(contenido, sort_keys=False, allow_unicode=True))
    return ruta_yaml


def main() -> None:
    """Punto de entrada CLI."""
    ruta = generar()
    print(f"dataset.yaml generado en: {ruta}")


if __name__ == "__main__":
    main()
