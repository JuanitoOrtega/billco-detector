"""División de un dataset plano en train/val/test (formato YOLO).

Espera un directorio de origen con dos subcarpetas: 'images' y 'labels',
con archivos emparejados por nombre (img1.jpg <-> img1.txt).

Ejecutable como CLI:  python -m src.dataset.dividir_dataset <origen> [destino]
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from config import config

EXTENSIONES_IMG = (".jpg", ".jpeg", ".png")


def _emparejar(dir_origen: Path) -> list[tuple[Path, Path]]:
    """Devuelve pares (imagen, etiqueta) que existen en origen/images y origen/labels."""
    dir_img = dir_origen / "images"
    dir_lbl = dir_origen / "labels"
    pares: list[tuple[Path, Path]] = []
    for img in sorted(dir_img.iterdir()):
        if img.suffix.lower() not in EXTENSIONES_IMG:
            continue
        txt = dir_lbl / f"{img.stem}.txt"
        if txt.exists():
            pares.append((img, txt))
    return pares


def dividir(
    dir_origen: Path,
    dir_destino: Path,
    prop: tuple[float, float, float] = (0.7, 0.2, 0.1),
    semilla: int = 42,
) -> dict[str, int]:
    """Copia los pares imagen/etiqueta a train/val/test según `prop`.

    Devuelve el conteo de ejemplos por split.
    """
    pares = _emparejar(dir_origen)
    random.Random(semilla).shuffle(pares)
    n = len(pares)
    n_train = int(n * prop[0])
    n_val = int(n * prop[1])
    reparto = {
        "train": pares[:n_train],
        "val": pares[n_train:n_train + n_val],
        "test": pares[n_train + n_val:],
    }
    conteo: dict[str, int] = {}
    for split, items in reparto.items():
        (dir_destino / split / "images").mkdir(parents=True, exist_ok=True)
        (dir_destino / split / "labels").mkdir(parents=True, exist_ok=True)
        for img, txt in items:
            shutil.copy2(img, dir_destino / split / "images" / img.name)
            shutil.copy2(txt, dir_destino / split / "labels" / txt.name)
        conteo[split] = len(items)
    return conteo


def main() -> None:
    """Punto de entrada CLI."""
    parser = argparse.ArgumentParser(description="Divide un dataset YOLO en train/val/test.")
    parser.add_argument("origen", type=Path, help="Directorio con images/ y labels/")
    parser.add_argument("destino", type=Path, nargs="?", default=config.DIR_DATASET)
    args = parser.parse_args()
    conteo = dividir(args.origen, args.destino)
    print("División completada:", conteo)


if __name__ == "__main__":
    main()
