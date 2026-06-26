"""Verificación del formato YOLO de un dataset de etiquetas.

Ejecutable como CLI:  python -m src.dataset.verificar_dataset
"""
from __future__ import annotations

from pathlib import Path

from config import config


def linea_valida(linea: str, num_clases: int) -> bool:
    """Valida una línea de anotación YOLO: '<clase> <x> <y> <w> <h>'.

    La clase debe estar en [0, num_clases) y las 4 coordenadas en [0, 1].
    """
    partes = linea.split()
    if len(partes) != 5:
        return False
    try:
        id_clase = int(partes[0])
        coords = [float(p) for p in partes[1:]]
    except ValueError:
        return False
    if not 0 <= id_clase < num_clases:
        return False
    return all(0.0 <= c <= 1.0 for c in coords)


def verificar_etiqueta(ruta_txt: Path, num_clases: int) -> list[str]:
    """Devuelve la lista de errores encontrados en un archivo de etiqueta."""
    errores: list[str] = []
    for n, linea in enumerate(ruta_txt.read_text().splitlines(), start=1):
        if not linea.strip():
            continue
        if not linea_valida(linea, num_clases):
            errores.append(f"{ruta_txt.name}:{n} -> línea inválida: '{linea}'")
    return errores


def verificar_directorio(dir_etiquetas: Path, num_clases: int) -> list[str]:
    """Verifica todas las etiquetas .txt de un directorio."""
    errores: list[str] = []
    for txt in sorted(dir_etiquetas.glob("*.txt")):
        errores.extend(verificar_etiqueta(txt, num_clases))
    return errores


def main() -> None:
    """Punto de entrada CLI: verifica las etiquetas del dataset configurado."""
    num_clases = len(config.CLASES)
    total_errores: list[str] = []
    for split in ("train", "val", "test"):
        dir_labels = config.DIR_DATASET / split / "labels"
        if not dir_labels.exists():
            print(f"[aviso] no existe {dir_labels}")
            continue
        errores = verificar_directorio(dir_labels, num_clases)
        print(f"{split}: {len(errores)} error(es)")
        total_errores.extend(errores)
    for e in total_errores:
        print("  -", e)
    print(f"\nTotal de errores: {len(total_errores)}")


if __name__ == "__main__":
    main()
