"""Entrenamiento del modelo YOLO11 para denominaciones bolivianas.

Ejecutable como CLI:  python -m src.entrenamiento.entrenar
Requiere: el dataset construido (carpetas dataset/train|val/images con
imágenes) y las dependencias de Ultralytics instaladas.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from config import config

# Divisiones que Ultralytics necesita para entrenar (test es opcional aquí).
DIVISIONES_REQUERIDAS: tuple[str, ...] = ("train", "val")
EXTENSIONES_IMG: tuple[str, ...] = (".jpg", ".jpeg", ".png")


def _dataset_incompleto(raiz_dataset: Path) -> list[str]:
    """Devuelve los problemas que impiden entrenar (lista vacía si todo está OK).

    Verifica que cada división requerida tenga una carpeta ``images`` con al
    menos una imagen.
    """
    problemas: list[str] = []
    for division in DIVISIONES_REQUERIDAS:
        dir_img = raiz_dataset / division / "images"
        if not dir_img.is_dir():
            problemas.append(f"Falta la carpeta '{dir_img}'.")
            continue
        hay_imagenes = any(
            p.suffix.lower() in EXTENSIONES_IMG for p in dir_img.iterdir()
        )
        if not hay_imagenes:
            problemas.append(f"La carpeta '{dir_img}' no contiene imágenes.")
    return problemas


def _mensaje_dataset_faltante(problemas: list[str]) -> str:
    """Construye un mensaje claro en español explicando cómo armar el dataset."""
    detalle = "\n".join(f"  - {p}" for p in problemas)
    return (
        "No se puede entrenar: el dataset todavía no está listo.\n\n"
        f"{detalle}\n\n"
        "Para construirlo:\n"
        "  1. Reúne tus fotos etiquetadas (formato YOLO) en una carpeta con\n"
        "     subcarpetas 'images/' y 'labels/'.\n"
        "  2. Divide el dataset:   python -m src.dataset.dividir_dataset <origen>\n"
        "  3. Verifica etiquetas:  python -m src.dataset.verificar_dataset\n"
        "  4. Vuelve a entrenar:   python -m src.entrenamiento.entrenar"
    )


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

    Antes de entrenar verifica que el dataset exista; si falta, lanza
    ``FileNotFoundError`` con instrucciones claras en lugar del error críptico
    de Ultralytics.

    Devuelve la ruta final de best.pt.
    """
    problemas = _dataset_incompleto(config.DIR_DATASET)
    if problemas:
        raise FileNotFoundError(_mensaje_dataset_faltante(problemas))

    from ultralytics import YOLO  # import perezoso: módulo testeable sin la librería

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
    try:
        ruta = entrenar(epochs=args.epochs, batch=args.batch)
    except FileNotFoundError as error:
        # Mensaje claro en español, sin traceback.
        print(error, file=sys.stderr)
        raise SystemExit(1)
    print(f"Entrenamiento finalizado. Modelo guardado en: {ruta}")


if __name__ == "__main__":
    main()
