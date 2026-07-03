"""Configuración central del proyecto Billco Detector.

Única fuente de verdad para clases, valores monetarios, rutas y parámetros.
"""
from __future__ import annotations

from pathlib import Path

# --- Versión del proyecto (versionado semántico: MAYOR.MENOR.PARCHE) ---
VERSION: str = "1.1.3"
AUTOR: str = "Juanito Ortega"

# --- Clases del modelo (orden fijo: el índice es el id de clase YOLO) ---
CLASES: list[str] = [
    "moneda_050",
    "moneda_1",
    "moneda_2",
    "moneda_5",
    "billete_10",
    "billete_20",
    "billete_50",
    "billete_100",
    "billete_200",
]

# --- Valor monetario en bolivianos de cada clase ---
VALORES_BS: dict[str, float] = {
    "moneda_050": 0.50,
    "moneda_1": 1.0,
    "moneda_2": 2.0,
    "moneda_5": 5.0,
    "billete_10": 10.0,
    "billete_20": 20.0,
    "billete_50": 50.0,
    "billete_100": 100.0,
    "billete_200": 200.0,
}

# --- Nombre legible para mostrar en la interfaz ---
NOMBRES_MOSTRAR: dict[str, str] = {
    "moneda_050": "Bs 0.50",
    "moneda_1": "Bs 1",
    "moneda_2": "Bs 2",
    "moneda_5": "Bs 5",
    "billete_10": "Bs 10",
    "billete_20": "Bs 20",
    "billete_50": "Bs 50",
    "billete_100": "Bs 100",
    "billete_200": "Bs 200",
}

# --- Rutas del proyecto ---
RAIZ: Path = Path(__file__).resolve().parent.parent
DIR_DATASET: Path = RAIZ / "dataset"
DIR_MODELOS: Path = RAIZ / "modelos"
RUTA_MODELO: Path = DIR_MODELOS / "best.pt"
DIR_CAPTURAS: Path = RAIZ / "capturas"
RUTA_DATASET_YAML: Path = RAIZ / "dataset.yaml"
RUTA_ICONO: Path = RAIZ / "assets" / "icono.png"

# --- Parámetros de entrenamiento ---
MODELO_BASE: str = "yolo11n.pt"
IMGSZ: int = 640
EPOCHS: int = 100
BATCH: int = 16
PATIENCE: int = 20
LR0: float = 0.01

# --- Parámetros de inferencia ---
CONF_MIN: float = 0.5
