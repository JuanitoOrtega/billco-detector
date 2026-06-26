"""Utilidades para mapear clases detectadas a valores en bolivianos."""
from __future__ import annotations

from collections import Counter

from config import config


def valor_de(clave: str) -> float:
    """Devuelve el valor en Bs de una clave de denominación."""
    return config.VALORES_BS[clave]


def nombre_de(clave: str) -> str:
    """Devuelve el nombre legible de una clave (ej. 'Bs 0.50')."""
    return config.NOMBRES_MOSTRAR[clave]


def clave_de_id(id_clase: int) -> str:
    """Devuelve la clave de denominación a partir del id de clase YOLO."""
    return config.CLASES[id_clase]


def monto_total(claves: list[str]) -> float:
    """Suma los valores en Bs de una lista de claves detectadas."""
    return sum(valor_de(c) for c in claves)


def resumen_conteo(claves: list[str]) -> dict[str, int]:
    """Devuelve el conteo de ocurrencias por clave."""
    return dict(Counter(claves))
