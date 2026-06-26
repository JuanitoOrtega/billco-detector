"""Extrae las notas de una versión desde CHANGELOG.md.

Uso:  python -m scripts.notas_release <version> [ruta_changelog]

Imprime en stdout el cuerpo de la sección de esa versión, listo para usarse
como notas de un release. Sale con código 1 si la versión no está en el
changelog (así el flujo de CI obliga a documentar cada versión).
"""
from __future__ import annotations

import sys
from pathlib import Path


def extraer_notas(version: str, texto: str) -> str:
    """Devuelve el cuerpo de la sección ``## [<version>]`` del changelog.

    Es el texto entre el encabezado de esa versión y el de la siguiente
    (o el final del archivo), sin el encabezado y sin espacios sobrantes.
    Devuelve cadena vacía si la versión no aparece.
    """
    lineas = texto.splitlines()
    inicio: int | None = None
    for indice, linea in enumerate(lineas):
        if linea.startswith(f"## [{version}]"):
            inicio = indice + 1
            break
    if inicio is None:
        return ""

    cuerpo: list[str] = []
    for linea in lineas[inicio:]:
        if linea.startswith("## "):
            break
        cuerpo.append(linea)
    return "\n".join(cuerpo).strip()


def main() -> None:
    """Punto de entrada CLI."""
    if len(sys.argv) < 2:
        print(
            "Uso: python -m scripts.notas_release <version> [changelog]",
            file=sys.stderr,
        )
        raise SystemExit(2)

    version = sys.argv[1]
    ruta = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("CHANGELOG.md")
    notas = extraer_notas(version, ruta.read_text(encoding="utf-8"))
    if not notas:
        print(
            f"No se encontró la sección de la versión {version} en {ruta}.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(notas)


if __name__ == "__main__":
    main()
