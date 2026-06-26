"""Tests del extractor de notas de release desde el CHANGELOG."""
from scripts.notas_release import extraer_notas

CHANGELOG = """# Changelog

## [1.2.0] - 2026-07-01
### Añadido
- Función nueva.

## [1.1.0] - 2026-06-26
### Añadido
- Botón "Acerca de".
### Cambiado
- Título con versión.

## [1.0.0] - 2026-06-26
### Añadido
- Primera versión.
"""


def test_extrae_cuerpo_de_version_intermedia():
    notas = extraer_notas("1.1.0", CHANGELOG)
    assert "Botón" in notas
    assert "Título con versión" in notas
    # No incluye el encabezado de la versión ni el de la siguiente.
    assert "## [1.1.0]" not in notas
    assert "1.0.0" not in notas


def test_extrae_ultima_version_hasta_el_final():
    notas = extraer_notas("1.0.0", CHANGELOG)
    assert "Primera versión." in notas
    # No se cuela ningún encabezado de versión.
    assert "## [" not in notas


def test_version_inexistente_devuelve_vacio():
    assert extraer_notas("9.9.9", CHANGELOG) == ""
