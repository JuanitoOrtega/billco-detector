"""Tests de la verificación previa del dataset antes de entrenar."""
from pathlib import Path

from src.entrenamiento import entrenar as e


def _crear_division(raiz: Path, division: str, con_imagen: bool) -> None:
    """Crea raiz/<division>/images, opcionalmente con una imagen dentro."""
    dir_img = raiz / division / "images"
    dir_img.mkdir(parents=True)
    if con_imagen:
        (dir_img / "ejemplo.jpg").write_bytes(b"x")


def test_dataset_incompleto_sin_carpetas(tmp_path):
    # Ni train/images ni val/images existen -> dos problemas.
    assert len(e._dataset_incompleto(tmp_path)) == 2


def test_dataset_incompleto_carpetas_vacias(tmp_path):
    _crear_division(tmp_path, "train", con_imagen=False)
    _crear_division(tmp_path, "val", con_imagen=False)
    # Existen pero sin imágenes -> dos problemas.
    assert len(e._dataset_incompleto(tmp_path)) == 2


def test_dataset_incompleto_falta_val(tmp_path):
    _crear_division(tmp_path, "train", con_imagen=True)
    # Falta val -> un problema.
    assert len(e._dataset_incompleto(tmp_path)) == 1


def test_dataset_completo(tmp_path):
    _crear_division(tmp_path, "train", con_imagen=True)
    _crear_division(tmp_path, "val", con_imagen=True)
    assert e._dataset_incompleto(tmp_path) == []
