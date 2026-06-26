from pathlib import Path

from src.dataset import verificar_dataset as v


def test_linea_valida_correcta():
    assert v.linea_valida("0 0.5 0.5 0.2 0.3", num_clases=9) is True


def test_linea_valida_clase_fuera_de_rango():
    assert v.linea_valida("9 0.5 0.5 0.2 0.3", num_clases=9) is False


def test_linea_valida_coordenada_fuera_de_rango():
    assert v.linea_valida("0 1.5 0.5 0.2 0.3", num_clases=9) is False


def test_linea_valida_campos_incompletos():
    assert v.linea_valida("0 0.5 0.5", num_clases=9) is False


def test_verificar_etiqueta_archivo_correcto(tmp_path):
    txt = tmp_path / "img1.txt"
    txt.write_text("0 0.5 0.5 0.2 0.3\n3 0.1 0.1 0.1 0.1\n")
    assert v.verificar_etiqueta(txt, num_clases=9) == []


def test_verificar_etiqueta_detecta_error(tmp_path):
    txt = tmp_path / "img2.txt"
    txt.write_text("0 0.5 0.5 0.2 0.3\n99 0.1 0.1 0.1 0.1\n")
    errores = v.verificar_etiqueta(txt, num_clases=9)
    assert len(errores) == 1
