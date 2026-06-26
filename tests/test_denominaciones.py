import pytest

from src.utils import denominaciones as d


def test_valor_de_moneda():
    assert d.valor_de("moneda_050") == 0.50
    assert d.valor_de("billete_200") == 200.0


def test_valor_de_clave_invalida():
    with pytest.raises(KeyError):
        d.valor_de("inexistente")


def test_clave_de_id():
    assert d.clave_de_id(0) == "moneda_050"
    assert d.clave_de_id(8) == "billete_200"


def test_nombre_de():
    assert d.nombre_de("billete_10") == "Bs 10"


def test_monto_total():
    claves = ["moneda_050", "moneda_1", "billete_10"]
    assert d.monto_total(claves) == pytest.approx(11.50)


def test_monto_total_vacio():
    assert d.monto_total([]) == 0.0


def test_resumen_conteo():
    claves = ["moneda_1", "moneda_1", "billete_50"]
    assert d.resumen_conteo(claves) == {"moneda_1": 2, "billete_50": 1}
