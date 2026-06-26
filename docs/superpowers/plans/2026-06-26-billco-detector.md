# Billco Detector — Plan de Implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir el proyecto completo de detección de monedas/billetes bolivianos con YOLO11 + PyQt6, en tres módulos documentados (dataset, entrenamiento, app de escritorio).

**Architecture:** Configuración centralizada en `config/`. Lógica pura (valores de denominaciones, validación de dataset) probada con tests. Scripts CLI para dataset y entrenamiento sobre Ultralytics YOLO11. App PyQt6 con captura de cámara en hilo (`QThread`) y un wrapper de detección; la app abre aunque no exista `best.pt`.

**Tech Stack:** Python 3.10+, Ultralytics (YOLO11), PyQt6, OpenCV (`opencv-python`), NumPy, PyYAML, pytest.

## Global Constraints

- Idioma: documentación, comentarios, docstrings e interfaz en **español**.
- Calidad: PEP8, clases, type hints, docstrings, sin duplicación.
- Configuración única en `config/config.py` (rutas, clases, valores, parámetros).
- 9 clases en orden fijo: `moneda_050, moneda_1, moneda_2, moneda_5, billete_10, billete_20, billete_50, billete_100, billete_200`.
- La app debe abrir y funcionar (cámara) aunque no exista `best.pt`.
- Modelo base de entrenamiento: `yolo11n`. Imagen `imgsz=640`.

---

## Estructura de archivos

```
billco-detector/
├── README.md                      # Tarea 7
├── requirements.txt               # Tarea 1
├── .gitignore                     # Tarea 1
├── dataset.yaml                   # generado en runtime por Tarea 3
├── main.py                        # Tarea 6
├── docs/
│   └── documento_tecnico.md       # Tarea 7
├── config/
│   ├── __init__.py                # Tarea 1
│   └── config.py                  # Tarea 1
├── src/
│   ├── __init__.py                # Tarea 1
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── dividir_dataset.py     # Tarea 3
│   │   ├── generar_yaml.py        # Tarea 3
│   │   └── verificar_dataset.py   # Tarea 3
│   ├── entrenamiento/
│   │   ├── __init__.py
│   │   ├── entrenar.py            # Tarea 4
│   │   └── evaluar.py             # Tarea 4
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── detector.py            # Tarea 5
│   │   └── camara.py              # Tarea 5
│   ├── app/
│   │   ├── __init__.py
│   │   ├── widgets.py             # Tarea 6
│   │   └── ventana_principal.py   # Tarea 6
│   └── utils/
│       ├── __init__.py
│       └── denominaciones.py      # Tarea 2
└── tests/
    ├── __init__.py
    ├── test_denominaciones.py     # Tarea 2
    └── test_verificar_dataset.py  # Tarea 3
```

---

### Task 1: Configuración base y esqueleto del proyecto

**Files:**
- Create: `requirements.txt`, `.gitignore`
- Create: `config/__init__.py`, `config/config.py`
- Create: `src/__init__.py`, `src/dataset/__init__.py`, `src/entrenamiento/__init__.py`, `src/vision/__init__.py`, `src/app/__init__.py`, `src/utils/__init__.py`, `tests/__init__.py`

**Interfaces:**
- Produces (`config/config.py`):
  - `CLASES: list[str]` — 9 claves en orden fijo.
  - `VALORES_BS: dict[str, float]` — clave → valor en Bs.
  - `NOMBRES_MOSTRAR: dict[str, str]` — clave → etiqueta legible (ej. `"Bs 0.50"`).
  - `RAIZ: Path`, `DIR_DATASET: Path`, `DIR_MODELOS: Path`, `RUTA_MODELO: Path` (`DIR_MODELOS / "best.pt"`).
  - `IMGSZ: int = 640`, `EPOCHS: int = 100`, `BATCH: int = 16`, `PATIENCE: int = 20`, `LR0: float = 0.01`, `CONF_MIN: float = 0.5`.
  - `MODELO_BASE: str = "yolo11n.pt"`.

- [ ] **Step 1: Crear `requirements.txt`**

```
ultralytics>=8.3.0
PyQt6>=6.6.0
opencv-python>=4.9.0
numpy>=1.26.0
PyYAML>=6.0
pytest>=8.0.0
```

- [ ] **Step 2: Crear `.gitignore`**

```
__pycache__/
*.py[cod]
.venv/
venv/
*.pt
runs/
dataset/
*.jpg
*.jpeg
*.png
capturas/
.DS_Store
```

- [ ] **Step 3: Crear los `__init__.py` vacíos**

Crear archivos vacíos: `config/__init__.py`, `src/__init__.py`, `src/dataset/__init__.py`, `src/entrenamiento/__init__.py`, `src/vision/__init__.py`, `src/app/__init__.py`, `src/utils/__init__.py`, `tests/__init__.py`.

- [ ] **Step 4: Crear `config/config.py`**

```python
"""Configuración central del proyecto Billco Detector.

Única fuente de verdad para clases, valores monetarios, rutas y parámetros.
"""
from __future__ import annotations

from pathlib import Path

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

# --- Parámetros de entrenamiento ---
MODELO_BASE: str = "yolo11n.pt"
IMGSZ: int = 640
EPOCHS: int = 100
BATCH: int = 16
PATIENCE: int = 20
LR0: float = 0.01

# --- Parámetros de inferencia ---
CONF_MIN: float = 0.5
```

- [ ] **Step 5: Verificar que la config importa y es coherente**

Run: `python -c "from config import config as c; assert len(c.CLASES)==9; assert set(c.CLASES)==set(c.VALORES_BS); assert set(c.CLASES)==set(c.NOMBRES_MOSTRAR); print('config OK')"`
Expected: imprime `config OK`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore config src tests
git commit -m "chore: esqueleto del proyecto y configuración central"
```

---

### Task 2: Utilidades de denominaciones (TDD)

**Files:**
- Create: `src/utils/denominaciones.py`
- Test: `tests/test_denominaciones.py`

**Interfaces:**
- Consumes: `config.config.CLASES`, `VALORES_BS`, `NOMBRES_MOSTRAR`.
- Produces:
  - `valor_de(clave: str) -> float` — valor en Bs de una clave; `KeyError` si no existe.
  - `nombre_de(clave: str) -> str` — nombre legible.
  - `clave_de_id(id_clase: int) -> str` — clave a partir del id YOLO.
  - `monto_total(claves: list[str]) -> float` — suma de valores de una lista de claves detectadas.
  - `resumen_conteo(claves: list[str]) -> dict[str, int]` — conteo por clave.

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/test_denominaciones.py
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
```

- [ ] **Step 2: Ejecutar y verificar que fallan**

Run: `python -m pytest tests/test_denominaciones.py -v`
Expected: FAIL con `ModuleNotFoundError` o `AttributeError` (módulo aún no existe)

- [ ] **Step 3: Implementar `src/utils/denominaciones.py`**

```python
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
```

- [ ] **Step 4: Ejecutar y verificar que pasan**

Run: `python -m pytest tests/test_denominaciones.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/utils/denominaciones.py tests/test_denominaciones.py
git commit -m "feat: utilidades de denominaciones con cálculo de monto total"
```

---

### Task 3: Módulo dataset — verificación (TDD), división y generación de YAML

**Files:**
- Create: `src/dataset/verificar_dataset.py`, `src/dataset/dividir_dataset.py`, `src/dataset/generar_yaml.py`
- Test: `tests/test_verificar_dataset.py`

**Interfaces:**
- Consumes: `config.config.CLASES`, `DIR_DATASET`, `RUTA_DATASET_YAML`.
- Produces:
  - `verificar_dataset.linea_valida(linea: str, num_clases: int) -> bool` — valida una línea YOLO.
  - `verificar_dataset.verificar_etiqueta(ruta_txt: Path, num_clases: int) -> list[str]` — lista de errores (vacía si OK).
  - `dividir_dataset.dividir(dir_origen: Path, dir_destino: Path, prop=(0.7,0.2,0.1), semilla=42) -> dict[str,int]` — copia pares imagen/etiqueta a `train/val/test`; devuelve conteo por split.
  - `generar_yaml.generar(ruta_yaml: Path = RUTA_DATASET_YAML) -> Path` — escribe `dataset.yaml`.

- [ ] **Step 1: Escribir tests de verificación que fallan**

```python
# tests/test_verificar_dataset.py
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
```

- [ ] **Step 2: Ejecutar y verificar que fallan**

Run: `python -m pytest tests/test_verificar_dataset.py -v`
Expected: FAIL (módulo no existe)

- [ ] **Step 3: Implementar `src/dataset/verificar_dataset.py`**

```python
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
```

- [ ] **Step 4: Ejecutar y verificar que pasan**

Run: `python -m pytest tests/test_verificar_dataset.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Implementar `src/dataset/dividir_dataset.py`**

```python
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
```

- [ ] **Step 6: Implementar `src/dataset/generar_yaml.py`**

```python
"""Generación del archivo dataset.yaml para entrenamiento YOLO.

Ejecutable como CLI:  python -m src.dataset.generar_yaml
"""
from __future__ import annotations

from pathlib import Path

import yaml

from config import config


def generar(ruta_yaml: Path = config.RUTA_DATASET_YAML) -> Path:
    """Escribe dataset.yaml con rutas y clases, y devuelve la ruta escrita."""
    contenido = {
        "path": str(config.DIR_DATASET),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(config.CLASES),
        "names": {i: nombre for i, nombre in enumerate(config.CLASES)},
    }
    ruta_yaml.write_text(yaml.safe_dump(contenido, sort_keys=False, allow_unicode=True))
    return ruta_yaml


def main() -> None:
    """Punto de entrada CLI."""
    ruta = generar()
    print(f"dataset.yaml generado en: {ruta}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Verificar generación de YAML**

Run: `python -m src.dataset.generar_yaml && python -c "import yaml; d=yaml.safe_load(open('dataset.yaml')); assert d['nc']==9 and len(d['names'])==9; print('yaml OK')"`
Expected: imprime ruta y `yaml OK`

- [ ] **Step 8: Commit**

```bash
git add src/dataset tests/test_verificar_dataset.py dataset.yaml
git commit -m "feat: módulo dataset (verificación, división y generación de yaml)"
```

---

### Task 4: Módulo entrenamiento — entrenar y evaluar

**Files:**
- Create: `src/entrenamiento/entrenar.py`, `src/entrenamiento/evaluar.py`

**Interfaces:**
- Consumes: `config.config` (`MODELO_BASE`, `RUTA_DATASET_YAML`, `EPOCHS`, `BATCH`, `IMGSZ`, `PATIENCE`, `LR0`, `DIR_MODELOS`).
- Produces:
  - `entrenar.entrenar(...) -> Path` — entrena y devuelve la ruta a `best.pt`.
  - `evaluar.evaluar(ruta_modelo: Path) -> dict[str, float]` — métricas sobre el split test.

**Nota:** estos scripts requieren Ultralytics instalado y un dataset real para ejecutarse de verdad; no se prueban con pytest. La verificación es que importan sin error.

- [ ] **Step 1: Implementar `src/entrenamiento/entrenar.py`**

```python
"""Entrenamiento del modelo YOLO11 para denominaciones bolivianas.

Ejecutable como CLI:  python -m src.entrenamiento.entrenar
Requiere: dataset.yaml válido y dependencias de Ultralytics instaladas.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from config import config


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

    Devuelve la ruta final de best.pt.
    """
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
    ruta = entrenar(epochs=args.epochs, batch=args.batch)
    print(f"Entrenamiento finalizado. Modelo guardado en: {ruta}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Implementar `src/entrenamiento/evaluar.py`**

```python
"""Evaluación del modelo entrenado sobre el conjunto de test.

Ejecutable como CLI:  python -m src.entrenamiento.evaluar
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from config import config


def evaluar(ruta_modelo: Path = config.RUTA_MODELO) -> dict[str, float]:
    """Evalúa el modelo sobre el split de test y devuelve métricas clave."""
    modelo = YOLO(str(ruta_modelo))
    metricas = modelo.val(data=str(config.RUTA_DATASET_YAML), split="test")
    return {
        "precision": float(metricas.box.mp),
        "recall": float(metricas.box.mr),
        "mAP50": float(metricas.box.map50),
        "mAP50_95": float(metricas.box.map),
    }


def main() -> None:
    """Punto de entrada CLI."""
    parser = argparse.ArgumentParser(description="Evalúa el modelo sobre test.")
    parser.add_argument("--modelo", type=Path, default=config.RUTA_MODELO)
    args = parser.parse_args()
    m = evaluar(args.modelo)
    print("Métricas sobre test:")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verificar que los módulos importan**

Run: `python -c "import ast; ast.parse(open('src/entrenamiento/entrenar.py').read()); ast.parse(open('src/entrenamiento/evaluar.py').read()); print('sintaxis OK')"`
Expected: imprime `sintaxis OK` (no requiere Ultralytics instalado para esta verificación de sintaxis)

- [ ] **Step 4: Commit**

```bash
git add src/entrenamiento
git commit -m "feat: módulo de entrenamiento y evaluación con YOLO11"
```

---

### Task 5: Núcleo de visión — detector y cámara

**Files:**
- Create: `src/vision/detector.py`, `src/vision/camara.py`

**Interfaces:**
- Consumes: `config.config` (`RUTA_MODELO`, `CONF_MIN`, `CLASES`), `src.utils.denominaciones`.
- Produces:
  - `detector.Deteccion` — dataclass: `clave: str`, `confianza: float`, `caja: tuple[int,int,int,int]`.
  - `detector.Detector` — clase con:
    - `__init__(self, ruta_modelo: Path = RUTA_MODELO)`; carga si el archivo existe.
    - `cargado: bool` — propiedad de estado.
    - `cargar(self, ruta: Path) -> bool` — carga un modelo manualmente.
    - `detectar(self, frame) -> list[Deteccion]`.
    - `dibujar(self, frame, detecciones) -> frame` — dibuja cajas + etiqueta + confianza.
  - `camara.HiloCamara(QThread)` — emite señal `frame_listo(np.ndarray)`; métodos `run()`, `detener()`.

- [ ] **Step 1: Implementar `src/vision/detector.py`**

```python
"""Wrapper de inferencia YOLO para detección de denominaciones."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import config
from src.utils import denominaciones


@dataclass
class Deteccion:
    """Una detección individual en un frame."""

    clave: str
    confianza: float
    caja: tuple[int, int, int, int]  # x1, y1, x2, y2


class Detector:
    """Carga un modelo YOLO y ejecuta inferencia sobre frames de imagen."""

    def __init__(self, ruta_modelo: Path = config.RUTA_MODELO) -> None:
        self._modelo = None
        self.ruta_modelo: Path | None = None
        if ruta_modelo.exists():
            self.cargar(ruta_modelo)

    @property
    def cargado(self) -> bool:
        """Indica si hay un modelo cargado y listo para inferir."""
        return self._modelo is not None

    def cargar(self, ruta: Path) -> bool:
        """Carga un modelo desde la ruta dada. Devuelve True si tuvo éxito."""
        from ultralytics import YOLO  # import perezoso: la app abre sin Ultralytics

        if not ruta.exists():
            return False
        self._modelo = YOLO(str(ruta))
        self.ruta_modelo = ruta
        return True

    def detectar(self, frame: np.ndarray) -> list[Deteccion]:
        """Ejecuta inferencia sobre un frame BGR y devuelve las detecciones."""
        if not self.cargado:
            return []
        resultados = self._modelo(frame, conf=config.CONF_MIN, verbose=False)[0]
        detecciones: list[Deteccion] = []
        for caja in resultados.boxes:
            id_clase = int(caja.cls[0])
            x1, y1, x2, y2 = (int(v) for v in caja.xyxy[0])
            detecciones.append(
                Deteccion(
                    clave=denominaciones.clave_de_id(id_clase),
                    confianza=float(caja.conf[0]),
                    caja=(x1, y1, x2, y2),
                )
            )
        return detecciones

    def dibujar(self, frame: np.ndarray, detecciones: list[Deteccion]) -> np.ndarray:
        """Dibuja las cajas, el nombre de la denominación y la confianza."""
        for det in detecciones:
            x1, y1, x2, y2 = det.caja
            etiqueta = f"{denominaciones.nombre_de(det.clave)} {det.confianza:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(
                frame, etiqueta, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2,
            )
        return frame
```

- [ ] **Step 2: Verificación de inferencia sin modelo (degradación elegante)**

Run: `python -c "import numpy as np; from src.vision.detector import Detector; d=Detector(); print('cargado:', d.cargado); print('dets:', d.detectar(np.zeros((10,10,3),dtype='uint8')))"`
Expected: `cargado: False` y `dets: []` (no falla aunque no haya modelo)

- [ ] **Step 3: Implementar `src/vision/camara.py`**

```python
"""Captura de cámara en un hilo separado para no congelar la interfaz."""
from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class HiloCamara(QThread):
    """Lee frames de la webcam en segundo plano y los emite por señal."""

    frame_listo = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, indice_camara: int = 0) -> None:
        super().__init__()
        self._indice = indice_camara
        self._corriendo = False

    def run(self) -> None:
        """Bucle de captura; emite cada frame leído."""
        captura = cv2.VideoCapture(self._indice)
        if not captura.isOpened():
            self.error.emit("No se pudo abrir la cámara.")
            return
        self._corriendo = True
        while self._corriendo:
            ok, frame = captura.read()
            if not ok:
                self.error.emit("Fallo al leer un frame de la cámara.")
                break
            self.frame_listo.emit(frame)
        captura.release()

    def detener(self) -> None:
        """Detiene el bucle de captura y espera a que el hilo termine."""
        self._corriendo = False
        self.wait()
```

- [ ] **Step 4: Verificar sintaxis del módulo de cámara**

Run: `python -c "import ast; ast.parse(open('src/vision/camara.py').read()); print('camara sintaxis OK')"`
Expected: imprime `camara sintaxis OK`

- [ ] **Step 5: Commit**

```bash
git add src/vision
git commit -m "feat: detector YOLO y captura de cámara en hilo"
```

---

### Task 6: Aplicación PyQt6

**Files:**
- Create: `src/app/widgets.py`, `src/app/ventana_principal.py`, `main.py`

**Interfaces:**
- Consumes: `src.vision.detector.Detector`, `src.vision.camara.HiloCamara`, `src.utils.denominaciones`, `config.config`.
- Produces:
  - `widgets.PanelDetecciones(QWidget)` — muestra lista, monto total, conteo, FPS, estado modelo. Método `actualizar(detecciones, fps, modelo_cargado)`.
  - `ventana_principal.VentanaPrincipal(QMainWindow)` — orquesta cámara + detector + panel + botones.
  - `main.py` — crea `QApplication`, muestra la ventana.

- [ ] **Step 1: Implementar `src/app/widgets.py`**

```python
"""Componentes reutilizables de la interfaz."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGroupBox, QLabel, QListWidget, QVBoxLayout, QWidget

from src.utils import denominaciones
from src.vision.detector import Deteccion


class PanelDetecciones(QWidget):
    """Panel lateral con la información de detección en tiempo real."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        self.lbl_estado = QLabel("Modelo: no cargado")
        self.lbl_estado.setStyleSheet("font-weight: bold; color: #b00;")
        self.lbl_fps = QLabel("FPS: 0.0")
        self.lbl_conteo = QLabel("Objetos detectados: 0")
        self.lbl_total = QLabel("Total: Bs 0.00")
        self.lbl_total.setStyleSheet("font-size: 20px; font-weight: bold; color: #060;")

        grupo = QGroupBox("Detecciones")
        gl = QVBoxLayout(grupo)
        self.lista = QListWidget()
        gl.addWidget(self.lista)

        for w in (self.lbl_estado, self.lbl_fps, self.lbl_conteo, grupo, self.lbl_total):
            layout.addWidget(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    def actualizar(
        self, detecciones: list[Deteccion], fps: float, modelo_cargado: bool
    ) -> None:
        """Refresca todos los indicadores con el estado del frame actual."""
        if modelo_cargado:
            self.lbl_estado.setText("Modelo: cargado")
            self.lbl_estado.setStyleSheet("font-weight: bold; color: #060;")
        else:
            self.lbl_estado.setText("Modelo: no cargado")
            self.lbl_estado.setStyleSheet("font-weight: bold; color: #b00;")

        self.lbl_fps.setText(f"FPS: {fps:.1f}")
        self.lbl_conteo.setText(f"Objetos detectados: {len(detecciones)}")

        claves = [d.clave for d in detecciones]
        self.lbl_total.setText(f"Total: Bs {denominaciones.monto_total(claves):.2f}")

        self.lista.clear()
        for det in detecciones:
            self.lista.addItem(
                f"{denominaciones.nombre_de(det.clave)}  —  {det.confianza:.0%}"
            )
```

- [ ] **Step 2: Implementar `src/app/ventana_principal.py`**

```python
"""Ventana principal de la aplicación Billco Detector."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from config import config
from src.app.widgets import PanelDetecciones
from src.vision.camara import HiloCamara
from src.vision.detector import Detector


class VentanaPrincipal(QMainWindow):
    """Orquesta la cámara, el detector y la interfaz."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Billco Detector — Detección de moneda boliviana")
        self.resize(1100, 700)

        self.detector = Detector()
        self.hilo: HiloCamara | None = None
        self._ultimo_frame: np.ndarray | None = None
        self._t_anterior = time.time()

        self._construir_ui()

    def _construir_ui(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)

        # Vista de cámara
        self.vista = QLabel("Cámara detenida")
        self.vista.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.vista.setMinimumSize(720, 540)
        self.vista.setStyleSheet("background:#111; color:#888;")
        layout.addWidget(self.vista, stretch=3)

        # Panel lateral + botones
        lateral = QVBoxLayout()
        self.panel = PanelDetecciones()
        lateral.addWidget(self.panel)

        self.btn_iniciar = QPushButton("Iniciar cámara")
        self.btn_detener = QPushButton("Detener cámara")
        self.btn_capturar = QPushButton("Capturar imagen")
        self.btn_guardar = QPushButton("Guardar captura")
        self.btn_modelo = QPushButton("Cargar modelo…")
        self.btn_salir = QPushButton("Salir")

        self.btn_iniciar.clicked.connect(self.iniciar_camara)
        self.btn_detener.clicked.connect(self.detener_camara)
        self.btn_capturar.clicked.connect(self.capturar)
        self.btn_guardar.clicked.connect(self.guardar_captura)
        self.btn_modelo.clicked.connect(self.cargar_modelo)
        self.btn_salir.clicked.connect(self.close)

        for b in (
            self.btn_iniciar, self.btn_detener, self.btn_capturar,
            self.btn_guardar, self.btn_modelo, self.btn_salir,
        ):
            lateral.addWidget(b)

        contenedor = QWidget()
        contenedor.setLayout(lateral)
        layout.addWidget(contenedor, stretch=1)

        self.setCentralWidget(central)
        self.panel.actualizar([], 0.0, self.detector.cargado)

    def iniciar_camara(self) -> None:
        """Arranca el hilo de captura."""
        if self.hilo and self.hilo.isRunning():
            return
        self.hilo = HiloCamara()
        self.hilo.frame_listo.connect(self._procesar_frame)
        self.hilo.error.connect(lambda m: QMessageBox.warning(self, "Cámara", m))
        self.hilo.start()

    def detener_camara(self) -> None:
        """Detiene el hilo de captura."""
        if self.hilo:
            self.hilo.detener()
            self.hilo = None
        self.vista.setText("Cámara detenida")

    def _procesar_frame(self, frame: np.ndarray) -> None:
        """Recibe un frame, ejecuta detección, dibuja y actualiza la UI."""
        detecciones = self.detector.detectar(frame)
        frame = self.detector.dibujar(frame, detecciones)
        self._ultimo_frame = frame.copy()

        ahora = time.time()
        fps = 1.0 / max(ahora - self._t_anterior, 1e-6)
        self._t_anterior = ahora

        self.panel.actualizar(detecciones, fps, self.detector.cargado)
        self._mostrar(frame)

    def _mostrar(self, frame: np.ndarray) -> None:
        """Convierte un frame BGR a QPixmap y lo muestra escalado."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        imagen = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(imagen).scaled(
            self.vista.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.vista.setPixmap(pix)

    def capturar(self) -> None:
        """Congela el último frame mostrándolo como captura."""
        if self._ultimo_frame is None:
            QMessageBox.information(self, "Capturar", "No hay frame disponible.")
            return
        self.detener_camara()
        self._mostrar(self._ultimo_frame)

    def guardar_captura(self) -> None:
        """Guarda el último frame en disco."""
        if self._ultimo_frame is None:
            QMessageBox.information(self, "Guardar", "No hay captura para guardar.")
            return
        config.DIR_CAPTURAS.mkdir(parents=True, exist_ok=True)
        sugerida = str(config.DIR_CAPTURAS / "captura.jpg")
        ruta, _ = QFileDialog.getSaveFileName(self, "Guardar captura", sugerida, "Imágenes (*.jpg *.png)")
        if ruta:
            cv2.imwrite(ruta, self._ultimo_frame)
            QMessageBox.information(self, "Guardar", f"Captura guardada en:\n{ruta}")

    def cargar_modelo(self) -> None:
        """Permite seleccionar un best.pt manualmente."""
        ruta, _ = QFileDialog.getOpenFileName(self, "Cargar modelo", str(config.RAIZ), "Modelos (*.pt)")
        if ruta and self.detector.cargar(Path(ruta)):
            QMessageBox.information(self, "Modelo", "Modelo cargado correctamente.")
            self.panel.actualizar([], 0.0, True)

    def closeEvent(self, event) -> None:  # noqa: N802 (firma de Qt)
        """Detiene la cámara al cerrar la ventana."""
        self.detener_camara()
        super().closeEvent(event)
```

- [ ] **Step 3: Implementar `main.py`**

```python
"""Punto de entrada de la aplicación Billco Detector."""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from src.app.ventana_principal import VentanaPrincipal


def main() -> None:
    """Lanza la aplicación de escritorio."""
    app = QApplication(sys.argv)
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verificar sintaxis de los tres archivos**

Run: `python -c "import ast; [ast.parse(open(p).read()) for p in ('src/app/widgets.py','src/app/ventana_principal.py','main.py')]; print('app sintaxis OK')"`
Expected: imprime `app sintaxis OK`

- [ ] **Step 5: Verificación de importación de la app (requiere PyQt6 instalado)**

Run: `python -c "import src.app.ventana_principal; print('import app OK')"`
Expected: imprime `import app OK`. Si PyQt6 no está instalado, documentar en el commit que la verificación se hará tras `pip install -r requirements.txt`.

- [ ] **Step 6: Commit**

```bash
git add src/app main.py
git commit -m "feat: aplicación de escritorio PyQt6 con cámara, detección y monto total"
```

---

### Task 7: Documentación — README y documento técnico

**Files:**
- Create: `README.md`, `docs/documento_tecnico.md`

**Interfaces:** ninguna (documentación).

- [ ] **Step 1: Escribir `README.md`**

Debe incluir, en español, secciones concretas:
- Descripción del proyecto y las 9 clases.
- Requisitos e instalación (`python -m venv`, `pip install -r requirements.txt`).
- **Módulo 1 — Dataset:** estructura esperada (`images/`, `labels/`), recomendaciones de captura (150–300 imágenes/clase, iluminación, fondos, rotaciones, distancias, perspectivas, oclusión, varias monedas y denominaciones, escenarios reales), guía de etiquetado paso a paso con LabelImg/Roboflow y buenas prácticas, comandos: `python -m src.dataset.dividir_dataset <origen>`, `python -m src.dataset.generar_yaml`, `python -m src.dataset.verificar_dataset`.
- **Módulo 2 — Entrenamiento:** `python -m src.entrenamiento.entrenar`, parámetros, `python -m src.entrenamiento.evaluar`, dónde queda `best.pt` (`modelos/best.pt`).
- **Módulo 3 — App:** `python main.py`, descripción de botones y panel; nota de que abre sin modelo.
- Estructura de carpetas.

- [ ] **Step 2: Escribir `docs/documento_tecnico.md`**

Documento académico en español que explique el proceso completo:
- Introducción y objetivo del sistema.
- Construcción del dataset: cantidad de imágenes, variaciones (iluminación, fondos, rotaciones, distancias, perspectivas, oclusión, escenas con varias monedas/denominaciones, escenarios reales), división train/val/test (70/20/10) y por qué.
- Formato YOLO explicado con ejemplo de línea de anotación y `dataset.yaml`.
- Etiquetado: pasos, ejemplo, buenas prácticas (cajas ajustadas, etiquetar todas las instancias, consistencia de clases).
- Entrenamiento: elección de `yolo11n`, parámetros (epochs, batch, imgsz, learning rate, data augmentation, early stopping con `patience`).
- Métricas: definición de Precision, Recall, mAP50, mAP50-95 y loss; cómo interpretarlas.
- Interpretación de gráficas de Ultralytics: `results.png`, matriz de confusión, curvas P/R; cómo obtener `best.pt`.
- Arquitectura de la aplicación: separación interfaz / visión / modelo / utilidades / configuración, flujo cámara→detector→panel.
- Conclusiones.

- [ ] **Step 3: Verificar que los documentos existen y no están vacíos**

Run: `python -c "import os; assert os.path.getsize('README.md')>1500 and os.path.getsize('docs/documento_tecnico.md')>2000; print('docs OK')"`
Expected: imprime `docs OK`

- [ ] **Step 4: Commit**

```bash
git add README.md docs/documento_tecnico.md
git commit -m "docs: README y documento técnico del proyecto"
```

---

## Self-Review

**Cobertura del spec:**
- §3 clases → Tarea 1 (`CLASES`, `VALORES_BS`, `NOMBRES_MOSTRAR`). ✓
- §4 arquitectura/separación → Tareas 1–7 (estructura de carpetas). ✓
- §5 dataset (scripts + docs) → Tarea 3 + Tarea 7. ✓
- §6 entrenamiento (entrenar/evaluar + métricas) → Tarea 4 + doc técnico Tarea 7. ✓
- §7 app PyQt6 (cámara, boxes, nombre, confianza, lista, total, FPS, estado, conteo, 5 botones) → Tarea 6. ✓ (incluye además "Cargar modelo" por la robustez sin pesos del spec §2/§7).
- §8 calidad (PEP8, clases, type hints, docstrings) → presente en todo el código. ✓
- §9 documentación (README + documento técnico) → Tarea 7. ✓
- §2 app abre sin best.pt → Tarea 5 (Detector degradado) + Tarea 6 (estado + cargar modelo). ✓

**Placeholders:** las Tareas 1–6 contienen código completo. La Tarea 7 (documentación) describe contenido por secciones en lugar de texto íntegro; es contenido narrativo extenso que el implementador redacta siguiendo el índice dado — aceptable para documentación.

**Consistencia de tipos:** `Deteccion(clave, confianza, caja)` se define en Tarea 5 y se consume con esos nombres en Tareas 5 y 6. `denominaciones.monto_total/nombre_de/clave_de_id` definidos en Tarea 2 y usados en Tareas 5–6 con las mismas firmas. `Detector.cargado/cargar/detectar/dibujar` consistentes entre Tareas 5 y 6. ✓
```