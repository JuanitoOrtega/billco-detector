# Guía de entrenamiento

## Etiquetado de las imágenes (con labelImg)

Las fotos van en `mis_fotos/images/` y las etiquetas se guardan en `mis_fotos/labels/`. El archivo `mis_fotos/clases.txt` contiene las 9 clases **en el orden correcto** (no lo cambies: ese orden define el `id` numérico de cada clase, y debe coincidir con `config.CLASES`).

### 1. Abre labelImg con las clases cargadas
```bash
cd /Users/juanitodev/Courses/claude/billco-detector
labelImg mis_fotos/images mis_fotos/clases.txt
```
Así el desplegable de clases ya aparece con las 9 denominaciones en el orden correcto (no las escribas a mano).

### 2. Cambia el formato a YOLO (¡lo más importante!)
En la barra de la izquierda, debajo de **Save**, hay un botón que alterna el formato. Haz clic hasta que diga **YOLO**. Si queda en otro formato (PascalVOC/CreateML), los `.txt` no servirán para entrenar.

### 3. Indica las carpetas
- **Open Dir** → `mis_fotos/images` (si no se abrió sola).
- **Change Save Dir** → `mis_fotos/labels` (créala antes si no existe).

### 4. Etiqueta cada imagen
1. Pulsa **W** (o el botón "Create RectBox").
2. Dibuja un rectángulo **bien ajustado** alrededor de la moneda/billete.
3. Elige la clase correcta del desplegable (ej. `billete_20`).
4. Repite para **todos** los objetos visibles en la imagen.
5. **Ctrl+S** para guardar y **D** para pasar a la siguiente.

**Atajos:** `W` crear caja · `D` / `A` siguiente / anterior · `Ctrl+S` guardar · `Del` borrar caja seleccionada.

**Buenas prácticas:** cajas pegadas al borde visible del objeto; etiqueta *todas* las instancias presentes; misma denominación = misma clase siempre; si algo está tapado más del 50 % y no se identifica con certeza, omítelo.

> labelImg crea también un `classes.txt` dentro de `mis_fotos/labels/`. Es normal y no estorba: al dividir el dataset solo se copian los `.txt` que tienen su imagen, así que ese archivo queda fuera.

**Solución de problemas (macOS + Python 3.12):** si labelImg se cierra de golpe al dibujar una caja o al hacer scroll, con un error tipo `TypeError ... unexpected type 'float'`, es un bug conocido de esa versión. Se corrige convirtiendo a `int()` las coordenadas en `libs/canvas.py` (llamadas a `drawRect`/`drawLine`) y en `labelImg.py` (llamadas a `setValue`) dentro del entorno virtual.

---

Cuando termines de etiquetar **todas** las fotos, sigue estos pasos, en orden:

## 1. Verifica que cada foto tenga su etiqueta
Antes de dividir, asegúrate de no dejar imágenes sin su `.txt` (las que no etiquetaste no servirán). Puedes revisar rápido:
```bash
cd /Users/juanitodev/Courses/claude/billco-detector
# nº de imágenes vs nº de etiquetas
ls mis_fotos/images | wc -l
ls mis_fotos/labels/*.txt | grep -v classes.txt | wc -l
```
Idealmente los números coinciden (cada imagen con su etiqueta).

## 2. Divide el dataset (train/val/test)
```bash
python -m src.dataset.dividir_dataset mis_fotos
```
Esto crea `dataset/train`, `dataset/val` y `dataset/test` (70/20/10), emparejando cada imagen con su etiqueta. Solo copia los `.txt` que tienen su imagen, así que el `classes.txt` que generó labelImg se queda fuera (correcto).

## 3. Verifica el formato YOLO
```bash
python -m src.dataset.verificar_dataset
```
Debe reportar **0 errores**. Si marca alguno, dime el mensaje y lo revisamos antes de entrenar.

## 4. Entrena el modelo
```bash
python -m src.entrenamiento.entrenar
```
- Usa los parámetros de `config/config.py` (yolo11n, 640px, 100 epochs, batch 16, early stopping con patience 20).
- En tu Mac (M2, CPU) será **lento**; si quieres una corrida de prueba más rápida primero:
  ```bash
  python -m src.entrenamiento.entrenar --epochs 20 --batch 8
  ```
- Al terminar copia automáticamente los pesos a `modelos/best.pt`.

## 5. Evalúa el modelo
```bash
python -m src.entrenamiento.evaluar
```
Te imprime Precision, Recall, mAP50 y mAP50-95 sobre el conjunto de test.

## 6. Pruébalo en la app
```bash
python main.py
```
Como ya existirá `modelos/best.pt`, la app lo carga solo y el panel mostrará **"Modelo: cargado"**. Apunta la cámara a un billete/moneda y verás las cajas + el monto total en Bs.

---

⚠️ **Una advertencia importante de calidad:** para que el modelo detecte bien, cada clase necesita suficientes ejemplos variados (la guía recomienda ~150–300 por clase). Si tienes pocas fotos de alguna denominación, el modelo fallará en esa clase aunque entrene sin errores. No es un problema del código, sino de cantidad/variedad de datos.
