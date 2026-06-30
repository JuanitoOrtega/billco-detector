¡Excelente progreso! Cuando termines de etiquetar **todas** las fotos, estos son los pasos, en orden:

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
