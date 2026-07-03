# Glosario — términos de visión por computadora y entrenamiento

Explicaciones sencillas de los términos técnicos que aparecen en este proyecto,
con analogías cotidianas.

## Dominio (y "en dominio" / "fuera de dominio")
El **dominio** es el "mundo" de datos con el que el modelo aprendió: en este caso, *billetes planos, sobre mesa blanca, sacados de video*.

- **En dominio** = se le muestra algo parecido a lo que vio al entrenar (billete sobre blanco) → funciona bien.
- **Fuera de dominio** = se le muestra algo distinto (billete en la mano, frente al cuerpo) → falla.

> 🧠 Analogía: un niño que solo vio perros en fotos de estudio con fondo blanco. Le enseñas un perro en un parque con barro y no lo reconoce. El perro es el mismo; el *contexto* cambió.

Cuando el rendimiento se cae al cambiar de contexto, se llama **domain shift** (*desajuste de dominio*).

## Sesgo (*bias*)
Es una **inclinación aprendida** del modelo hacia algo, por culpa de cómo eran los datos.
En este caso: como *todas* las fotos tenían un billete, el modelo aprendió el prejuicio *"siempre hay un billete"*. Por eso inventa detecciones aunque la pantalla esté vacía.

> 🧠 Analogía: si a alguien solo le muestras fotos donde siempre hay una moneda, empezará a "ver" monedas hasta donde no las hay, porque asume que siempre debe haber una.

El sesgo casi siempre viene de un **dataset sesgado** (poco variado, no representativo del uso real).

## Generalización
Es la meta de todo modelo: que lo aprendido **funcione con ejemplos nuevos** que nunca vio, no solo con los del entrenamiento. Un modelo que *memorizó* su dominio pero **no generaliza** bien falla fuera de él.

## Sobreajuste (*overfitting*)
Cuando el modelo **memoriza** los datos de entrenamiento en vez de aprender lo esencial. Rinde perfecto en lo que vio y mal en lo nuevo. Es lo opuesto a generalizar.

> 🧠 Analogía: un estudiante que memoriza las respuestas del examen del año pasado. Saca 100 si cae el mismo examen, pero se hunde si cambian las preguntas.

(Lo contrario, **subajuste / *underfitting***, es cuando ni siquiera aprende lo básico.)

## Fuga de datos (*data leakage*)
Cuando información del conjunto de prueba "se cuela" en el de entrenamiento, y las métricas salen **falsamente altas**. Ejemplo real de este proyecto: al dividir *frames casi idénticos* del mismo video entre entrenamiento y validación, el modelo se evaluó con copias de lo que ya había visto → mAP de 0.98 engañoso.

> 🧠 Analogía: tomar el examen con las mismas preguntas que estudiaste de práctica. Tu nota no mide cuánto sabes de verdad.

## Falsos positivos y falsos negativos
- **Falso positivo:** dice "hay un billete de 20" cuando **no lo hay** (el problema con la pantalla vacía).
- **Falso negativo:** **no** ve un billete que **sí** está.

De aquí salen dos métricas clave:
- **Precisión (*precision*):** de lo que detectó, ¿cuánto acertó? (poca precisión = muchos falsos positivos).
- **Sensibilidad (*recall*):** de lo que había, ¿cuánto encontró? (poco recall = muchos falsos negativos).

## Otros términos usados
- **Aprendizaje por transferencia (*transfer learning*):** no se entrena desde cero; se parte de YOLO11n, que ya sabía "ver" formas, y se reentrena para la tarea. Ahorra datos y tiempo.
- **Aumento de datos (*data augmentation*):** crear variaciones artificiales de las fotos (giros, brillo, recortes) para simular más variedad. Ayuda, pero **no inventa** fondos que nunca se vieron.
- **Imágenes negativas:** fotos **sin** dinero que se agregan al dataset para enseñarle al modelo qué es "nada". Reducen los falsos positivos.
- **Umbral de confianza (`CONF_MIN`):** qué tan seguro debe estar el modelo para mostrar una detección. Subirlo corta falsos positivos.

---

En una frase que conecta todo: *el modelo **sobreajustó** su **dominio** estrecho por un **dataset sesgado**, sus métricas se inflaron por **fuga de datos**, y por eso **no generaliza** y produce **falsos positivos** fuera de ese dominio.*
