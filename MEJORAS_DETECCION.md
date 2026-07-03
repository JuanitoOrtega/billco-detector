# Mejoras del detector — diagnóstico y soluciones

Buen diagnóstico de tu parte, y tiene una explicación clara. No toco nada; te doy el panorama y las mejoras ordenadas por impacto.

## Por qué pasa esto (la raíz)

Hay **dos problemas** que se combinan:

**1. Tus métricas de 0.98 eran engañosas (fuga de datos / *data leakage*).** Sacaste 41 frames de cada video, y luego la división train/val/test fue **aleatoria por frame**. Como los frames de un mismo video son casi idénticos, terminaron fotos casi iguales en entrenamiento *y* en validación/test. El modelo no "generalizó": se evaluó a sí mismo con copias de lo que ya había visto. Por eso el número salió casi perfecto pero en la realidad falla.

**2. Desajuste de dominio (*domain shift*).** El modelo aprendió el contexto, no solo el billete:
- **Fondo:** siempre mesa blanca → aprendió "billete = objeto sobre blanco". Frente a tu cuerpo/mano/cuarto, ese contexto desaparece.
- **Pose:** billetes planos sobre la mesa vs. sostenidos, inclinados, doblados, tapados por los dedos.
- **Fuente:** frames de video vs. webcam en vivo (otra nitidez, otro encuadre, otra distancia).

**3. Falsos positivos con pantalla vacía.** Si *todas* tus fotos tienen exactamente un billete centrado, el modelo aprende un **sesgo fuerte de "siempre hay un billete"** y nunca vio ejemplos de "aquí no hay nada". Sin imágenes negativas, inventa detecciones.

---

## Mejoras, ordenadas por impacto

### 🔴 Lo que más mueve la aguja: el dataset
1. **Captura datos que se parezcan al uso real.** Graba/fotografía los billetes **sostenidos frente a la webcam** (tu escenario real), no sobre la mesa. Con **la misma cámara** que usará la app.
2. **Variedad, variedad, variedad:** distintos fondos (tu cuerpo, manos, cuartos, desorden), iluminaciones, distancias, ángulos, billetes doblados/gastados, **ambas caras** de cada denominación, y oclusión parcial por los dedos.
3. **Agrega imágenes negativas (sin dinero):** manos vacías, tu cara, fondos varios, otros objetos. En YOLO son imágenes sin etiqueta (o `.txt` vacío). Esto es lo que **más reduce los falsos positivos**.
4. **Más ejemplos *distintos*, no más frames del mismo video.** Apunta a 150–300 escenas **variadas** por clase. Si extraes de video, toma pocos frames por video y de **muchos** videos/condiciones distintas.

### 🟠 Evaluación honesta (para saber si de verdad mejoras)
5. **Divide por video/escena, no por frame.** Todos los frames de un mismo video van al mismo conjunto. Mejor aún: un **test independiente** capturado aparte, en el escenario real (sostenido en mano). Espera que las métricas *bajen* respecto al 0.98 — pero por fin serán reales.

### 🟡 Entrenamiento
6. **Aumento de datos dirigido** a simular lo real: cambios de fondo, brillo/HSV, rotación, perspectiva, desenfoque, recortes/oclusión. Ojo: el aumento **no inventa** fondos que el modelo nunca vio; los datos reales variados ganan siempre.
7. Cuando el dataset sea bueno, prueba un modelo un poco mayor (`yolo11s`). **Primero los datos, después el modelo** — no al revés.

### 🟢 Mitigaciones en la app (parches, no arreglan la raíz)
8. **Subir el umbral de confianza** (`CONF_MIN` de 0.5 a ~0.6–0.8): corta falsos positivos, pero también baja la sensibilidad.
9. **Suavizado temporal:** exigir que una detección aparezca en varios frames seguidos antes de darla por válida. Reduce muchísimo el parpadeo y los falsos positivos, y mejora la sensación de uso.
10. Filtros de cordura: descartar cajas demasiado pequeñas o con proporción imposible para un billete.

---

## Mi recomendación en una frase
El detector no está "roto": está **sobreajustado a un dominio muy estrecho** y evaluado con fuga de datos. La mejora sustancial (80% del resultado) viene de **recolectar un dataset representativo del uso real + imágenes negativas + dividir por video**. Las mitigaciones en la app (8–10) son un buen alivio inmediato mientras reúnes mejores datos.
