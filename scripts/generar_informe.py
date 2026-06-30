"""Genera el informe técnico del proyecto en formato Word (.docx).

Uso:  python -m scripts.generar_informe

Construye un documento profesional con la descripción de la aplicación, su
arquitectura, el dataset, el entrenamiento, los resultados y el análisis de
cada gráfico. Embebe los gráficos reales de la carpeta ``resultados/``.
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from config import config

VERDE = RGBColor(0x1B, 0x5E, 0x20)
VERDE_CLARO = "E8F5E9"
GRIS = RGBColor(0x55, 0x55, 0x55)
RAIZ = config.RAIZ
RES = RAIZ / "resultados"
SALIDA = RAIZ / "Informe_Tecnico_Billco_Detector.docx"


def _sombrear(celda, hex_color: str) -> None:
    """Aplica color de fondo a una celda de tabla."""
    tcpr = celda._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_color)
    tcpr.append(shd)


def _numero_pagina(footer) -> None:
    """Inserta el número de página centrado en el pie."""
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fldsimple = OxmlElement("w:fldSimple")
    fldsimple.set(qn("w:instr"), "PAGE")
    run._r.append(fldsimple)


def _img(doc: Document, nombre: str, ancho: float) -> None:
    """Inserta una imagen de resultados/ centrada."""
    ruta = RES / nombre
    if not ruta.exists():
        doc.add_paragraph(f"[No se encontró el gráfico: {nombre}]")
        return
    doc.add_picture(str(ruta), width=Inches(ancho))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


def _pie(doc: Document, texto: str) -> None:
    """Añade un pie de figura en cursiva y centrado."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(texto)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = GRIS


def _tabla(doc: Document, encabezados: list[str], filas: list[list[str]]) -> None:
    """Crea una tabla con estilo y encabezado sombreado en verde."""
    tabla = doc.add_table(rows=1, cols=len(encabezados))
    tabla.style = "Light Grid Accent 1"
    tabla.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, texto in enumerate(encabezados):
        celda = tabla.rows[0].cells[i]
        celda.text = ""
        run = celda.paragraphs[0].add_run(texto)
        run.bold = True
        run.font.size = Pt(10)
        _sombrear(celda, VERDE_CLARO)
    for fila in filas:
        celdas = tabla.add_row().cells
        for i, valor in enumerate(fila):
            celdas[i].text = ""
            run = celdas[i].paragraphs[0].add_run(str(valor))
            run.font.size = Pt(10)


def _config_estilos(doc: Document) -> None:
    """Ajusta la fuente base y el color de los títulos."""
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    for nivel in ("Title", "Heading 1", "Heading 2", "Heading 3"):
        try:
            doc.styles[nivel].font.color.rgb = VERDE
        except KeyError:
            pass


def _portada(doc: Document) -> None:
    """Construye la portada del informe."""
    for _ in range(3):
        doc.add_paragraph()
    icono = RAIZ / "assets" / "icono.png"
    if icono.exists():
        doc.add_picture(str(icono), width=Inches(1.4))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("Billco Detector")
    r.bold = True
    r.font.size = Pt(34)
    r.font.color.rgb = VERDE

    s = doc.add_paragraph()
    s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rs = s.add_run("Informe Técnico\nSistema Inteligente de Detección y Cálculo de Monedas\ny Billetes Bolivianos con YOLO11 y PyQt6")
    rs.font.size = Pt(15)
    rs.font.color.rgb = GRIS

    for _ in range(6):
        doc.add_paragraph()

    for etiqueta, valor in (
        ("Autor", config.AUTOR),
        ("Versión del sistema", f"v{config.VERSION}"),
        ("Fecha", "Junio de 2026"),
        ("Repositorio", "github.com/JuanitoOrtega/billco-detector"),
    ):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        rb = p.add_run(f"{etiqueta}: ")
        rb.bold = True
        p.add_run(valor)
    doc.add_page_break()


def _h(doc: Document, texto: str, nivel: int = 1):
    return doc.add_heading(texto, level=nivel)


def _p(doc: Document, texto: str):
    return doc.add_paragraph(texto)


def _bullet(doc: Document, texto: str):
    return doc.add_paragraph(texto, style="List Bullet")


def construir() -> Path:
    """Genera el documento completo y lo guarda. Devuelve la ruta."""
    doc = Document()
    _config_estilos(doc)
    _numero_pagina(doc.sections[0].footer)
    _portada(doc)

    # 1. Resumen ejecutivo
    _h(doc, "1. Resumen ejecutivo")
    _p(doc, "Billco Detector es un sistema de visión artificial que detecta en tiempo real "
            "monedas y billetes del Estado Plurinacional de Bolivia y calcula automáticamente "
            "el monto total visible en la cámara. El proyecto abarca el flujo completo de un "
            "sistema de Visión Artificial: la construcción del dataset, el entrenamiento de un "
            "modelo YOLO11 propio y una aplicación de escritorio desarrollada en PyQt6.")
    _p(doc, "El modelo final, entrenado específicamente para esta tarea, alcanzó métricas "
            "sobresalientes en el conjunto de prueba: una precisión media de 0.991, un recall "
            "de 1.000 y un mAP50 de 0.995. El sistema reconoce nueve denominaciones distintas y "
            "es capaz de operar en tiempo real sobre CPU.")

    # 2. Introducción y objetivos
    _h(doc, "2. Introducción y objetivos")
    _p(doc, "El objetivo del proyecto es demostrar, de principio a fin, la construcción de un "
            "sistema de Visión Artificial aplicado a un problema real: el reconocimiento "
            "automático de dinero boliviano. Una premisa fundamental es que el sistema NO depende "
            "de modelos preentrenados que ya reconozcan dinero boliviano; el modelo se entrena "
            "específicamente a partir de un dataset construido por el autor.")
    _p(doc, "El proyecto se organiza en tres módulos principales:")
    _bullet(doc, "Módulo 1 — Construcción del dataset: recolección, etiquetado y división de imágenes.")
    _bullet(doc, "Módulo 2 — Entrenamiento del modelo YOLO11 y su evaluación.")
    _bullet(doc, "Módulo 3 — Aplicación de escritorio para la detección en tiempo real.")
    _p(doc, "Las nueve denominaciones que el sistema reconoce son las siguientes:")
    _tabla(doc, ["ID", "Clase", "Denominación", "Valor (Bs)"], [
        ["0", "moneda_050", "Moneda 0.50", "0.50"],
        ["1", "moneda_1", "Moneda 1", "1.00"],
        ["2", "moneda_2", "Moneda 2", "2.00"],
        ["3", "moneda_5", "Moneda 5", "5.00"],
        ["4", "billete_10", "Billete 10", "10.00"],
        ["5", "billete_20", "Billete 20", "20.00"],
        ["6", "billete_50", "Billete 50", "50.00"],
        ["7", "billete_100", "Billete 100", "100.00"],
        ["8", "billete_200", "Billete 200", "200.00"],
    ])

    # 3. Descripción de la aplicación
    _h(doc, "3. Descripción de la aplicación")
    _p(doc, "La aplicación de escritorio, desarrollada en PyQt6, ofrece una interfaz moderna para "
            "la detección de dinero en tiempo real. Captura el video de la cámara web, ejecuta la "
            "inferencia del modelo sobre cada fotograma y muestra los resultados de forma visual e "
            "inmediata.")
    _h(doc, "3.1. Funcionalidades principales", 2)
    _bullet(doc, "Vista de cámara en tiempo real con cajas delimitadoras (bounding boxes) sobre cada objeto detectado.")
    _bullet(doc, "Etiqueta con el nombre de la denominación y el porcentaje de confianza de cada detección.")
    _bullet(doc, "Lista de objetos detectados en el fotograma actual.")
    _bullet(doc, "Cálculo y visualización del monto total acumulado en bolivianos (Bs).")
    _bullet(doc, "Indicador de fotogramas por segundo (FPS) del flujo de video.")
    _bullet(doc, "Indicador del estado del modelo (cargado / no cargado).")
    _bullet(doc, "Contador de la cantidad de objetos detectados.")
    _h(doc, "3.2. Controles", 2)
    _tabla(doc, ["Botón", "Acción"], [
        ["Iniciar cámara", "Arranca la captura de video en tiempo real."],
        ["Detener cámara", "Congela la imagen y detiene el hilo de captura."],
        ["Capturar imagen", "Retiene el último fotograma para su revisión."],
        ["Guardar captura", "Exporta la captura a disco (.jpg / .png)."],
        ["Cargar modelo…", "Abre un diálogo para seleccionar un archivo de pesos .pt."],
        ["Acerca de", "Muestra nombre, versión, autor y descripción."],
        ["Salir", "Cierra la aplicación, solicitando confirmación."],
    ])
    _h(doc, "3.3. Robustez", 2)
    _p(doc, "Una decisión de diseño relevante es que la aplicación abre y funciona aunque no exista "
            "el archivo de pesos best.pt. En ese caso muestra el estado «Modelo: no cargado», la "
            "cámara sigue operando y un botón permite cargar el modelo manualmente. La carga del "
            "modelo YOLO es perezosa (lazy import), de modo que la interfaz arranca incluso sin las "
            "dependencias pesadas instaladas. La captura de cámara se ejecuta en un hilo separado "
            "(QThread) para no congelar la interfaz.")

    # 4. Arquitectura y estructura
    _h(doc, "4. Arquitectura y estructura del proyecto")
    _p(doc, "El proyecto sigue una arquitectura limpia con separación clara de responsabilidades. "
            "Cada componente tiene un propósito único y se comunica con los demás a través de "
            "interfaces bien definidas. La configuración del proyecto (clases, valores, rutas y "
            "parámetros) reside en un único módulo, config/config.py, que actúa como fuente única "
            "de verdad.")
    _h(doc, "4.1. Estructura de carpetas", 2)
    arbol = (
        "billco-detector/\n"
        "├── config/            Configuración central (fuente única de verdad)\n"
        "├── src/\n"
        "│   ├── dataset/       Módulo 1: dividir, verificar y generar dataset.yaml\n"
        "│   ├── entrenamiento/ Módulo 2: entrenar y evaluar el modelo\n"
        "│   ├── vision/        Núcleo de visión: detector y cámara (hilo)\n"
        "│   ├── app/           Módulo 3: interfaz PyQt6 (ventana y widgets)\n"
        "│   └── utils/         Utilidades: cálculo de montos por denominación\n"
        "├── tests/             Pruebas automatizadas (pytest)\n"
        "├── scripts/           Utilidades del proyecto (releases, informe)\n"
        "├── assets/            Recursos gráficos (icono)\n"
        "├── resultados/        Gráficos del entrenamiento\n"
        "└── main.py            Punto de entrada de la aplicación"
    )
    mono = doc.add_paragraph()
    run = mono.add_run(arbol)
    run.font.name = "Consolas"
    run.font.size = Pt(9.5)
    _h(doc, "4.2. Separación de responsabilidades", 2)
    _tabla(doc, ["Capa", "Ubicación", "Responsabilidad"], [
        ["Interfaz", "src/app/", "Ventana, widgets y eventos de usuario."],
        ["Visión artificial", "src/vision/", "Captura de cámara e inferencia sobre fotogramas."],
        ["Modelo YOLO", "src/entrenamiento/ + detector.py", "Entrenamiento, evaluación y carga del modelo."],
        ["Utilidades", "src/utils/", "Mapeo de clase a valor en Bs y suma de montos."],
        ["Configuración", "config/", "Clases, valores, rutas y parámetros."],
    ])
    _h(doc, "4.3. Calidad del código", 2)
    _p(doc, "El código sigue la guía de estilo PEP8 y emplea clases, anotaciones de tipo (type "
            "hints), docstrings en español y comentarios donde aportan claridad. Se evita la "
            "duplicación y se favorecen archivos pequeños y enfocados, cada uno con una "
            "responsabilidad única.")

    # 5. Dataset
    _h(doc, "5. Construcción del dataset")
    _p(doc, "El dataset es la base de todo sistema de Visión Artificial. Para este proyecto se "
            "capturaron fotografías propias de las nueve denominaciones y se etiquetaron en formato "
            "YOLO.")
    _h(doc, "5.1. Formato YOLO", 2)
    _p(doc, "Cada imagen se acompaña de un archivo de texto con el mismo nombre. Cada línea de ese "
            "archivo describe un objeto con cinco valores: el identificador de la clase y las "
            "coordenadas normalizadas (de 0 a 1) del centro y el tamaño de la caja:")
    ej = doc.add_paragraph()
    rej = ej.add_run("<clase> <x_centro> <y_centro> <ancho> <alto>\n0 0.512 0.488 0.210 0.305")
    rej.font.name = "Consolas"
    rej.font.size = Pt(10)
    _h(doc, "5.2. Recomendaciones de captura", 2)
    _p(doc, "Para un dataset de calidad se recomienda variar las condiciones de captura:")
    for t in ("Cantidad: idealmente entre 150 y 300 imágenes por clase.",
              "Iluminación: luz natural, artificial, directa y difusa, con sombras parciales.",
              "Fondos: superficies de distintos colores, texturas y patrones.",
              "Rotaciones, distancias y perspectivas variadas.",
              "Oclusión parcial y escenas con varias monedas y denominaciones juntas.",
              "Escenarios reales (mostrador, billetera, escritorio)."):
        _bullet(doc, t)
    _h(doc, "5.3. Etiquetado y división", 2)
    _p(doc, "El etiquetado se realizó con la herramienta LabelImg, exportando en formato YOLO y "
            "manteniendo el orden exacto de las clases definido en la configuración, para que el "
            "identificador numérico de cada clase coincida con su valor monetario. Las cajas se "
            "dibujan ajustadas al contorno visible de cada moneda o billete y se etiquetan todas "
            "las instancias presentes en cada imagen.")
    _p(doc, "El dataset se divide automáticamente en tres conjuntos mediante el script "
            "dividir_dataset, con una proporción de 70 % para entrenamiento, 20 % para validación "
            "y 10 % para prueba. Un segundo script, verificar_dataset, comprueba que todas las "
            "etiquetas cumplan el formato YOLO antes de entrenar.")

    # 6. Entrenamiento
    _h(doc, "6. Entrenamiento del modelo")
    _p(doc, "El entrenamiento parte del modelo base YOLO11n (variante «nano», la más ligera y "
            "rápida) mediante una técnica de aprendizaje por transferencia (transfer learning): se "
            "aprovecha un modelo que ya sabe «ver» y se reentrena para la tarea concreta de "
            "reconocer dinero boliviano. Esto reduce drásticamente la cantidad de imágenes y el "
            "tiempo necesarios.")
    _h(doc, "6.1. Parámetros de entrenamiento", 2)
    _tabla(doc, ["Parámetro", "Valor", "Significado"], [
        ["modelo base", "yolo11n", "Modelo de partida, versión nano (rápida y ligera)."],
        ["imgsz", "640", "Las imágenes se redimensionan a 640 × 640 píxeles."],
        ["epochs", "100", "Número máximo de pasadas por todo el dataset."],
        ["batch", "16", "Imágenes procesadas a la vez por paso."],
        ["lr0", "0.01", "Tasa de aprendizaje inicial (tamaño del ajuste por paso)."],
        ["patience", "20", "Detiene si no mejora en 20 epochs (early stopping)."],
    ])
    _h(doc, "6.2. Early stopping y duración", 2)
    _p(doc, "El entrenamiento se detuvo automáticamente: tras 20 epochs consecutivas sin mejora en "
            "validación, el sistema concluyó que no tenía sentido continuar. Aunque el límite era "
            "de 100 epochs, el proceso llegó hasta la epoch 74 y guardó como modelo definitivo la "
            "mejor versión, obtenida en la epoch 54. El entrenamiento completo tomó 2.494 horas "
            "sobre CPU (Apple M2). Este mecanismo evita el sobreajuste y ahorra tiempo de cómputo.")
    _p(doc, "Adicionalmente, el script de entrenamiento incorpora una verificación previa: si el "
            "dataset no está disponible, muestra un mensaje claro con los pasos a seguir en lugar "
            "de un error técnico, y el modelo final (best.pt) se copia automáticamente a la carpeta "
            "modelos/.")

    # 7. Resultados
    _h(doc, "7. Resultados del entrenamiento")
    _p(doc, "El modelo se evaluó sobre el conjunto de prueba (73 imágenes que no participaron en el "
            "entrenamiento). Las métricas globales obtenidas son:")
    _tabla(doc, ["Métrica", "Valor", "Interpretación"], [
        ["Precision", "0.991", "De lo que detectó, casi todo fue correcto."],
        ["Recall", "1.000", "Encontró todos los objetos presentes."],
        ["mAP50", "0.995", "Precisión media con solapamiento del 50 %."],
        ["mAP50-95", "0.980", "Precisión media bajo el criterio más estricto."],
    ])
    _p(doc, "El modelo final es liviano (5.5 MB; 2.58 millones de parámetros; 101 capas; 6.3 "
            "GFLOPs) y rápido: aproximadamente 76 ms por imagen en CPU (0.3 ms de preprocesamiento, "
            "75.6 ms de inferencia y 0.3 ms de posprocesamiento), lo que equivale a unos 13 "
            "fotogramas por segundo, suficiente para operar en tiempo real.")
    _h(doc, "7.1. Métricas por clase", 2)
    _tabla(doc, ["Clase", "Imágenes", "Precision", "Recall", "mAP50", "mAP50-95"], [
        ["moneda_050", "6", "0.987", "1.000", "0.995", "0.995"],
        ["moneda_1", "10", "0.998", "1.000", "0.995", "0.961"],
        ["moneda_2", "7", "0.990", "1.000", "0.995", "0.995"],
        ["moneda_5", "9", "0.993", "1.000", "0.995", "0.972"],
        ["billete_10", "6", "0.988", "1.000", "0.995", "0.995"],
        ["billete_20", "9", "0.991", "1.000", "0.995", "0.958"],
        ["billete_50", "11", "0.993", "1.000", "0.995", "0.951"],
        ["billete_100", "8", "0.990", "1.000", "0.995", "0.995"],
        ["billete_200", "7", "0.990", "1.000", "0.995", "0.995"],
    ])
    _p(doc, "Nota de honestidad metodológica: estos resultados, casi perfectos, se obtuvieron con "
            "un dataset relativamente pequeño (entre 6 y 11 imágenes por clase en validación) y con "
            "condiciones de captura similares. Para confirmar que el modelo generaliza a "
            "situaciones nuevas conviene ampliar y diversificar el dataset.")

    # 8. Análisis de gráficos
    _h(doc, "8. Análisis de los gráficos del entrenamiento")
    _p(doc, "Durante el entrenamiento, YOLO genera automáticamente un conjunto de gráficos que "
            "permiten evaluar la calidad del proceso y del modelo. A continuación se analiza cada "
            "uno en detalle.")

    _h(doc, "8.1. Curvas de pérdida y métricas (results.png)", 2)
    _img(doc, "results.png", 6.3)
    _pie(doc, "Figura 1. Evolución de las pérdidas y métricas a lo largo de las epochs.")
    _p(doc, "Este panel resume todo el entrenamiento. Las tres primeras columnas muestran las "
            "funciones de pérdida —box_loss (precisión de la ubicación de las cajas), cls_loss "
            "(acierto en la clase) y dfl_loss (refinamiento del borde de la caja)— tanto en el "
            "conjunto de entrenamiento (fila superior) como en el de validación (fila inferior). "
            "Las dos últimas columnas muestran las métricas de calidad. La lectura correcta es que "
            "las pérdidas deben descender y las métricas ascender hasta estabilizarse. Es "
            "especialmente importante que las curvas de entrenamiento y validación bajen juntas: si "
            "la de entrenamiento bajara mientras la de validación sube, sería señal de sobreajuste. "
            "En este caso, todas las pérdidas descienden de forma sostenida y las métricas suben "
            "hasta valores cercanos a 1.0 y se aplanan, lo que indica un entrenamiento sano y bien "
            "ajustado.")

    _h(doc, "8.2. Matriz de confusión normalizada", 2)
    _img(doc, "confusion_matrix_normalized.png", 5.2)
    _pie(doc, "Figura 2. Matriz de confusión normalizada (proporciones).")
    _p(doc, "La matriz de confusión contrasta la clase predicha por el modelo (filas) con la clase "
            "real (columnas). Los valores de la diagonal principal representan los aciertos, "
            "mientras que cualquier valor fuera de ella indicaría una confusión entre dos "
            "denominaciones. En su versión normalizada, cada celda expresa una proporción de 0 a 1. "
            "El resultado es una diagonal perfecta, con todas las celdas en 1.00 y ningún valor "
            "fuera de ella: el modelo no confunde ninguna denominación con otra en el conjunto de "
            "validación.")

    _h(doc, "8.3. Matriz de confusión en conteos", 2)
    _img(doc, "confusion_matrix.png", 5.2)
    _pie(doc, "Figura 3. Matriz de confusión en número absoluto de detecciones.")
    _p(doc, "Esta variante presenta la misma información pero con el número absoluto de detecciones "
            "en cada celda, en lugar de proporciones. Permite ver cuántos ejemplos de cada clase se "
            "evaluaron y confirmar que todas las detecciones cayeron en la celda correcta de la "
            "diagonal, sin dispersión hacia otras clases ni hacia el fondo (background).")

    _h(doc, "8.4. Curva Precision–Recall (BoxPR_curve.png)", 2)
    _img(doc, "BoxPR_curve.png", 5.6)
    _pie(doc, "Figura 4. Curva Precision–Recall por clase.")
    _p(doc, "La curva Precision–Recall muestra el equilibrio entre la precisión (proporción de "
            "detecciones correctas) y el recall (proporción de objetos encontrados) a medida que se "
            "varía el umbral de decisión. El área bajo la curva corresponde al mAP. Cuanto más se "
            "acerca la curva a la esquina superior derecha —el punto ideal (1, 1)—, mejor es el "
            "modelo. En este proyecto, todas las curvas se sitúan prácticamente en esa esquina, con "
            "un mAP50 global de 0.995.")

    _h(doc, "8.5. Curva F1 vs confianza (BoxF1_curve.png)", 2)
    _img(doc, "BoxF1_curve.png", 5.6)
    _pie(doc, "Figura 5. Puntuación F1 en función del umbral de confianza.")
    _p(doc, "La puntuación F1 combina precisión y recall en un solo número (su media armónica). "
            "Esta curva muestra cómo varía el F1 según el umbral de confianza que se exija a las "
            "detecciones. El punto más alto de la curva señala el umbral de confianza óptimo y "
            "sirve para elegir el parámetro CONF_MIN de la aplicación. En este caso la curva se "
            "mantiene alta y plana en un amplio rango de confianza, lo que indica que el modelo es "
            "robusto y que el umbral de 0.5 utilizado por la aplicación es una elección segura.")

    _h(doc, "8.6. Curvas de Precisión y Recall vs confianza", 2)
    _img(doc, "BoxP_curve.png", 5.4)
    _pie(doc, "Figura 6. Precisión en función del umbral de confianza.")
    _img(doc, "BoxR_curve.png", 5.4)
    _pie(doc, "Figura 7. Recall en función del umbral de confianza.")
    _p(doc, "Estas dos curvas descomponen el F1 en sus dos componentes. La curva de precisión "
            "tiende a subir cuando se exige más confianza, porque se descartan las detecciones "
            "dudosas y disminuyen los falsos positivos. La curva de recall, en cambio, tiende a "
            "bajar al exigir más confianza, porque algunos objetos válidos quedan por debajo del "
            "umbral y se pierden. El equilibrio entre ambas es justamente lo que resume la curva "
            "F1. En ambos casos los valores se mantienen muy altos y estables.")

    _h(doc, "8.7. Distribución del dataset (labels.jpg)", 2)
    _img(doc, "labels.jpg", 5.6)
    _pie(doc, "Figura 8. Distribución de instancias, posiciones y tamaños de las cajas.")
    _p(doc, "Este gráfico resume la composición del dataset: el número de instancias por clase y la "
            "distribución de la posición y el tamaño de las cajas etiquetadas. Sirve para detectar "
            "desbalances: si una clase tuviera muchas más muestras que las demás, el modelo podría "
            "sesgarse hacia ella. En este proyecto el dataset es pequeño pero está razonablemente "
            "balanceado entre las nueve clases. Conviene recordar que un dataset reducido invita a "
            "validar el modelo con fotografías nuevas y variadas.")

    _h(doc, "8.8. Lotes de entrenamiento (train_batch)", 2)
    _img(doc, "train_batch0.jpg", 5.8)
    _pie(doc, "Figura 9. Ejemplo de lote de entrenamiento con aumento de datos.")
    _p(doc, "Estas imágenes muestran fotogramas reales tal como entran al modelo durante el "
            "entrenamiento, con sus etiquetas y con las transformaciones del aumento de datos (data "
            "augmentation): recortes, giros, cambios de brillo y composiciones tipo mosaico. El "
            "aumento de datos multiplica artificialmente la variedad del dataset y ayuda al modelo "
            "a generalizar. Revisar estos lotes permite confirmar que las cajas y las clases llegan "
            "correctamente al proceso de entrenamiento.")

    _h(doc, "8.9. Predicciones sobre validación (val_batch_pred)", 2)
    _img(doc, "val_batch0_pred.jpg", 5.8)
    _pie(doc, "Figura 10. Predicciones del modelo sobre imágenes de validación.")
    _p(doc, "Aquí se observa el modelo aplicado a imágenes que no utilizó para entrenar. Cada caja "
            "incluye la clase predicha y la confianza, mostradas juntas (por ejemplo, «moneda_5 "
            "1.0» significa la clase moneda_5 con una confianza de 1.0). Estas predicciones se "
            "comparan con las etiquetas reales de las mismas imágenes (figura siguiente): si "
            "coinciden, el modelo acierta.")

    _h(doc, "8.10. Etiquetas reales de validación (val_batch_labels)", 2)
    _img(doc, "val_batch0_labels.jpg", 5.8)
    _pie(doc, "Figura 11. Etiquetas reales (verdad de terreno) de las imágenes de validación.")
    _p(doc, "Esta figura muestra la «verdad de terreno»: las etiquetas correctas de las mismas "
            "imágenes de validación de la figura anterior. Comparando ambas se comprueba que las "
            "predicciones del modelo coinciden con las clases reales, lo que confirma "
            "cualitativamente las excelentes métricas obtenidas.")

    # 9. Pruebas
    _h(doc, "9. Pruebas y calidad del software")
    _p(doc, "El proyecto incluye un conjunto de pruebas automatizadas con pytest que verifican la "
            "lógica crítica del sistema. Las pruebas se escribieron siguiendo la metodología de "
            "desarrollo guiado por pruebas (TDD): primero la prueba que falla y luego la "
            "implementación mínima que la hace pasar. En total son 20 pruebas, todas en verde.")
    _tabla(doc, ["Archivo de pruebas", "Nº", "Qué verifica"], [
        ["test_denominaciones.py", "7", "Mapeo de clase a valor en Bs y cálculo del monto total."],
        ["test_verificar_dataset.py", "6", "Validación del formato YOLO de las etiquetas."],
        ["test_entrenar_preflight.py", "4", "Verificación previa del dataset antes de entrenar."],
        ["test_notas_release.py", "3", "Extracción de notas de versión desde el CHANGELOG."],
    ])
    _p(doc, "Las pruebas se centran en la lógica pura (cálculos y validaciones), que es la que más "
            "se beneficia de la verificación automática. Los componentes gráficos y el entrenamiento "
            "se verifican de forma complementaria mediante comprobaciones de importación y ejecución.")
    _h(doc, "9.1. Versionado y publicación automática", 2)
    _p(doc, "El proyecto emplea versionado semántico (MAYOR.MENOR.PARCHE) con una única fuente de "
            "verdad para la versión. Un flujo de integración continua en GitHub Actions ejecuta las "
            "pruebas en cada cambio y, cuando la versión cambia, crea automáticamente la etiqueta y "
            "publica el release tomando las notas del archivo CHANGELOG.md.")

    # 10. Conclusiones
    _h(doc, "10. Conclusiones y trabajo futuro")
    _p(doc, "El proyecto cumple su objetivo: demuestra el flujo completo de un sistema de Visión "
            "Artificial, desde la construcción del dataset hasta una aplicación funcional, pasando "
            "por el entrenamiento de un modelo propio. El modelo YOLO11n entrenado alcanza métricas "
            "sobresalientes y opera en tiempo real sobre CPU, y la aplicación de escritorio integra "
            "la detección con el cálculo automático del monto total.")
    _p(doc, "Como líneas de trabajo futuro se proponen: ampliar y diversificar el dataset (más "
            "imágenes por clase, fondos, iluminaciones y escenarios) para robustecer la "
            "generalización; evaluar variantes mayores del modelo (yolo11s/m) si se dispone de GPU; "
            "y empaquetar la aplicación como ejecutable para su distribución.")

    doc.save(str(SALIDA))
    return SALIDA


def main() -> None:
    """Punto de entrada CLI."""
    ruta = construir()
    print(f"Informe generado en: {ruta}")


if __name__ == "__main__":
    main()
