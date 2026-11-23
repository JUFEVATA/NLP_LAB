"""Aplicación didáctica Streamlit para el laboratorio de NLP.

Ejecutar con: python -m streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from nlp_utils import (
    cadena,
    clasificar_sentimiento,
    comparar_bow,
    comparar_embeddings,
    detalle_spacy,
    limpiar,
    tokenizar_beto,
)


st.set_page_config(
    page_title="NLP Lab | Laboratorio de textos",
    page_icon="🔤",
    layout="wide",
)

MODELO_BETO = "dccuchile/bert-base-spanish-wwm-cased"
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODELO_SENTIMIENTOS = "nlptown/bert-base-multilingual-uncased-sentiment"


@st.cache_resource(show_spinner="Cargando spaCy en español…")
def cargar_spacy():
    import spacy

    return spacy.load("es_core_news_sm")


@st.cache_resource(show_spinner="Descargando/cargando el tokenizador BETO…")
def cargar_beto():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODELO_BETO)


@st.cache_resource(show_spinner="Descargando/cargando Sentence Transformers…")
def cargar_embeddings():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODELO_EMBEDDINGS, device="cpu")


@st.cache_resource(show_spinner="Descargando/cargando el modelo de sentimientos…")
def cargar_clasificador():
    from transformers import pipeline

    return pipeline("text-classification", model=MODELO_SENTIMIENTOS, device=-1)


def obtener_modelo(funcion, nombre):
    """Muestra un error comprensible si falla la carga de un recurso."""
    try:
        return funcion()
    except Exception as exc:
        st.error(f"No fue posible cargar {nombre}.")
        st.caption(
            "Verifica la conexión a Internet, los paquetes de requirements.txt "
            "y las instrucciones de README.md."
        )
        st.exception(exc)
        st.stop()


def mostrar_descarga(tabla: pd.DataFrame, nombre: str):
    st.download_button(
        "Descargar resultados en CSV",
        data=tabla.to_csv(index=False).encode("utf-8-sig"),
        file_name=nombre,
        mime="text/csv",
    )


def pagina_analisis():
    st.subheader("1. Análisis lingüístico y tokenización")
    st.write("Compara la separación por espacios, spaCy y el tokenizador BETO.")
    texto = st.text_area(
        "Texto para analizar",
        value="¿Dónde encuentro la grabación del encuentro 2? No me gustó nada.",
        height=120,
        key="entrada_analisis",
    )
    columna_a, columna_b = st.columns(2)
    with columna_a:
        conservar_negaciones = st.checkbox(
            "Preservar negaciones durante la limpieza", value=False
        )
    with columna_b:
        max_length = st.slider(
            "Límite didáctico de tokens BETO", 8, 128, 16, 1
        )
    if st.button("Analizar texto", type="primary", key="accion_analisis"):
        if not texto.strip():
            st.warning("Escribe un texto antes de analizarlo.")
            return
        nlp = obtener_modelo(cargar_spacy, "spaCy")
        beto = obtener_modelo(cargar_beto, "BETO")
        tabla = detalle_spacy(texto, nlp)
        tokens = tokenizar_beto(texto, beto, max_length)
        st.session_state["resultado_analisis"] = {
            "texto": texto,
            "tabla": tabla,
            "tokens": tokens,
            "limpio": limpiar(texto, nlp, conservar_negaciones),
            "palabras_split": len(texto.split()),
        }
    resultado = st.session_state.get("resultado_analisis")
    if resultado:
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Palabras con split()", resultado["palabras_split"])
        c2.metric("Tokens spaCy", len(resultado["tabla"]))
        c3.metric("Tokens BETO (incluye especiales)", resultado["tokens"]["n_tokens"])
        st.markdown("**Limpieza del texto**")
        st.code(resultado["limpio"] or "(sin tokens después de la limpieza)")
        st.markdown("**Detalle de spaCy**")
        st.dataframe(resultado["tabla"], use_container_width=True, hide_index=True)
        st.markdown("**Subpalabras e identificadores BETO**")
        st.write(resultado["tokens"]["tokens_originales"])
        with st.expander("Mostrar identificadores numéricos"):
            st.write(resultado["tokens"]["ids_originales"])
        if resultado["tokens"]["truncado"]:
            st.warning("El límite didáctico descarta parte del mensaje.")
            st.write("Tokens conservados:", resultado["tokens"]["tokens_conservados"])
        else:
            st.success("El texto cabe en el límite de tokens seleccionado.")
        st.caption(
            "BETO añade tokens especiales como [CLS] y [SEP]. "
            "La limpieza de stop words puede eliminar negaciones que cambian el sentido."
        )


def pagina_similitud():
    st.subheader("2. Comparador de similitud textual")
    st.write("Contrasta coincidencia de palabras y semejanza semántica.")
    a, b = st.columns(2)
    with a:
        texto_a = st.text_area(
            "Texto A", value="La clase de hoy me gustó mucho", height=110
        )
    with b:
        texto_b = st.text_area(
            "Texto B", value="No me gustó la clase de hoy", height=110
        )
    bigramas = st.checkbox("Incluir bigramas (pares de palabras) en Bag of Words")
    incluir_embeddings = st.checkbox(
        "Comparar también con embeddings semánticos", value=True
    )
    if st.button("Comparar textos", type="primary", key="accion_comparacion"):
        if not texto_a.strip() or not texto_b.strip():
            st.warning("Ingresa ambos textos.")
            return
        try:
            puntaje_bow, matriz = comparar_bow(texto_a, texto_b, bigramas)
        except ValueError as exc:
            st.error(str(exc))
            return
        resultado = {"bow": puntaje_bow, "matriz": matriz}
        if incluir_embeddings:
            modelo = obtener_modelo(cargar_embeddings, "Sentence Transformers")
            puntaje_emb, dimensiones = comparar_embeddings(texto_a, texto_b, modelo)
            resultado["embedding"] = puntaje_emb
            resultado["dimensiones"] = dimensiones
        st.session_state["resultado_comparacion"] = resultado
    resultado = st.session_state.get("resultado_comparacion")
    if resultado:
        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Similitud Bag of Words", f"{resultado['bow']:.3f}")
        if "embedding" in resultado:
            c2.metric("Similitud de embeddings", f"{resultado['embedding']:.3f}")
            st.caption(f"Dimensiones de cada embedding: {resultado['dimensiones']}")
        st.markdown("**Matriz de conteos de Bag of Words**")
        st.dataframe(resultado["matriz"], use_container_width=True)
        st.info(
            "Una similitud alta no implica que dos frases expresen el mismo "
            "sentimiento. Prueba agregar o quitar 'no'."
        )


def pagina_sentimiento():
    st.subheader("3. Clasificación de sentimientos")
    st.write(
        "Usa un modelo multilingüe que devuelve estrellas de 1 a 5. "
        "No reconoce el motivo administrativo de un mensaje."
    )
    texto = st.text_area(
        "Mensaje", value="La guía tiene errores en el código del paso 4", height=130
    )
    max_length = st.slider("Longitud máxima de entrada", 16, 256, 128, 8)
    if st.button("Clasificar sentimiento", type="primary", key="accion_sentimiento"):
        if not texto.strip():
            st.warning("Escribe un mensaje para clasificar.")
            return
        clasificador = obtener_modelo(cargar_clasificador, "clasificador de sentimientos")
        resultado = clasificar_sentimiento(texto, clasificador, max_length)
        st.session_state["resultado_sentimiento"] = resultado
    resultado = st.session_state.get("resultado_sentimiento")
    if resultado:
        st.divider()
        a, b, c = st.columns(3)
        a.metric("Etiqueta predicha", resultado["etiqueta"])
        b.metric("Puntuación del modelo", f"{resultado['confianza']:.1%}")
        c.metric("Tokens del clasificador", resultado["tokens_clasificador"])
        if resultado["truncado_clasificador"]:
            st.warning("El mensaje ha sido truncado para ajustarse al límite de entrada.")
        st.caption(
            "La puntuación mide la preferencia interna del modelo por esa etiqueta. "
            "No certifica que el sentimiento sea correcto. Este modelo fue "
            "entrenado con reseñas y sus etiquetas son estrellas."
        )


def pagina_lotes():
    st.subheader("4. Procesamiento de mensajes por lotes")
    st.write(
        "Carga un CSV UTF-8 con una columna de mensajes, o usa el archivo de ejemplo. "
        "Se procesan como máximo 50 registros por ejecución."
    )
    ejemplo = Path(__file__).parent / "ejemplos_mesa_ayuda.csv"
    if ejemplo.exists():
        st.download_button(
            "Descargar CSV de ejemplo",
            ejemplo.read_bytes(),
            file_name=ejemplo.name,
            mime="text/csv",
        )
    archivo = st.file_uploader("Selecciona un CSV", type=["csv"])
    if archivo is None:
        st.info("Carga un CSV para seleccionar la columna que contiene los mensajes.")
        return
    try:
        df = pd.read_csv(archivo, sep=None, engine="python", encoding="utf-8-sig")
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        st.error(f"No se pudo leer el CSV: {exc}")
        return
    if df.empty or not len(df.columns):
        st.warning("El archivo no contiene datos.")
        return
    columna = st.selectbox("Columna que contiene el texto", list(df.columns))
    max_length = st.slider(
        "Límite de tokens para los modelos", 16, 256, 128, 8, key="batch_limit"
    )
    mensajes = df[columna].dropna().astype(str).str.strip()
    mensajes = mensajes[mensajes.ne("")].head(50).tolist()
    st.write(f"Mensajes válidos que se analizarán: **{len(mensajes)}**")
    if st.button("Procesar mensajes", type="primary", key="accion_lote"):
        if not mensajes:
            st.warning("No hay mensajes para analizar en esa columna.")
            return
        nlp = obtener_modelo(cargar_spacy, "spaCy")
        beto = obtener_modelo(cargar_beto, "BETO")
        clasificador = obtener_modelo(cargar_clasificador, "clasificador de sentimientos")
        progreso = st.progress(0, text="Procesando mensajes…")
        filas = []
        for i, mensaje in enumerate(mensajes, start=1):
            filas.append(cadena(mensaje, nlp, beto, clasificador, max_length))
            progreso.progress(i / len(mensajes), text=f"Mensaje {i}/{len(mensajes)}")
        progreso.empty()
        st.session_state["resultado_lote"] = pd.DataFrame(filas)
    resultado = st.session_state.get("resultado_lote")
    if resultado is not None:
        st.divider()
        st.dataframe(resultado, use_container_width=True, hide_index=True)
        mostrar_descarga(resultado, "resultados_nlp.csv")
        st.caption(
            "truncado_beto se calcula con BETO, mientras que "
            "truncado_clasificador utiliza el tokenizador propio del modelo "
            "de sentimientos. Sus conteos pueden ser diferentes."
        )


st.title("NLP Lab")
st.caption("Encuentro 3 · Representación de textos y selección de modelos")
st.sidebar.title("Laboratorio NLP")
pagina = st.sidebar.radio(
    "Selecciona un módulo",
    ["Análisis de texto", "Similitud", "Sentimientos", "Procesamiento por lotes"],
)
st.sidebar.divider()
st.sidebar.caption(
    "Modelos preentrenados · Procesamiento local/servidor · "
    "No se entrenan modelos con los textos ingresados."
)

if pagina == "Análisis de texto":
    pagina_analisis()
elif pagina == "Similitud":
    pagina_similitud()
elif pagina == "Sentimientos":
    pagina_sentimiento()
else:
    pagina_lotes()

st.divider()
st.caption(
    "Proyecto didáctico. Los mensajes se procesan en el servidor donde se ejecuta "
    "Streamlit. Evita usar datos personales o sensibles sin autorización."
)
