"""NLP Lab — frontend Streamlit con la paleta azul/gris de la referencia.

Mantiene el mismo núcleo funcional del Encuentro 3:
frase → tokenización del clasificador → IDs → cinco puntuaciones.
"""
from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from nlp_core import analizar_frase
from ui_components import barras_puntuaciones, fichas_tokens, tarjeta_resultado


MODELO_SENTIMIENTO = "nlptown/bert-base-multilingual-uncased-sentiment"
MAX_TOKENS = 128

st.set_page_config(
    page_title="NLP Lab | Tokens y predicción",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hoja CSS local incluida en el repositorio (sin dependencias de CDN).
estilos = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
st.markdown(f"<style>{estilos}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Cargando modelo de español...")
def cargar_spacy():
    import spacy

    return spacy.load("es_core_news_sm")


@st.cache_resource(show_spinner="Descargando o cargando el clasificador de sentimientos...")
def cargar_clasificador():
    import torch
    from transformers import pipeline

    torch.set_num_threads(2)
    return pipeline(
        task="text-classification",
        model=MODELO_SENTIMIENTO,
        device=-1,  # CPU en Streamlit Community Cloud
    )


def titulo_seccion(numero: str, titulo: str, descripcion: str = ""):
    st.markdown(
        '<div class="section-heading">'
        f'<span class="section-index">{escape(numero)}</span>'
        f'<h2 class="section-title">{escape(titulo)}</h2>'
        '</div>',
        unsafe_allow_html=True,
    )
    if descripcion:
        st.markdown(
            f'<div class="section-subtitle">{escape(descripcion)}</div>',
            unsafe_allow_html=True,
        )


st.markdown(
    """
    <header class="nlp-hero">
      <div class="nlp-brandline">
        <div class="nlp-brand"><span class="nlp-brandmark">N</span><span>NLP LAB</span></div>
        <span class="nlp-edition">ENCUENTRO 03 · MÓDULO M2</span>
      </div>
      <h1>Del lenguaje a una<br><span class="accent">predicción.</span></h1>
      <p>Descubre cómo un modelo de inteligencia artificial transforma una frase en
      subpalabras, identificadores numéricos y una clasificación de sentimiento.</p>
      <div class="nlp-pipeline" aria-label="Flujo de procesamiento">
        <span class="step">Frase</span><span class="arrow">→</span>
        <span class="step">Tokens</span><span class="arrow">→</span>
        <span class="step">IDs</span><span class="arrow">→</span>
        <span class="step">Transformer</span><span class="arrow">→</span>
        <span class="step">Predicción</span>
      </div>
    </header>
    """,
    unsafe_allow_html=True,
)

titulo_seccion("01", "Escribe una frase", "Analizaremos exactamente el texto que ingreses, sin eliminar negaciones ni palabras frecuentes.")

with st.form("formulario_frase"):
    frase = st.text_area(
        "Tu texto",
        value="Me gustan mucho las clases con el profesor Juan Felipe",
        max_chars=4000,
        height=120,
        placeholder="Ej.: No me gustó la clase de hoy...",
        help="El clasificador recibirá el texto original. Máximo 4.000 caracteres.",
    )
    analizar = st.form_submit_button(
        "Analizar frase  →",
        type="primary",
        width="stretch",
    )

if analizar:
    if not frase.strip():
        st.session_state.pop("nlp_salida", None)
        st.warning("Escribe una frase para iniciar el análisis.")
    else:
        try:
            with st.spinner("Procesando la frase con spaCy y el modelo Transformer..."):
                modelo_spacy = cargar_spacy()
                clasificador = cargar_clasificador()
                resultado = analizar_frase(frase, modelo_spacy, clasificador, MAX_TOKENS)
            st.session_state["nlp_salida"] = resultado
        except Exception as error:
            st.session_state.pop("nlp_salida", None)
            st.error("No se pudo completar el análisis. Revisa los registros de Streamlit Cloud.")
            with st.expander("Detalles técnicos"):
                st.exception(error)

salida = st.session_state.get("nlp_salida")
if salida:
    st.markdown(
        '<div class="result-context">'
        '<div class="eyebrow">FRASE ANALIZADA</div>'
        f'<div class="phrase">{escape(salida["texto"])}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    titulo_seccion("02", "Predicción del modelo", "Resultado de sentimiento y distribución de las cinco etiquetas posibles.")
    tarjeta_clase, tarjeta_score = tarjeta_resultado(salida["etiqueta"], salida["puntuacion"])
    col_etiqueta, col_score = st.columns(2, gap="medium")
    with col_etiqueta:
        st.markdown(tarjeta_clase, unsafe_allow_html=True)
    with col_score:
        st.markdown(tarjeta_score, unsafe_allow_html=True)

    st.markdown("#### Distribución de puntuaciones")
    st.markdown(barras_puntuaciones(salida["distribucion"]), unsafe_allow_html=True)
    st.caption(
        "El modelo produce cinco puntuaciones que suman aproximadamente 100 %. "
        "La más alta determina la etiqueta ganadora. No representan precisión "
        "validada ni el porcentaje de positividad del texto."
    )
    with st.expander("Consultar los cinco valores exactos"):
        datos = pd.DataFrame(
            [
                {
                    "Etiqueta": fila["etiqueta"],
                    "Puntuación (%)": round(100 * fila["puntuacion"], 3),
                }
                for fila in salida["distribucion"]
            ]
        )
        st.dataframe(datos, hide_index=True, width="stretch")

    titulo_seccion(
        "03",
        "Tokens del clasificador",
        "Cada ficha representa un token real utilizado por el modelo. Debajo aparece su ID numérico.",
    )
    st.markdown(fichas_tokens(salida["tokens_modelo"]), unsafe_allow_html=True)
    st.caption(
        f'Tokens enviados al modelo (incluidos los especiales): {salida["total_modelo"]}. '
        f'Tokens antes del truncamiento: {salida["total_original_modelo"]}.'
    )
    if salida["truncado"]:
        st.warning(
            f'La secuencia superó {MAX_TOKENS} tokens y fue truncada antes de hacer la predicción. '
            "La parte descartada no fue considerada por el modelo."
        )
    with st.expander("Tabla completa: posición, token e ID"):
        st.dataframe(
            pd.DataFrame(salida["tokens_modelo"]),
            hide_index=True,
            width="stretch",
        )
    with st.expander("Comparación educativa: tokenización con spaCy"):
        st.write(
            "spaCy detecta tokens, lemas, palabras frecuentes y puntuación. "
            "Estos tokens **no son los que recibe el clasificador**. "
            "El clasificador usa su propio tokenizador WordPiece."
        )
        st.dataframe(
            pd.DataFrame(salida["tokens_spacy"]),
            hide_index=True,
            width="stretch",
        )

    titulo_seccion("04", "¿Cómo llega el modelo a una predicción?", "Tres operaciones distintas que se conectan entre sí, pero no son equivalentes.")
    st.markdown(
        """
        <div class="learning-grid">
          <article class="learning-card">
            <div class="learning-num">PASO 01</div>
            <div class="learning-title">Tokenización</div>
            <div class="learning-copy">Divide la frase en piezas. Una palabra como «gusta» puede representarse mediante «gus» y «##ta».</div>
          </article>
          <article class="learning-card">
            <div class="learning-num">PASO 02</div>
            <div class="learning-title">Representación</div>
            <div class="learning-copy">Cada token recibe un ID, que el Transformer transforma en vectores numéricos (embeddings).</div>
          </article>
          <article class="learning-card">
            <div class="learning-num">PASO 03</div>
            <div class="learning-title">Inferencia</div>
            <div class="learning-copy">El Transformer utiliza el contexto y calcula cinco puntuaciones. Se muestra la etiqueta con el valor mayor.</div>
          </article>
        </div>
        <div class="learning-note"><strong>Idea clave:</strong> dividir una oración en tokens no significa comprenderla.
        La predicción depende de los parámetros aprendidos por el modelo y puede ser incorrecta.</div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("¿De qué parte del notebook sale cada funcionalidad?"):
        st.markdown(
            """
            | Notebook Encuentro 3 | Aplicación Streamlit |
            |---|---|
            | Tokenización con spaCy (celdas 8–9) | Tabla comparativa opcional de tokens y lemas |
            | BETO (celdas 20–22) | Solo referencia conceptual: **no** se utiliza como tokenizador del clasificador |
            | Clasificación Hugging Face (celda 28) | Modelo que entrega las cinco puntuaciones |
            | Función `cadena()` (celda 31) | Inspiración para `analizar_frase()` en `nlp_core.py` |
            """
        )

st.markdown(
    '<footer class="nlp-footer">'
    'NLP LAB · Procesamiento de lenguaje natural · Entorno educativo<br>'
    'Modelo: nlptown/bert-base-multilingual-uncased-sentiment'
    '</footer>',
    unsafe_allow_html=True,
)
