"""Interfaz web para el laboratorio: una frase, sus tokens y una predicción.

Streamlit Community Cloud ejecuta `app.py` desde el repositorio GitHub.
No es necesario subir el modelo a GitHub: Hugging Face lo descarga al
inicializarlo por primera vez (se requiere acceso a Internet).
"""

import streamlit as st

from nlp_core import analizar_frase


MODELO_SENTIMIENTO = "nlptown/bert-base-multilingual-uncased-sentiment"
MAX_TOKENS = 128

st.set_page_config(page_title="NLP Lab · Tokens y predicción", page_icon="🔤", layout="centered")


@st.cache_resource(show_spinner="Cargando spaCy en español...")
def cargar_spacy():
    import spacy

    return spacy.load("es_core_news_sm")


@st.cache_resource(show_spinner="Cargando clasificador de sentimientos...")
def cargar_clasificador():
    import torch
    from transformers import pipeline

    # Comunidad Streamlit: usamos solo CPU y reutilizamos el mismo modelo.
    torch.set_num_threads(2)
    return pipeline(
        task="text-classification",
        model=MODELO_SENTIMIENTO,
        device=-1,
    )


st.title("NLP Lab: de una frase a una predicción")
st.write(
    "Escribe una frase. La app la dividirá en tokens con spaCy y mostrará "
    "los tokens que utiliza el clasificador de Hugging Face para predecir "
    "una calificación de **1 a 5 estrellas**."
)

with st.form("formulario_frase"):
    frase = st.text_area(
        "Frase para analizar",
        value="No me gustó la clase de hoy",
        max_chars=4000,
        height=110,
        help="Escribe una frase en español; el texto se enviará al modelo sin quitar negaciones.",
    )
    analizar = st.form_submit_button("Mostrar tokens y predicción", type="primary")

if analizar:
    if not frase.strip():
        st.warning("Ingresa una frase antes de analizar.")
    else:
        try:
            with st.spinner("Analizando frase..."):
                nlp = cargar_spacy()
                clasificador = cargar_clasificador()
                salida = analizar_frase(frase, nlp, clasificador, MAX_TOKENS)
        except Exception as exc:
            st.error("El análisis no pudo completarse. Revisa los logs del despliegue y las dependencias.")
            with st.expander("Detalle técnico del error"):
                st.exception(exc)
        else:
            st.divider()
            st.subheader("1. Predicción del modelo")
            col1, col2 = st.columns(2)
            col1.metric("Etiqueta predicha", salida["etiqueta"])
            col2.metric("Puntuación de la etiqueta", f'{salida["puntuacion"]:.1%}')
            st.caption(
                "El modelo está ajustado con reseñas de productos: predice 1–5 estrellas. "
                "Su puntuación no garantiza una interpretación correcta de una frase educativa."
            )

            st.subheader("2. Tokens utilizados por el clasificador")
            st.write([fila["Token del modelo"] for fila in salida["tokens_modelo"]])
            st.dataframe(
                salida["tokens_modelo"],
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                f'Tokens realmente enviados al modelo (incluidos especiales): {salida["total_modelo"]}. '
                f'Tokens antes del truncamiento: {salida["total_original_modelo"]}.'
            )
            if salida["truncado"]:
                st.warning(
                    f'La frase supera {MAX_TOKENS} tokens. El modelo recibió solo los primeros '
                    "tokens, respetando sus tokens especiales."
                )

            with st.expander("3. Comparar con la tokenización de spaCy (notebook)"):
                st.write([fila["Token"] for fila in salida["tokens_spacy"]])
                st.dataframe(salida["tokens_spacy"], hide_index=True, use_container_width=True)
                st.caption(
                    "spaCy separa palabras y puntuación; el modelo Transformer maneja "
                    "subpalabras e IDs. Son dos representaciones distintas."
                )

st.divider()
st.caption(
    "Basado en Encuentro 3 · Representación de texto y selección de modelos. "
    "El tokenizador BETO del notebook es un experimento aparte: aquí se "
    "visualizan los tokens del modelo que realmente predice las estrellas."
)
