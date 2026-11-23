"""Funciones de PLN utilizadas por la interfaz Streamlit.

Basado en 'Encuentro_3_Laboratorio_version_compacta.ipynb'.
Los modelos se cargan en app.py y se pasan a estas funciones.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


NEGACIONES = {"no", "ni", "nunca", "nada", "tampoco", "jamás"}


def detalle_spacy(texto: str, nlp) -> pd.DataFrame:
    """Devuelve tokens, lemas, categorías gramaticales y stop words."""
    doc = nlp(texto)
    return pd.DataFrame(
        [
            {
                "Token": t.text,
                "Lema": t.lemma_,
                "Clase gramatical": t.pos_,
                "Stop word": bool(t.is_stop),
                "Puntuación": bool(t.is_punct),
            }
            for t in doc
        ],
        columns=["Token", "Lema", "Clase gramatical", "Stop word", "Puntuación"],
    )


def limpiar(texto: str, nlp, conservar_negaciones: bool = False) -> str:
    """Elimina signos y stop words; opcionalmente protege las negaciones."""
    lemas = []
    for t in nlp(texto):
        if t.is_punct or t.is_space:
            continue
        if t.is_stop and not (conservar_negaciones and t.lower_ in NEGACIONES):
            continue
        lemas.append(t.lemma_.lower())
    return " ".join(lemas)


def tokenizar_beto(texto: str, tokenizador, max_length: int) -> dict:
    """Obtiene subpalabras, IDs y un ejemplo de truncamiento con BETO."""
    original = tokenizador(texto, truncation=False, add_special_tokens=True)
    limitado = tokenizador(
        texto, truncation=True, max_length=max_length, add_special_tokens=True
    )
    ids = original["input_ids"]
    ids_limitados = limitado["input_ids"]
    return {
        "tokens_originales": tokenizador.convert_ids_to_tokens(ids),
        "ids_originales": ids,
        "tokens_conservados": tokenizador.convert_ids_to_tokens(ids_limitados),
        "ids_conservados": ids_limitados,
        "n_tokens": len(ids),
        "truncado": len(ids) > max_length,
    }


def comparar_bow(texto_a: str, texto_b: str, incluir_bigramas: bool = False) -> tuple[float, pd.DataFrame]:
    """Devuelve similitud coseno y tabla de frecuencias de dos textos."""
    rango = (1, 2) if incluir_bigramas else (1, 1)
    vectorizador = CountVectorizer(ngram_range=rango)
    try:
        matriz = vectorizador.fit_transform([texto_a, texto_b])
    except ValueError as exc:
        raise ValueError(
            "No se pudieron construir vectores. Escribe palabras de al menos "
            "dos caracteres en ambos textos."
        ) from exc
    similitud = float(cosine_similarity(matriz)[0, 1])
    tabla = pd.DataFrame(
        matriz.toarray(),
        columns=vectorizador.get_feature_names_out(),
        index=["Texto A", "Texto B"],
    )
    return similitud, tabla


def comparar_embeddings(texto_a: str, texto_b: str, modelo) -> tuple[float, int]:
    """Devuelve similitud coseno y dimensiones del embedding."""
    vectores = modelo.encode([texto_a, texto_b], show_progress_bar=False)
    similitud = float(cosine_similarity(vectores)[0, 1])
    return similitud, int(vectores.shape[1])


def clasificar_sentimiento(texto: str, clasificador, max_length: int = 128) -> dict:
    """Predice estrellas y contabiliza tokens del modelo de sentimientos.

    No asume que el tokenizador del modelo de sentimientos sea BETO.
    """
    num_tokens = len(
        clasificador.tokenizer(texto, truncation=False, add_special_tokens=True)["input_ids"]
    )
    prediccion = clasificador(
        texto,
        truncation=True,
        max_length=max_length,
    )[0]
    return {
        "etiqueta": prediccion["label"],
        "confianza": float(prediccion["score"]),
        "tokens_clasificador": num_tokens,
        "truncado_clasificador": num_tokens > max_length,
    }


def cadena(texto: str, nlp, tokenizador_beto, clasificador, max_length: int = 128) -> dict:
    """Replica la cadena de inferencia del notebook y añade precisión diagnóstica."""
    ids_beto = tokenizador_beto(texto, truncation=False)["input_ids"]
    sentimiento = clasificar_sentimiento(texto, clasificador, max_length)
    return {
        "texto": texto,
        "palabras_split": len(texto.split()),
        "tokens_spacy": len(nlp(texto)),
        "tokens_beto": len(ids_beto),
        "truncado_beto": len(ids_beto) > max_length,
        "limpio_sin_stopwords": limpiar(texto, nlp),
        "etiqueta": sentimiento["etiqueta"],
        "confianza": round(sentimiento["confianza"], 3),
        "tokens_clasificador": sentimiento["tokens_clasificador"],
        "truncado_clasificador": sentimiento["truncado_clasificador"],
    }
