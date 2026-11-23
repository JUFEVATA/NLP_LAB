"""Funciones de inferencia extraídas del laboratorio de NLP.

Notebook de referencia:
- Celdas 8-9: spaCy (tokens, lemas y stop words).
- Celda 28: clasificador de sentimientos de Hugging Face.
- Celda 31: integración de ambos en una función de inferencia.

Importante: los tokens de BETO de la celda 20 pertenecen a otro
modelo distinto del clasificador de sentimientos de la celda 28.
"""

from __future__ import annotations


def analizar_frase(texto: str, nlp, clasificador, max_length: int = 128) -> dict:
    """Tokeniza el texto sin limpiarlo y ejecuta la predicción.

    Retorna los tokens de spaCy (descripción lingüística), los tokens reales
    del modelo de sentimientos (subpalabras + IDs) y la etiqueta predicha.

    `score` es la puntuación del modelo para su etiqueta elegida, no una
    garantía estadística de que la predicción sea correcta.
    """
    texto = texto.strip()
    if not texto:
        raise ValueError("Debes ingresar una frase no vacía.")
    if max_length < 3:
        raise ValueError("max_length debe ser al menos 3.")

    # Nivel 2: tokenización lingüística del notebook, con spaCy.
    doc = nlp(texto)
    tokens_spacy = [
        {
            "Token": token.text,
            "Lema": token.lemma_,
            "Stop word": bool(token.is_stop),
            "Puntuación": bool(token.is_punct),
        }
        for token in doc
    ]

    # Cada modelo Transformer tiene su propio tokenizador. La aplicación
    # muestra el del CLASIFICADOR, no el de BETO del experimento didáctico.
    tokenizer = clasificador.tokenizer
    ids_originales = tokenizer(texto, truncation=False, add_special_tokens=True)["input_ids"]
    ids_usados = tokenizer(
        texto, truncation=True, max_length=max_length, add_special_tokens=True
    )["input_ids"]
    tokens_modelo = tokenizer.convert_ids_to_tokens(ids_usados)
    tabla_tokens = [
        {"Posición": i, "Token del modelo": token, "ID": int(token_id)}
        for i, (token, token_id) in enumerate(zip(tokens_modelo, ids_usados))
    ]

    # Nivel de selección de modelos y nivel 4 del notebook: predicción real.
    # El texto original se conserva: eliminar 'no' puede cambiar el sentido.
    prediccion = clasificador(
        texto, truncation=True, max_length=max_length
    )[0]

    return {
        "texto": texto,
        "tokens_spacy": tokens_spacy,
        "tokens_modelo": tabla_tokens,
        "total_spacy": len(tokens_spacy),
        "total_modelo": len(ids_usados),
        "total_original_modelo": len(ids_originales),
        "truncado": len(ids_originales) > max_length,
        "etiqueta": str(prediccion["label"]),
        "puntuacion": float(prediccion["score"]),
    }
