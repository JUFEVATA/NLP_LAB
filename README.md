# NLP Lab: tokens y predicción

Aplicación educativa con Streamlit basada en el notebook **Encuentro 3 · Representación de texto y selección de modelos**. Introduce una frase en español y observa sus **tokens** y la **predicción de sentimientos (1–5 estrellas)**.

## ¿Qué parte del notebook se utiliza?

| Notebook | `nlp_lab_frase_streamlit` | Para qué sirve |
|---|---|---|
| Celdas 8–9 (`nlp`, `doc`, `t.text`, `t.lemma_`) | `cargar_spacy()` y `analizar_frase()` | Mostrar los tokens lingüísticos de spaCy |
| Celdas 20–22 (`AutoTokenizer` de BETO) | No se carga BETO | Son experimentos de tokenización, **no** el modelo que predice el sentimiento |
| Celda 28 (`pipeline(...sentiment)`) | `cargar_clasificador()` | Cargar el modelo preentrenado que asigna estrellas |
| Celda 31 (`cadena()`) | `analizar_frase()` | Encadenar texto → tokens → predicción |
| Celdas 14–18 y 25 (BoW, embeddings) | Excluidas | No son necesarias para una frase y su predicción |

**Importante:** la app muestra los tokens del tokenizador que **sí recibe el modelo de sentimientos**: `clasificador.tokenizer`. Es diferente al tokenizador BETO utilizado para el ejercicio didáctico. Por ello, se presentan dos vistas: tokens del clasificador y tokens de spaCy. No se eliminan negaciones ni palabras vacías antes de predecir.

## Estructura para GitHub

```text
nlp-lab-streamlit/
├── app.py                  # Interfaz: recibe frase y muestra resultados
├── nlp_core.py             # Procesamiento e inferencia de NLP
├── requirements.txt        # Paquetes y modelo spaCy
├── README.md               # Esta guía
├── .gitignore
└── .streamlit/
    └── config.toml         # Tema visual
```

En GitHub no es necesario subir los pesos del modelo de Hugging Face ni el notebook para que funcione la aplicación. El notebook se puede adjuntar como recurso pedagógico opcional.

## Publicar directamente con GitHub y Streamlit Community Cloud

1. Crea o abre tu repositorio `nlp-lab-streamlit` en GitHub.
2. Sube **los archivos descomprimidos** a la raíz del repositorio; no subas solo el ZIP.
3. En <https://share.streamlit.io/> selecciona crear aplicación desde GitHub.
4. Selecciona tu repositorio, rama `main` y archivo principal `app.py`.
5. En **Advanced settings** elige **Python 3.12**.
6. Haz clic en **Deploy**. El servicio instalará las dependencias de `requirements.txt` y descargará el modelo de Hugging Face en su primera ejecución.

Se fija **Streamlit 1.67.0** en `requirements.txt`. No es necesario indicar la versión en la interfaz de despliegue.

**Nota sobre límites:** el modelo de sentimientos es BERT multilingüe y consume más recursos que un ejemplo de Streamlit sin IA. La primera inferencia descargará archivos relativamente grandes; si la instancia se queda sin RAM o falla la conexión a Hugging Face, los logs de Streamlit indicarán el error. La ejecución con todos los modelos no fue verificada dentro de un servidor Community Cloud.

## Ejemplo de uso

Escribe `No me gustó la clase de hoy` y presiona **Mostrar tokens y predicción**. Verás:

- Tokens del modelo Transformer, con sus IDs.
- Tokens y lemas identificados por spaCy.
- Etiqueta `1 star`, `2 stars`…`5 stars` según lo que el modelo produzca, y su puntuación.

La predicción puede variar según la versión y el modelo y **no** se debe confundir la puntuación con exactitud real. Este modelo fue entrenado sobre reseñas de productos, no sobre solicitudes de estudiantes.
