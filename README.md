# NLP Lab · Frontend azul pizarra

Aplicación didáctica de **procesamiento de lenguaje natural** con Streamlit. El diseño sigue exactamente la paleta de seis colores compartida en la referencia:

| Uso | Hex |
|---|---|
| Fondo oscuro | `#06141B` |
| Panel principal | `#11212D` |
| Superficies | `#253745` |
| Bordes y barras secundarias | `#4A5C6A` |
| Texto secundario | `#9BA8AB` |
| Texto principal y barras activas | `#CCD0CF` |

El usuario escribe una frase y obtiene la **predicción 1–5 estrellas**, la distribución completa de puntuaciones, tokens reales con sus IDs, un análisis adicional con spaCy y una explicación educativa del proceso.

## ¿Qué archivo hace qué?

- `app.py`: interfaz de Streamlit, formularios y componentes de resultados.
- `styles.css`: diseño responsivo. Modifica aquí sombras, espaciados, colores y tamaños.
- `ui_components.py`: tarjetas de métricas, barras HTML y fichas de tokens.
- `nlp_core.py`: tokenización, análisis con spaCy e inferencia con Hugging Face.
- `.streamlit/config.toml`: tema oscuro nativo de Streamlit y los seis colores.
- `requirements.txt`: dependencias de la app y modelo spaCy.
- `NLP_Lab.ipynb`: notebook didáctico de referencia para Google Colab.
- `tests/`: pruebas de procesamiento de texto y de componentes visuales.

## Publicación en Streamlit Community Cloud

1. Descomprime el ZIP y sube **todos los archivos y carpetas** a la raíz de tu repositorio GitHub (sin incluir una carpeta contenedora adicional).
2. En [Streamlit Community Cloud](https://share.streamlit.io/), crea una app desde ese repositorio.
3. Selecciona `main` como rama y `app.py` como *Main file path*.
4. En opciones avanzadas selecciona **Python 3.12**.
5. Publica la app; el primer análisis descargará y cargará el modelo de sentimientos.

No se requiere HTML externo, JavaScript ni servicio API. La interfaz se construye con Streamlit y CSS local.

## Pruebas

Sin descargar los modelos de Hugging Face puedes verificar la lógica con:

```bash
python -m unittest discover -s tests -v
```

Estas pruebas no sustituyen un despliegue real en Community Cloud. Si el servidor gratuito agota la RAM o no puede descargar un modelo, revisa los logs del despliegue.

## Limitación importante

El modelo `nlptown/bert-base-multilingual-uncased-sentiment` fue entrenado para reseñas de productos y devuelve una etiqueta de **1 a 5 estrellas**. No está ajustado específicamente para evaluar experiencias educativas. Sus puntuaciones internas **no son estimaciones calibradas de exactitud o certeza**.
