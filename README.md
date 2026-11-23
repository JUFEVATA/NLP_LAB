# NLP Lab – Aplicación didáctica con Streamlit

Esta aplicación adapta el archivo `Encuentro_3_Laboratorio_version_compacta.ipynb` a una interfaz web. Permite reproducir los experimentos del encuentro 3 sobre tokenización, representación de texto y selección de modelos.

## Archivos

```text
nlp_lab_streamlit/
├── app.py                     # Interfaz y navegación Streamlit
├── nlp_utils.py               # Funciones de procesamiento de lenguaje
├── requirements.txt           # Bibliotecas y modelo spaCy en español
├── ejemplos_mesa_ayuda.csv     # Mensajes ficticios para pruebas
├── .streamlit/
│   └── config.toml            # Apariencia y límite de subida
├── .gitignore
└── README.md                  # Esta guía
```

## Requisitos

- Python 3.11 o 3.12 (recomendado: 3.11).
- Internet en la instalación y en el primer uso de cada modelo, porque los modelos públicos de Hugging Face se descargan la primera vez.
- Suficiente memoria RAM y espacio de disco para cargar modelos de lenguaje. Es preferible ejecutar localmente si el hosting gratuito dispone de pocos recursos.

## Cómo ejecutarlo en VS Code, Windows, Linux o macOS

1. Descomprime la carpeta y ábrela en VS Code.
2. Abre una terminal dentro de la carpeta (donde está `app.py`).
3. Crea un entorno virtual:

   ```bash
   python -m venv .venv
   ```

4. Actívalo:

   - **Windows / PowerShell:** `.venv\Scripts\Activate.ps1`
   - **Windows / CMD:** `.venv\Scripts\activate.bat`
   - **macOS / Linux:** `source .venv/bin/activate`

5. Instala los paquetes y el modelo de spaCy:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

6. Inicia la aplicación:

   ```bash
   python -m streamlit run app.py
   ```

7. Visita la dirección local indicada en la terminal (normalmente `http://localhost:8501`).

**Nota:** en algunos sistemas se usa `python3` o `py` en lugar de `python`. Selecciona el intérprete del entorno `.venv` en VS Code.

## Módulos

1. **Análisis de texto:** cuenta palabras, muestra tokens, lemas y stop words con spaCy; compara subpalabras e IDs de BETO y permite simular truncamiento.
2. **Similitud:** compara Bag of Words con o sin bigramas y embeddings de oraciones mediante similitud del coseno.
3. **Sentimientos:** utiliza el modelo `nlptown/bert-base-multilingual-uncased-sentiment`, que entrega etiquetas de 1 a 5 estrellas.
4. **Procesamiento por lotes:** carga un CSV, analiza hasta 50 mensajes y descarga los resultados como CSV.

## Modelos utilizados

| Función | Modelo |
|---|---|
| Tokenización, lemas, stop words | `es_core_news_sm` (spaCy) |
| Tokenización por subpalabras | `dccuchile/bert-base-spanish-wwm-cased` (BETO) |
| Embeddings de oraciones | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Sentimiento (1 a 5 estrellas) | `nlptown/bert-base-multilingual-uncased-sentiment` |

Los modelos se cargan **solo cuando se necesitan** y Streamlit conserva los recursos en caché para reutilizarlos. La primera ejecución de cada módulo puede necesitar una descarga.

## Cómo probar

- En **Análisis de texto**, usa «No me gustó la clase de hoy» y comprueba cómo cambia la limpieza al conservar la negación.
- En **Similitud**, compara «La clase de hoy me gustó mucho» con «No me gustó la clase de hoy». Activa los bigramas y observa cómo cambia la similitud.
- En **Sentimientos**, utiliza «Quisiera cambiar de grupo al de los martes» y evalúa si una etiqueta de estrellas responde realmente al motivo de consulta.
- En **Procesamiento por lotes**, descarga y vuelve a subir `ejemplos_mesa_ayuda.csv`.

## Publicación opcional

Para publicar la aplicación en Streamlit Community Cloud:

1. Sube esta carpeta a un repositorio de GitHub (lo más sencillo es que `app.py` y `requirements.txt` estén en la raíz del repositorio).
2. En https://share.streamlit.io/ inicia sesión y crea una aplicación a partir de ese repositorio.
3. Usa `app.py` como archivo de entrada y selecciona Python 3.11 si la plataforma ofrece esa elección.
4. El servidor instalará las dependencias descritas en `requirements.txt`. Necesita acceso a GitHub para descargar `es_core_news_sm` y a Hugging Face para obtener los otros modelos.

**Limitación de recursos:** los modelos Transformer pueden superar los límites de memoria o disco en hostings gratuitos. Si ocurre, utiliza ejecución local, un servidor con mayor capacidad o deshabilita los módulos que no necesitas.

## Alcance y precauciones

- La aplicación **no entrena** modelos nuevos; trabaja con modelos preentrenados.
- El modelo de sentimiento califica de 1 a 5 estrellas y **no identifica motivos administrativos** como «plataforma», «trámite» o «docente».
- Una similitud alta no significa que dos mensajes tengan el mismo sentimiento.
- `tokens_beto` y `tokens_clasificador` utilizan **tokenizadores diferentes**. Por tanto, los indicadores de truncamiento no tienen por qué coincidir.
- Los textos que ingresas se procesan en el equipo o servidor donde se ejecuta la aplicación. No subas información sensible sin autorización.

## Solución de problemas

- **`No module named streamlit`:** activa `.venv` e instala `requirements.txt`.
- **`Can't find model 'es_core_news_sm'`:** comprueba que se instaló la última línea de `requirements.txt`. Como alternativa ejecuta `python -m spacy download es_core_news_sm`.
- **Error al descargar Hugging Face:** verifica Internet, proxy, permisos de red y espacio de almacenamiento.
- **Sin memoria:** reinicia la aplicación y utiliza los módulos por separado o ejecútala en una máquina con más RAM.
