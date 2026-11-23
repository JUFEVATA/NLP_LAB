"""Componentes HTML de presentación; sin lógica NLP ni dependencias de Streamlit."""
from __future__ import annotations

from html import escape
import math
import re


STAR_PATTERN = re.compile(r"^([1-5]) stars?$", flags=re.IGNORECASE)


def _score_percent(score: float) -> float:
    """Restringe a 0-100 y evita inyectar valores no finitos en CSS."""
    n = float(score)
    if not math.isfinite(n):
        raise ValueError("La puntuación debe ser un número finito.")
    return max(0.0, min(100.0, n * 100))


def tarjeta_resultado(etiqueta: str, puntuacion: float) -> tuple[str, str]:
    """Tarjeta con la clase ganadora y tarjeta con la puntuación de la etiqueta."""
    encontrado = STAR_PATTERN.fullmatch(str(etiqueta).strip())
    if encontrado is None:
        raise ValueError(f"Etiqueta no reconocida: {etiqueta}")
    estrellas = int(encontrado.group(1))
    porcentaje = _score_percent(puntuacion)
    tarjeta_etiqueta = f"""
      <div class="metric-card">
        <div class="metric-caption">ETIQUETA PREDICHA</div>
        <div class="metric-number">{estrellas}<span class="metric-denominator"> / 5</span></div>
        <div class="metric-footer">{escape(str(etiqueta))} · escala de sentimiento</div>
        <span class="metric-decoration" aria-hidden="true">✦</span>
      </div>"""
    tarjeta_puntuacion = f"""
      <div class="metric-card metric-card-confidence">
        <div class="metric-caption">PUNTUACIÓN DE LA ETIQUETA</div>
        <div class="metric-number">{porcentaje:.1f}<span class="metric-denominator">%</span></div>
        <div class="meter" aria-label="Puntuación de etiqueta: {porcentaje:.1f} por ciento"><div style="width: {porcentaje:.2f}%;"></div></div>
        <div class="metric-footer">Puntuación interna · no equivale a precisión</div>
      </div>"""
    return tarjeta_etiqueta, tarjeta_puntuacion


def barras_puntuaciones(distribucion: list[dict]) -> str:
    """Cinco barras HTML accesibles, sin JavaScript, basadas en datos reales."""
    if not distribucion:
        return '<div class="empty-state">No hay puntuaciones disponibles.</div>'
    ganadora = max(distribucion, key=lambda x: x["puntuacion"])["estrellas"]
    filas = []
    for fila in sorted(distribucion, key=lambda x: x["estrellas"]):
        numero = int(fila["estrellas"])
        if numero not in range(1, 6):
            raise ValueError("Solo se admiten etiquetas de una a cinco estrellas.")
        porcentaje = _score_percent(fila["puntuacion"])
        activo = " result-winning" if numero == ganadora else ""
        filas.append(f"""
          <div class="score-row{activo}">
            <div class="score-label">{numero} <span aria-hidden="true">★</span></div>
            <div class="score-track" role="progressbar" aria-label="{numero} estrellas" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{porcentaje:.2f}">
              <div class="score-fill" style="width: {porcentaje:.2f}%;"></div>
            </div>
            <div class="score-value">{porcentaje:.1f}%</div>
          </div>""")
    return '<div class="score-chart">' + "\n".join(filas) + '</div>'


def fichas_tokens(tokens: list[dict], max_visibles: int = 32) -> str:
    """Representación visual de tokens/IDs; escapa contenido para HTML seguro."""
    chips = []
    for token in tokens[:max_visibles]:
        contenido = escape(str(token["Token del modelo"]))
        posicion = int(token["Posición"])
        identificador = int(token["ID"])
        especial = " token-special" if str(token["Token del modelo"]).startswith("[") else ""
        chips.append(f"""
          <div class="token-chip{especial}" title="Posición {posicion} · ID {identificador}">
            <span class="chip-text">{contenido}</span>
            <span class="chip-id">#{identificador}</span>
          </div>""")
    cantidad_restante = len(tokens) - min(len(tokens), max_visibles)
    if cantidad_restante:
        chips.append(f'<div class="token-extra">+ {cantidad_restante} tokens más en la tabla</div>')
    return '<div class="token-cloud">' + "\n".join(chips) + '</div>'
