"""Pruebas rápidas de componentes HTML (sin Streamlit ni modelos externos)."""
import unittest
from ui_components import barras_puntuaciones, fichas_tokens, tarjeta_resultado


class TestUiComponents(unittest.TestCase):
    def test_metricas(self):
        cards = tarjeta_resultado("4 stars", 0.468)
        self.assertIn("46.8", cards[1])
        self.assertIn("4", cards[0])

    def test_cinco_barras_resaltan_ganador(self):
        datos = [{"estrellas": n, "puntuacion": n / 15} for n in range(1, 6)]
        html = barras_puntuaciones(datos)
        self.assertEqual(html.count('class="score-row'), 5)
        self.assertEqual(html.count("result-winning"), 1)
        self.assertIn('aria-label="5 estrellas"', html)

    def test_escapa_tokens(self):
        contenido = fichas_tokens([{"Posición": 0, "Token del modelo": '<script>alert(1)</script>', "ID": 101}])
        self.assertNotIn("<script>", contenido)
        self.assertIn("&lt;script&gt;", contenido)

    def test_muestra_total_omitido(self):
        tokens = [{"Posición": i, "Token del modelo": "t", "ID": i} for i in range(5)]
        self.assertIn("+ 3 tokens más", fichas_tokens(tokens, max_visibles=2))

    def test_rechaza_puntaje_no_finito(self):
        with self.assertRaises(ValueError):
            tarjeta_resultado("4 stars", float("nan"))


if __name__ == "__main__":
    unittest.main()
