"""Cores dos cubos usadas pelo benchmark.

As faixas HSV correspondentes ficam em vision/color_detect.py (COLOR_RANGES);
as duas tabelas precisam andar juntas. As cores novas (verde, amarelo, roxo)
foram escolhidas longe do matiz da mesa (marrom, H~15 no OpenCV, S~139) e do
azul-claro do punho da RG2, mas so valem depois de passar no portao de
percepcao (scripts/bench_check_perception.py).
"""

# rgb: cor difusa no CoppeliaSim (0-1). pt/en: adjetivo usado nas instrucoes
COLORS = {
    "red": {"rgb": [0.80, 0.05, 0.05], "pt": "vermelho", "en": "red"},
    "blue": {"rgb": [0.05, 0.05, 0.85], "pt": "azul", "en": "blue"},
    "green": {"rgb": [0.05, 0.70, 0.05], "pt": "verde", "en": "green"},
    "yellow": {"rgb": [0.85, 0.75, 0.05], "pt": "amarelo", "en": "yellow"},
    "purple": {"rgb": [0.50, 0.05, 0.80], "pt": "roxo", "en": "purple"},
}

PALETTE = list(COLORS)


def color_word(color: str, lang: str) -> str:
    return COLORS[color][lang]


def cube_alias(color: str) -> str:
    """Alias do cubo na cena do CoppeliaSim (o mesmo padrao de /red_cube)."""
    return f"/{color}_cube"
