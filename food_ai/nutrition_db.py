"""
Shared nutrition database for all 3 models.
Each entry: [calories_per_100g, protein_g, fat_g, carbs_g]
Typical portion sizes in grams.
"""

NUTRITION_DB = {
    # ── Indian dishes ──────────────────────────────────────────────────────
    "biryani":              [150, 6.0,  5.0,  22.0],
    "butter chicken":       [150, 12.0, 9.0,  5.0],
    "paneer butter masala": [180, 10.0, 12.0, 8.0],
    "chole bhature":        [280, 9.0,  12.0, 35.0],
    "dal makhani":          [130, 7.0,  5.0,  16.0],
    "dal tadka":            [110, 6.5,  4.0,  14.0],
    "dosa":                 [168, 3.9,  3.7,  30.0],
    "masala dosa":          [190, 4.5,  6.0,  30.0],
    "idli":                 [58,  2.0,  0.4,  12.0],
    "sambar":               [50,  3.0,  1.5,  7.0],
    "samosa":               [308, 6.0,  17.0, 34.0],
    "chole":                [160, 8.0,  5.0,  22.0],
    "rajma":                [140, 8.0,  2.0,  24.0],
    "roti":                 [297, 9.0,  3.7,  60.0],
    "naan":                 [310, 9.0,  7.0,  52.0],
    "paratha":              [300, 7.0,  12.0, 42.0],
    "palak paneer":         [160, 9.0,  11.0, 6.0],
    "paneer tikka":         [230, 14.0, 16.0, 6.0],
    "kadai paneer":         [200, 10.0, 14.0, 8.0],
    "shahi paneer":         [220, 10.0, 16.0, 9.0],
    "matar paneer":         [170, 9.0,  10.0, 12.0],
    "aloo gobi":            [100, 3.0,  4.0,  14.0],
    "baingan bharta":       [90,  2.5,  5.0,  10.0],
    "chana masala":         [160, 8.0,  5.0,  22.0],
    "pav bhaji":            [200, 5.0,  8.0,  28.0],
    "vada pav":             [290, 7.0,  12.0, 40.0],
    "poha":                 [130, 3.0,  3.0,  22.0],
    "upma":                 [150, 4.0,  5.0,  22.0],
    "khichdi":              [120, 5.0,  3.0,  20.0],
    "puri":                 [340, 6.0,  16.0, 44.0],
    "tandoori chicken":     [165, 25.0, 6.0,  2.0],
    "korma":                [180, 12.0, 12.0, 6.0],
    "vindaloo":             [170, 14.0, 10.0, 6.0],
    "dhokla":               [160, 5.0,  4.0,  26.0],
    "medu vada":            [310, 8.0,  14.0, 40.0],
    "uttapam":              [180, 5.0,  5.0,  28.0],
    "kachori":              [420, 8.0,  22.0, 50.0],
    "halwa":                [350, 4.0,  14.0, 54.0],
    "kheer":                [180, 5.0,  6.0,  27.0],
    "gulab jamun":          [387, 6.0,  15.0, 60.0],
    "jalebi":               [400, 2.0,  14.0, 65.0],
    "rasgulla":             [186, 4.0,  4.0,  34.0],
    "ras malai":            [190, 7.0,  8.0,  24.0],
    "lassi":                [100, 4.0,  3.5,  14.0],
    "mishti doi":           [150, 5.0,  4.0,  24.0],
    "payasam":              [200, 5.0,  7.0,  30.0],
    "modak":                [300, 4.0,  10.0, 50.0],
    # ── Global foods ──────────────────────────────────────────────────────
    "apple":                [52,  0.3,  0.2,  14.0],
    "banana":               [89,  1.1,  0.3,  23.0],
    "orange":               [47,  0.9,  0.1,  12.0],
    "pizza":                [266, 11.0, 10.0, 33.0],
    "burger":               [295, 17.0, 14.0, 24.0],
    "salad":                [20,  1.5,  0.2,  3.0],
    "pasta":                [220, 8.0,  2.0,  43.0],
    "sandwich":             [250, 10.0, 8.0,  35.0],
    "rice":                 [130, 2.7,  0.3,  28.0],
    "bread":                [265, 9.0,  3.2,  49.0],
    "egg":                  [155, 13.0, 11.0, 1.1],
    "grilled chicken":      [165, 31.0, 3.6,  0.0],
    "fish":                 [136, 20.0, 6.0,  0.0],
    "oats":                 [389, 17.0, 7.0,  66.0],
    "greek yogurt":         [59,  10.0, 0.4,  3.6],
}

# Default portion sizes in grams
PORTION_G = {
    "biryani": 300,          "butter chicken": 250,    "paneer butter masala": 250,
    "chole bhature": 350,    "dal makhani": 200,       "dal tadka": 200,
    "dosa": 120,             "masala dosa": 150,       "idli": 120,
    "sambar": 150,           "samosa": 100,            "chole": 200,
    "rajma": 200,            "roti": 90,               "naan": 90,
    "paratha": 120,          "palak paneer": 200,      "paneer tikka": 200,
    "kadai paneer": 200,     "shahi paneer": 200,      "matar paneer": 200,
    "aloo gobi": 200,        "baingan bharta": 200,    "chana masala": 200,
    "pav bhaji": 300,        "vada pav": 150,          "poha": 200,
    "upma": 200,             "khichdi": 250,           "puri": 100,
    "tandoori chicken": 250, "korma": 200,             "vindaloo": 250,
    "dhokla": 150,           "medu vada": 120,         "uttapam": 150,
    "kachori": 100,          "halwa": 150,             "kheer": 150,
    "gulab jamun": 80,       "jalebi": 100,            "rasgulla": 100,
    "ras malai": 120,        "lassi": 250,             "mishti doi": 150,
    "payasam": 150,          "modak": 80,
    "apple": 182,            "banana": 118,            "orange": 131,
    "pizza": 285,            "burger": 220,            "salad": 150,
    "pasta": 250,            "sandwich": 200,          "rice": 200,
    "bread": 60,             "egg": 50,                "grilled chicken": 200,
    "fish": 200,             "oats": 80,               "greek yogurt": 150,
}

DISH_NAMES = list(NUTRITION_DB.keys())


def get_nutrition(dish: str, portion_g: float = None) -> dict:
    """Return scaled nutrition for a dish at given portion size."""
    dish = dish.lower()
    if dish not in NUTRITION_DB:
        return {}
    per100 = NUTRITION_DB[dish]
    g = portion_g or PORTION_G.get(dish, 200)
    scale = g / 100.0
    return {
        "dish":      dish,
        "portion_g": g,
        "calories":  round(per100[0] * scale, 1),
        "protein_g": round(per100[1] * scale, 1),
        "fat_g":     round(per100[2] * scale, 1),
        "carbs_g":   round(per100[3] * scale, 1),
    }
