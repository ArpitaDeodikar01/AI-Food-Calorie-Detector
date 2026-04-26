"""
Generates 2 publication-quality figures:
  Fig 1 — Confusion Matrix (simulated CLIP predictions on 20 Indian dishes)
  Fig 2 — Dataset Distribution (calories, macros, portion sizes across dish categories)

Run: python food_ai/visualize.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from food_ai.nutrition_db import NUTRITION_DB, PORTION_G

np.random.seed(42)

# ── Colour palette ────────────────────────────────────────────────────────
PALETTE = {
    "bg":      "#0f1117",
    "panel":   "#1a1d27",
    "accent1": "#7c6af7",   # purple
    "accent2": "#f7706a",   # coral
    "accent3": "#4ecdc4",   # teal
    "accent4": "#ffe66d",   # yellow
    "text":    "#e8e8f0",
    "grid":    "#2a2d3e",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["panel"],
    "axes.edgecolor":    PALETTE["grid"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["text"],
    "ytick.color":       PALETTE["text"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["grid"],
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         9,
})

# ============================================================
# FIGURE 1 — CONFUSION MATRIX
# ============================================================

CLASSES = [
    "biryani", "butter chicken", "chole bhature", "dal makhani", "dal tadka",
    "dosa", "gulab jamun", "idli", "jalebi", "kadai paneer",
    "masala dosa", "naan", "palak paneer", "paneer tikka", "paratha",
    "rajma", "samosa", "tandoori chicken", "uttapam", "vada pav",
]
N = len(CLASSES)

# Simulated CLIP accuracy per class (based on visual distinctiveness)
TRUE_ACC = {
    "biryani": 0.91,        "butter chicken": 0.83,  "chole bhature": 0.79,
    "dal makhani": 0.72,    "dal tadka": 0.68,        "dosa": 0.88,
    "gulab jamun": 0.90,    "idli": 0.87,             "jalebi": 0.93,
    "kadai paneer": 0.74,   "masala dosa": 0.82,      "naan": 0.85,
    "palak paneer": 0.76,   "paneer tikka": 0.80,     "paratha": 0.78,
    "rajma": 0.71,          "samosa": 0.92,           "tandoori chicken": 0.89,
    "uttapam": 0.81,        "vada pav": 0.84,
}

# Confusion pairs (which classes get confused with which)
CONFUSE = {
    "dal makhani":  {"dal tadka": 0.14, "rajma": 0.08, "kadai paneer": 0.06},
    "dal tadka":    {"dal makhani": 0.16, "rajma": 0.09, "palak paneer": 0.07},
    "rajma":        {"dal makhani": 0.12, "dal tadka": 0.10, "chole bhature": 0.07},
    "kadai paneer": {"palak paneer": 0.12, "paneer tikka": 0.08, "butter chicken": 0.06},
    "palak paneer": {"kadai paneer": 0.10, "butter chicken": 0.08, "dal makhani": 0.06},
    "masala dosa":  {"dosa": 0.10, "uttapam": 0.05, "paratha": 0.03},
    "dosa":         {"masala dosa": 0.07, "uttapam": 0.04, "paratha": 0.01},
    "paratha":      {"naan": 0.12, "roti": 0.05, "uttapam": 0.05},
    "naan":         {"paratha": 0.09, "uttapam": 0.04, "dosa": 0.02},
    "chole bhature":{"rajma": 0.10, "dal makhani": 0.07, "vada pav": 0.04},
    "paneer tikka": {"kadai paneer": 0.10, "tandoori chicken": 0.07, "butter chicken": 0.03},
}

# Build confusion matrix
cm = np.zeros((N, N), dtype=float)
n_samples = 200

for i, cls in enumerate(CLASSES):
    acc = TRUE_ACC[cls]
    cm[i, i] = round(acc * n_samples)
    remaining = n_samples - cm[i, i]
    conf = CONFUSE.get(cls, {})
    conf_total = sum(conf.values())

    for j, other in enumerate(CLASSES):
        if other in conf and i != j:
            frac = conf[other] / max(conf_total, 1e-9)
            cm[i, j] = round(frac * remaining * 0.85)

    # Spread leftover randomly
    leftover = n_samples - cm[i].sum()
    if leftover > 0:
        others = [j for j in range(N) if j != i and cm[i, j] == 0]
        if others:
            picks = np.random.choice(others, size=min(int(leftover), len(others)),
                                     replace=False)
            for p in picks:
                cm[i, p] += 1

# Normalise to percentages
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

fig1, ax1 = plt.subplots(figsize=(14, 11))
fig1.patch.set_facecolor(PALETTE["bg"])

cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(
    cm_pct, ax=ax1,
    annot=True, fmt=".0f",
    cmap="YlOrRd",
    linewidths=0.4, linecolor=PALETTE["bg"],
    xticklabels=[c.replace(" ", "\n") for c in CLASSES],
    yticklabels=[c.replace(" ", "\n") for c in CLASSES],
    cbar_kws={"label": "Prediction %", "shrink": 0.8},
    annot_kws={"size": 7},
    vmin=0, vmax=100,
)

# Highlight diagonal
for i in range(N):
    ax1.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                 edgecolor=PALETTE["accent3"], lw=2))

ax1.set_title("Figure 1 — Confusion Matrix: CLIP Zero-Shot Classification\n"
              "Indian Food Dishes (20 classes, n=200 per class, simulated)",
              fontsize=13, fontweight="bold", color=PALETTE["text"], pad=16)
ax1.set_xlabel("Predicted Label", fontsize=10, labelpad=10)
ax1.set_ylabel("True Label", fontsize=10, labelpad=10)
ax1.tick_params(axis="x", rotation=45, labelsize=7.5)
ax1.tick_params(axis="y", rotation=0,  labelsize=7.5)

# Overall accuracy annotation
overall_acc = np.diag(cm_pct).mean()
fig1.text(0.72, 0.02,
          f"Mean Top-1 Accuracy: {overall_acc:.1f}%   |   "
          f"Hardest: dal tadka ({TRUE_ACC['dal tadka']*100:.0f}%)   |   "
          f"Easiest: jalebi ({TRUE_ACC['jalebi']*100:.0f}%)",
          ha="center", fontsize=8.5, color=PALETTE["accent4"],
          style="italic")

plt.tight_layout(rect=[0, 0.04, 1, 1])
fig1.savefig("food_ai/confusion_matrix.png", dpi=150, bbox_inches="tight",
             facecolor=PALETTE["bg"])
print("Saved: food_ai/confusion_matrix.png")


# ============================================================
# FIGURE 2 — DATASET VISUALISATION (4 subplots)
# ============================================================

# Build dataset from nutrition_db
dishes      = list(NUTRITION_DB.keys())
calories    = [NUTRITION_DB[d][0] for d in dishes]
proteins    = [NUTRITION_DB[d][1] for d in dishes]
fats        = [NUTRITION_DB[d][2] for d in dishes]
carbs       = [NUTRITION_DB[d][3] for d in dishes]
portions    = [PORTION_G.get(d, 200) for d in dishes]
total_cals  = [NUTRITION_DB[d][0] * PORTION_G.get(d, 200) / 100 for d in dishes]

# Category labels
def categorise(dish):
    rice   = ["biryani", "rice", "khichdi", "poha", "upma"]
    bread  = ["roti", "naan", "paratha", "puri", "kachori", "bread", "vada pav",
              "pav bhaji", "sandwich"]
    paneer = ["palak paneer", "paneer tikka", "kadai paneer", "shahi paneer",
              "matar paneer", "paneer butter masala"]
    sweet  = ["gulab jamun", "jalebi", "rasgulla", "ras malai", "kheer",
              "halwa", "mishti doi", "modak", "payasam", "lassi", "mishti doi"]
    lentil = ["dal makhani", "dal tadka", "rajma", "chole", "chole bhature",
              "chana masala", "sambar"]
    snack  = ["samosa", "dhokla", "medu vada", "uttapam", "dosa", "masala dosa",
              "idli", "kachori"]
    protein= ["tandoori chicken", "butter chicken", "korma", "vindaloo",
              "paneer tikka", "grilled chicken", "fish", "egg"]
    fruit  = ["apple", "banana", "orange"]
    global_= ["pizza", "burger", "pasta", "salad", "oats", "greek yogurt",
               "bread", "sandwich", "rice"]
    if dish in rice:    return "Rice/Grain"
    if dish in bread:   return "Bread/Snack"
    if dish in paneer:  return "Paneer"
    if dish in sweet:   return "Sweet/Drink"
    if dish in lentil:  return "Lentil/Bean"
    if dish in snack:   return "Snack/Breakfast"
    if dish in protein: return "Protein/Meat"
    if dish in fruit:   return "Fruit"
    return "Global"

categories = [categorise(d) for d in dishes]
cat_set    = sorted(set(categories))
cat_colors = {c: col for c, col in zip(cat_set, [
    PALETTE["accent1"], PALETTE["accent2"], PALETTE["accent3"],
    PALETTE["accent4"], "#ff9f43", "#a29bfe", "#fd79a8", "#55efc4", "#74b9ff"
])}
colors = [cat_colors[c] for c in categories]

fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
fig2.patch.set_facecolor(PALETTE["bg"])
fig2.suptitle("Figure 2 — Dataset Visualisation: Nutrition Profile of 62 Food Dishes",
              fontsize=14, fontweight="bold", color=PALETTE["text"], y=0.98)

# ── Subplot A: Calories per 100g (bar chart, sorted) ─────────────────────
ax = axes[0, 0]
sorted_idx = np.argsort(calories)[::-1]
top_n = 25
idx = sorted_idx[:top_n]
bar_colors = [colors[i] for i in idx]
bars = ax.barh([dishes[i].replace(" ", "\n") for i in idx],
               [calories[i] for i in idx],
               color=bar_colors, edgecolor=PALETTE["bg"], linewidth=0.4, height=0.7)
ax.set_xlabel("Calories per 100g (kcal)", fontsize=9)
ax.set_title("A — Calories per 100g (Top 25)", fontsize=10, fontweight="bold",
             color=PALETTE["text"])
ax.axvline(np.mean(calories), color=PALETTE["accent4"], linestyle="--",
           linewidth=1.2, label=f"Mean: {np.mean(calories):.0f} kcal")
ax.legend(fontsize=8, facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"])
ax.grid(axis="x", alpha=0.4)
ax.tick_params(labelsize=7)
ax.invert_yaxis()

# ── Subplot B: Macro breakdown scatter (protein vs carbs, size=fat) ───────
ax = axes[0, 1]
sc = ax.scatter(carbs, proteins, s=[f * 8 + 20 for f in fats],
                c=colors, alpha=0.85, edgecolors=PALETTE["bg"], linewidth=0.5)
for i, d in enumerate(dishes):
    if calories[i] > 300 or proteins[i] > 20:
        ax.annotate(d, (carbs[i], proteins[i]),
                    fontsize=6.5, color=PALETTE["text"], alpha=0.8,
                    xytext=(4, 2), textcoords="offset points")
ax.set_xlabel("Carbohydrates (g per 100g)", fontsize=9)
ax.set_ylabel("Protein (g per 100g)", fontsize=9)
ax.set_title("B — Protein vs Carbs  (bubble size = fat content)",
             fontsize=10, fontweight="bold", color=PALETTE["text"])
ax.grid(alpha=0.3)
ax.axhline(np.mean(proteins), color=PALETTE["accent2"], linestyle=":",
           linewidth=1, alpha=0.7)
ax.axvline(np.mean(carbs), color=PALETTE["accent3"], linestyle=":",
           linewidth=1, alpha=0.7)

# ── Subplot C: Total calories per serving (portion-adjusted) ─────────────
ax = axes[1, 0]
cat_totals = {}
for d, tc, cat in zip(dishes, total_cals, categories):
    cat_totals.setdefault(cat, []).append(tc)

cat_means  = {c: np.mean(v) for c, v in cat_totals.items()}
cat_stds   = {c: np.std(v)  for c, v in cat_totals.items()}
sorted_cats = sorted(cat_means, key=cat_means.get, reverse=True)

xpos = np.arange(len(sorted_cats))
bar_c = [cat_colors[c] for c in sorted_cats]
ax.bar(xpos, [cat_means[c] for c in sorted_cats],
       yerr=[cat_stds[c] for c in sorted_cats],
       color=bar_c, edgecolor=PALETTE["bg"], linewidth=0.5,
       capsize=4, error_kw={"ecolor": PALETTE["text"], "alpha": 0.6})
ax.set_xticks(xpos)
ax.set_xticklabels([c.replace("/", "/\n") for c in sorted_cats],
                   rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Calories per serving (kcal)", fontsize=9)
ax.set_title("C — Avg Calories per Serving by Category (±1 std)",
             fontsize=10, fontweight="bold", color=PALETTE["text"])
ax.grid(axis="y", alpha=0.4)
ax.axhline(np.mean(total_cals), color=PALETTE["accent4"], linestyle="--",
           linewidth=1.2, alpha=0.8, label=f"Overall mean: {np.mean(total_cals):.0f} kcal")
ax.legend(fontsize=8, facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"])

# ── Subplot D: Macro composition stacked bar (top 20 dishes) ─────────────
ax = axes[1, 1]
top20 = sorted(range(len(dishes)), key=lambda i: total_cals[i], reverse=True)[:20]
d_labels = [dishes[i].replace(" ", "\n") for i in top20]
p_vals   = [proteins[i] * 4 for i in top20]   # kcal from protein
f_vals   = [fats[i]    * 9 for i in top20]   # kcal from fat
c_vals   = [carbs[i]   * 4 for i in top20]   # kcal from carbs
xp = np.arange(len(top20))
w  = 0.6

b1 = ax.bar(xp, p_vals, width=w, label="Protein (4 kcal/g)",
            color=PALETTE["accent3"], edgecolor=PALETTE["bg"], linewidth=0.3)
b2 = ax.bar(xp, f_vals, width=w, bottom=p_vals, label="Fat (9 kcal/g)",
            color=PALETTE["accent2"], edgecolor=PALETTE["bg"], linewidth=0.3)
b3 = ax.bar(xp, c_vals, width=w,
            bottom=[p + f for p, f in zip(p_vals, f_vals)],
            label="Carbs (4 kcal/g)",
            color=PALETTE["accent1"], edgecolor=PALETTE["bg"], linewidth=0.3)

ax.set_xticks(xp)
ax.set_xticklabels(d_labels, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Calories from macros (kcal/100g)", fontsize=9)
ax.set_title("D — Macro Energy Composition (Top 20 by serving calories)",
             fontsize=10, fontweight="bold", color=PALETTE["text"])
ax.legend(fontsize=8, facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
          loc="upper right")
ax.grid(axis="y", alpha=0.4)

# ── Category legend (shared) ─────────────────────────────────────────────
legend_patches = [mpatches.Patch(color=cat_colors[c], label=c) for c in cat_set]
fig2.legend(handles=legend_patches, loc="lower center", ncol=5,
            fontsize=8, facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
            title="Food Category", title_fontsize=8,
            bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.06, 1, 0.97])
fig2.savefig("food_ai/dataset_visualisation.png", dpi=150, bbox_inches="tight",
             facecolor=PALETTE["bg"])
print("Saved: food_ai/dataset_visualisation.png")

plt.close("all")
print("\nDone. Open the two PNG files in food_ai/")
