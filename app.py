"""
Indian Food Calorie Estimator — Accurate Inference App
Uses CLIP zero-shot classification (no training needed) + nutrition lookup.
Run: python app.py
"""

import numpy as np
import gradio as gr
import open_clip
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# 43 Indian dish classes + nutrition per 100g [cal, protein, fat, carbs]
# ---------------------------------------------------------------------------

DISHES = {
    "biryani":           [150, 6.0,  5.0,  22.0],
    "butter chicken":    [150, 12.0, 9.0,  5.0],
    "chole bhature":     [280, 9.0,  12.0, 35.0],
    "dal makhani":       [130, 7.0,  5.0,  16.0],
    "dosa":              [168, 3.9,  3.7,  30.0],
    "gulab jamun":       [387, 6.0,  15.0, 60.0],
    "idli":              [58,  2.0,  0.4,  12.0],
    "jalebi":            [400, 2.0,  14.0, 65.0],
    "kadai paneer":      [200, 10.0, 14.0, 8.0],
    "kheer":             [180, 5.0,  6.0,  27.0],
    "masala dosa":       [190, 4.5,  6.0,  30.0],
    "naan":              [310, 9.0,  7.0,  52.0],
    "palak paneer":      [160, 9.0,  11.0, 6.0],
    "paneer tikka":      [230, 14.0, 16.0, 6.0],
    "paratha":           [300, 7.0,  12.0, 42.0],
    "pav bhaji":         [200, 5.0,  8.0,  28.0],
    "poha":              [130, 3.0,  3.0,  22.0],
    "puri":              [340, 6.0,  16.0, 44.0],
    "rajma":             [140, 8.0,  2.0,  24.0],
    "rasgulla":          [186, 4.0,  4.0,  34.0],
    "sambar":            [50,  3.0,  1.5,  7.0],
    "samosa":            [308, 6.0,  17.0, 34.0],
    "shahi paneer":      [220, 10.0, 16.0, 9.0],
    "tandoori chicken":  [165, 25.0, 6.0,  2.0],
    "upma":              [150, 4.0,  5.0,  22.0],
    "vada pav":          [290, 7.0,  12.0, 40.0],
    "aloo gobi":         [100, 3.0,  4.0,  14.0],
    "baingan bharta":    [90,  2.5,  5.0,  10.0],
    "chana masala":      [160, 8.0,  5.0,  22.0],
    "dhokla":            [160, 5.0,  4.0,  26.0],
    "halwa":             [350, 4.0,  14.0, 54.0],
    "kachori":           [420, 8.0,  22.0, 50.0],
    "khichdi":           [120, 5.0,  3.0,  20.0],
    "korma":             [180, 12.0, 12.0, 6.0],
    "lassi":             [100, 4.0,  3.5,  14.0],
    "matar paneer":      [170, 9.0,  10.0, 12.0],
    "medu vada":         [310, 8.0,  14.0, 40.0],
    "mishti doi":        [150, 5.0,  4.0,  24.0],
    "modak":             [300, 4.0,  10.0, 50.0],
    "payasam":           [200, 5.0,  7.0,  30.0],
    "ras malai":         [190, 7.0,  8.0,  24.0],
    "uttapam":           [180, 5.0,  5.0,  28.0],
    "vindaloo":          [170, 14.0, 10.0, 6.0],
    # Common non-Indian foods so CLIP doesn't force-fit them
    "apple":             [52,  0.3,  0.2,  14.0],
    "banana":            [89,  1.1,  0.3,  23.0],
    "orange":            [47,  0.9,  0.1,  12.0],
    "pizza":             [266, 11.0, 10.0, 33.0],
    "burger":            [295, 17.0, 14.0, 24.0],
    "salad":             [20,  1.5,  0.2,  3.0],
    "pasta":             [220, 8.0,  2.0,  43.0],
    "sandwich":          [250, 10.0, 8.0,  35.0],
    "rice":              [130, 2.7,  0.3,  28.0],
    "bread":             [265, 9.0,  3.2,  49.0],
    "egg":               [155, 13.0, 11.0, 1.1],
    "chicken":           [165, 31.0, 3.6,  0.0],
    "fish":              [136, 20.0, 6.0,  0.0],
}

DISH_NAMES = list(DISHES.keys())

# Typical portion sizes in grams
PORTION_G = {
    "biryani": 300, "butter chicken": 250, "chole bhature": 350,
    "dal makhani": 200, "dosa": 120, "gulab jamun": 80,
    "idli": 120, "jalebi": 100, "kadai paneer": 200,
    "kheer": 150, "masala dosa": 150, "naan": 90,
    "palak paneer": 200, "paneer tikka": 200, "paratha": 120,
    "pav bhaji": 300, "poha": 200, "puri": 100,
    "rajma": 200, "rasgulla": 100, "sambar": 150,
    "samosa": 100, "shahi paneer": 200, "tandoori chicken": 250,
    "upma": 200, "vada pav": 150, "aloo gobi": 200,
    "baingan bharta": 200, "chana masala": 200, "dhokla": 150,
    "halwa": 150, "kachori": 100, "khichdi": 250,
    "korma": 200, "lassi": 250, "matar paneer": 200,
    "medu vada": 120, "mishti doi": 150, "modak": 80,
    "payasam": 150, "ras malai": 120, "uttapam": 150,
    "vindaloo": 250,
    "apple": 182, "banana": 118, "orange": 131, "pizza": 285,
    "burger": 220, "salad": 150, "pasta": 250, "sandwich": 200,
    "rice": 200, "bread": 60, "egg": 50, "chicken": 200, "fish": 200,
}

# ---------------------------------------------------------------------------
# Load CLIP model once at startup
# ---------------------------------------------------------------------------

print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
clip_model = clip_model.to(device).eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Pre-encode all text prompts once
PROMPTS = [f"a photo of {name}, Indian food" for name in DISH_NAMES]
with torch.no_grad():
    text_tokens = tokenizer(PROMPTS).to(device)
    text_features = clip_model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

print(f"CLIP ready on {device}. {len(DISH_NAMES)} food classes loaded.")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def estimate_calories(image: Image.Image, portion_override: float = 0):
    if image is None:
        return "No image provided", ""

    image = image.convert("RGB")
    img_tensor = clip_preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        img_features = clip_model.encode_image(img_tensor)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        similarities = (img_features @ text_features.T).squeeze(0)
        probs = similarities.softmax(dim=-1).cpu().numpy()

    top5_idx = probs.argsort()[::-1][:5]
    top_dish = DISH_NAMES[top5_idx[0]]
    confidence = float(probs[top5_idx[0]]) * 100

    portion_g = portion_override if portion_override > 0 else PORTION_G.get(top_dish, 200)
    nutrition_per100 = DISHES[top_dish]
    scale = portion_g / 100.0

    cal     = round(nutrition_per100[0] * scale, 1)
    protein = round(nutrition_per100[1] * scale, 1)
    fat     = round(nutrition_per100[2] * scale, 1)
    carbs   = round(nutrition_per100[3] * scale, 1)

    # Top-5 alternatives
    alts = "\n".join(
        f"  {i+2}. {DISH_NAMES[top5_idx[i+1]].title()} ({probs[top5_idx[i+1]]*100:.1f}%)"
        for i in range(4)
    )

    result = (
        f"🍽️  Dish:     {top_dish.title()}  ({confidence:.1f}% confidence)\n"
        f"⚖️  Portion:  {portion_g} g\n\n"
        f"🔥 Calories:      {cal} kcal\n"
        f"💪 Protein:       {protein} g\n"
        f"🧈 Fat:           {fat} g\n"
        f"🌾 Carbohydrates: {carbs} g\n\n"
        f"📋 Other possibilities:\n{alts}"
    )

    return top_dish.title(), result


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="🍛 Indian Food Calorie Estimator") as demo:
    gr.Markdown(
        "# 🍛 Indian Food Calorie Estimator\n"
        "Upload a photo or use your camera. Works on Indian food **and** common foods like fruits, pizza, etc."
    )

    with gr.Tabs():
        with gr.Tab("📁 Upload Photo"):
            with gr.Row():
                upload_img = gr.Image(type="pil", label="Upload food image",
                                      sources=["upload"], height=320)
                with gr.Column():
                    upload_dish  = gr.Textbox(label="Detected Dish", interactive=False)
                    upload_info  = gr.Textbox(label="Nutrition Info", lines=12, interactive=False)
            portion_upload = gr.Slider(0, 1000, value=0, step=10,
                                       label="Custom portion size (g) — 0 = use default")
            gr.Button("Estimate Calories", variant="primary").click(
                estimate_calories,
                inputs=[upload_img, portion_upload],
                outputs=[upload_dish, upload_info],
            )

        with gr.Tab("📷 Camera"):
            with gr.Row():
                cam_img = gr.Image(type="pil", label="Take a photo",
                                   sources=["webcam"], height=320)
                with gr.Column():
                    cam_dish = gr.Textbox(label="Detected Dish", interactive=False)
                    cam_info = gr.Textbox(label="Nutrition Info", lines=12, interactive=False)
            portion_cam = gr.Slider(0, 1000, value=0, step=10,
                                    label="Custom portion size (g) — 0 = use default")
            gr.Button("Estimate Calories", variant="primary").click(
                estimate_calories,
                inputs=[cam_img, portion_cam],
                outputs=[cam_dish, cam_info],
            )

    gr.Markdown("---\n*Powered by OpenAI CLIP zero-shot classification. Values are estimates.*")

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
