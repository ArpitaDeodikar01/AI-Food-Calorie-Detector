"""
Model 1 — CLIP Zero-Shot Food Classifier + Gradio UI
Works immediately, no training needed.
Run: python food_ai/app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gradio as gr
import numpy as np
from PIL import Image

import open_clip
from food_ai.nutrition_db import DISH_NAMES, NUTRITION_DB, PORTION_G, get_nutrition
from food_ai.user_profile import PRESET_PROFILES
from food_ai.model3_rl_agent import DQNAgent, DietEnv, MEAL_DB

# ---------------------------------------------------------------------------
# Load CLIP once
# ---------------------------------------------------------------------------

print("[Model 1] Loading CLIP ViT-B/32...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
clip_model = clip_model.to(device).eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

PROMPTS = [f"a photo of {name}, food dish" for name in DISH_NAMES]
with torch.no_grad():
    text_tokens   = tokenizer(PROMPTS).to(device)
    text_features = clip_model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

print(f"[Model 1] CLIP ready on {device}. {len(DISH_NAMES)} classes.")

# Load RL agent (try to load trained weights, else use untrained)
_env = DietEnv(PRESET_PROFILES["maintenance"])
rl_agent = DQNAgent(state_dim=_env.observation_space.shape[0],
                    action_dim=_env.action_space.n)
_ckpt = os.path.join(os.path.dirname(__file__), "rl_agent.pt")
if os.path.exists(_ckpt):
    rl_agent.q_net.load_state_dict(torch.load(_ckpt, map_location="cpu"))
    print("[Model 3] Loaded trained RL agent.")
else:
    print("[Model 3] No trained RL agent found — using untrained (random) recommendations.")

# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def clip_predict(image: Image.Image, portion_g: float = 0) -> dict:
    image = image.convert("RGB")
    img_t = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feat = clip_model.encode_image(img_t)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        probs = (img_feat @ text_features.T).squeeze(0).softmax(dim=-1).cpu().numpy()

    top5 = probs.argsort()[::-1][:5]
    top_dish = DISH_NAMES[top5[0]]
    g = portion_g if portion_g > 0 else PORTION_G.get(top_dish, 200)
    nutrition = get_nutrition(top_dish, g)
    nutrition["confidence"] = float(probs[top5[0]]) * 100
    nutrition["top5"] = [(DISH_NAMES[i], round(float(probs[i]) * 100, 1)) for i in top5]
    return nutrition


def format_output(result: dict, profile_name: str) -> tuple:
    if not result:
        return "Unknown", "", ""

    top5_text = "\n".join(
        f"  {i+1}. {d.title()} ({c:.1f}%)"
        for i, (d, c) in enumerate(result["top5"])
    )

    nutrition_text = (
        f"🍽️  Dish:     {result['dish'].title()}  ({result['confidence']:.1f}% confidence)\n"
        f"⚖️  Portion:  {result['portion_g']} g\n\n"
        f"🔥 Calories:      {result['calories']} kcal\n"
        f"💪 Protein:       {result['protein_g']} g\n"
        f"🧈 Fat:           {result['fat_g']} g\n"
        f"🌾 Carbohydrates: {result['carbs_g']} g\n\n"
        f"📋 Top-5 Predictions:\n{top5_text}"
    )

    # RL recommendation
    profile = PRESET_PROFILES[profile_name]
    profile.log_meal(result["dish"], result["calories"],
                     result["protein_g"], result["fat_g"], result["carbs_g"])

    env = DietEnv(profile)
    state = env.reset()[0]
    action = rl_agent.act(state, epsilon=0.0)
    rec_meal = MEAL_DB[action]
    rec_nutrition = get_nutrition(rec_meal)

    rec_text = (
        f"🤖 RL Agent Recommendation (goal: {profile.goal.replace('_',' ').title()})\n\n"
        f"Next meal suggestion: {rec_meal.title()}\n"
        f"  Calories: {rec_nutrition.get('calories', 0)} kcal\n"
        f"  Protein:  {rec_nutrition.get('protein_g', 0)} g\n\n"
        f"📊 Today so far:\n  {profile.summary()}"
    )

    return result["dish"].title(), nutrition_text, rec_text


def predict(image, portion_g, profile_name):
    if image is None:
        return "No image", "", ""
    result = clip_predict(image, float(portion_g))
    return format_output(result, profile_name)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="🍛 AI Food Calorie Estimator") as demo:
    gr.Markdown(
        "# 🍛 AI Food Calorie & Nutrition Estimator\n"
        "**Model 1 (CLIP)** identifies the dish · **Model 3 (RL Agent)** recommends your next meal"
    )

    with gr.Row():
        profile_dd = gr.Dropdown(
            choices=list(PRESET_PROFILES.keys()),
            value="maintenance",
            label="Your Health Goal",
        )

    with gr.Tabs():
        with gr.Tab("📁 Upload Photo"):
            with gr.Row():
                upload_img = gr.Image(type="pil", label="Upload food image",
                                      sources=["upload"], height=320)
                with gr.Column():
                    dish_out  = gr.Textbox(label="Detected Dish", interactive=False)
                    nutri_out = gr.Textbox(label="Nutrition Info", lines=12, interactive=False)
                    rec_out   = gr.Textbox(label="RL Diet Recommendation", lines=8, interactive=False)
            portion_sl = gr.Slider(50, 500, value=0, step=10,
                                   label="Custom portion (g) — 0 = auto")
            gr.Button("Analyse", variant="primary").click(
                predict,
                inputs=[upload_img, portion_sl, profile_dd],
                outputs=[dish_out, nutri_out, rec_out],
            )

        with gr.Tab("📷 Camera"):
            gr.Markdown(
                "> **Tip:** Allow camera access when the browser asks. "
                "Snap a photo then click Analyse."
            )
            with gr.Row():
                cam_img = gr.Image(type="pil", label="Take a photo",
                                   sources=["webcam"], height=320)
                with gr.Column():
                    cam_dish  = gr.Textbox(label="Detected Dish", interactive=False)
                    cam_nutri = gr.Textbox(label="Nutrition Info", lines=12, interactive=False)
                    cam_rec   = gr.Textbox(label="RL Diet Recommendation", lines=8, interactive=False)
            cam_portion = gr.Slider(50, 500, value=0, step=10,
                                    label="Custom portion (g) — 0 = auto")
            gr.Button("Analyse", variant="primary").click(
                predict,
                inputs=[cam_img, cam_portion, profile_dd],
                outputs=[cam_dish, cam_nutri, cam_rec],
            )

    gr.Markdown("---\n*CLIP zero-shot · RL agent · Nutrition values are estimates*")

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_port=7860, share=False)
