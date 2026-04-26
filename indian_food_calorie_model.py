"""
Multi-task Deep Neural Network for Indian Food Calorie Estimation
with Domain Adaptation using PyTorch Lightning.

Features:
  - Auto-downloads MM-Food-100K from Hugging Face (falls back to synthetic data)
  - Gradio UI with camera capture + photo upload for live calorie detection
"""

import io
import json
import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# Suppress verbose HF/transformers load messages
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTModel
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image

# ---------------------------------------------------------------------------
# Indian dish class names (43 classes)
# ---------------------------------------------------------------------------

DISH_CLASSES = [
    "biryani", "butter_chicken", "chole_bhature", "dal_makhani", "dosa",
    "gulab_jamun", "idli", "jalebi", "kadai_paneer", "kheer",
    "masala_dosa", "naan", "palak_paneer", "paneer_tikka", "paratha",
    "pav_bhaji", "poha", "puri", "rajma", "rasgulla",
    "sambar", "samosa", "shahi_paneer", "tandoori_chicken", "upma",
    "vada_pav", "aloo_gobi", "baingan_bharta", "chana_masala", "dhokla",
    "halwa", "kachori", "khichdi", "korma", "lassi",
    "matar_paneer", "medu_vada", "mishti_doi", "modak", "payasam",
    "ras_malai", "uttapam", "vindaloo",
]

# Approximate nutrition per 100g: [calories, protein_g, fat_g, carbs_g]
DISH_NUTRITION_PER_100G = {
    "biryani":          [150, 6.0, 5.0, 22.0],
    "butter_chicken":   [150, 12.0, 9.0, 5.0],
    "chole_bhature":    [280, 9.0, 12.0, 35.0],
    "dal_makhani":      [130, 7.0, 5.0, 16.0],
    "dosa":             [168, 3.9, 3.7, 30.0],
    "gulab_jamun":      [387, 6.0, 15.0, 60.0],
    "idli":             [58,  2.0, 0.4, 12.0],
    "jalebi":           [400, 2.0, 14.0, 65.0],
    "kadai_paneer":     [200, 10.0, 14.0, 8.0],
    "kheer":            [180, 5.0, 6.0, 27.0],
    "masala_dosa":      [190, 4.5, 6.0, 30.0],
    "naan":             [310, 9.0, 7.0, 52.0],
    "palak_paneer":     [160, 9.0, 11.0, 6.0],
    "paneer_tikka":     [230, 14.0, 16.0, 6.0],
    "paratha":          [300, 7.0, 12.0, 42.0],
    "pav_bhaji":        [200, 5.0, 8.0, 28.0],
    "poha":             [130, 3.0, 3.0, 22.0],
    "puri":             [340, 6.0, 16.0, 44.0],
    "rajma":            [140, 8.0, 2.0, 24.0],
    "rasgulla":         [186, 4.0, 4.0, 34.0],
    "sambar":           [50,  3.0, 1.5, 7.0],
    "samosa":           [308, 6.0, 17.0, 34.0],
    "shahi_paneer":     [220, 10.0, 16.0, 9.0],
    "tandoori_chicken": [165, 25.0, 6.0, 2.0],
    "upma":             [150, 4.0, 5.0, 22.0],
    "vada_pav":         [290, 7.0, 12.0, 40.0],
    "aloo_gobi":        [100, 3.0, 4.0, 14.0],
    "baingan_bharta":   [90,  2.5, 5.0, 10.0],
    "chana_masala":     [160, 8.0, 5.0, 22.0],
    "dhokla":           [160, 5.0, 4.0, 26.0],
    "halwa":            [350, 4.0, 14.0, 54.0],
    "kachori":          [420, 8.0, 22.0, 50.0],
    "khichdi":          [120, 5.0, 3.0, 20.0],
    "korma":            [180, 12.0, 12.0, 6.0],
    "lassi":            [100, 4.0, 3.5, 14.0],
    "matar_paneer":     [170, 9.0, 10.0, 12.0],
    "medu_vada":        [310, 8.0, 14.0, 40.0],
    "mishti_doi":       [150, 5.0, 4.0, 24.0],
    "modak":            [300, 4.0, 10.0, 50.0],
    "payasam":          [200, 5.0, 7.0, 30.0],
    "ras_malai":        [190, 7.0, 8.0, 24.0],
    "uttapam":          [180, 5.0, 5.0, 28.0],
    "vindaloo":         [170, 14.0, 10.0, 6.0],
}

# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 0.1):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MMFood100KDataset(Dataset):
    NUTRITION_KEYS = ["calories", "protein_g", "fat_g", "carbohydrate_g"]

    def __init__(self, hf_dataset, transform=None, domain: int = 0):
        self.data = hf_dataset
        self.domain = domain
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _parse_nutrition(self, profile):
        if isinstance(profile, str):
            profile = json.loads(profile)
        return [float(profile.get(k, 0.0)) for k in self.NUTRITION_KEYS]

    def _parse_portion(self, portion):
        if isinstance(portion, (int, float)):
            return float(portion)
        if isinstance(portion, str):
            return float("".join(c for c in portion if c.isdigit() or c == ".") or 0)
        return 0.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")
        image = self.transform(image)

        label = int(sample.get("label", sample.get("dish_class", 0)))
        nutrition = torch.tensor(self._parse_nutrition(sample["nutritional_profile"]),
                                 dtype=torch.float32)
        portion = torch.tensor([self._parse_portion(sample.get("portion_size", 0))],
                                dtype=torch.float32)
        domain = torch.tensor(int(sample.get("domain", self.domain)), dtype=torch.long)
        return image, label, nutrition, portion, domain


# ---------------------------------------------------------------------------
# Synthetic fallback dataset (used when HF dataset is unavailable)
# ---------------------------------------------------------------------------

class SyntheticFoodDataset(Dataset):
    """Generates random RGB images with plausible nutrition labels for testing."""

    def __init__(self, size: int = 1000, transform=None):
        self.size = size
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        label = idx % len(DISH_CLASSES)
        dish = DISH_CLASSES[label]
        base = DISH_NUTRITION_PER_100G[dish]
        portion = random.uniform(150, 400)  # grams

        # Scale nutrition by portion
        nutrition = torch.tensor(
            [v * portion / 100.0 + random.gauss(0, 2) for v in base],
            dtype=torch.float32,
        )
        portion_t = torch.tensor([portion], dtype=torch.float32)
        domain = torch.tensor(0, dtype=torch.long)

        # Random coloured image
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = self.transform(Image.fromarray(arr))
        return image, label, nutrition, portion_t, domain


# ---------------------------------------------------------------------------
# VAE Branch
# ---------------------------------------------------------------------------

class VAEBranch(nn.Module):
    def __init__(self, in_dim: int = 768, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, features):
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


# ---------------------------------------------------------------------------
# Main Lightning Module
# ---------------------------------------------------------------------------

class IndianFoodCalorieModel(pl.LightningModule):
    NUM_CLASSES = 43
    NUM_NUTRITION = 4

    def __init__(self, lr=1e-4, weight_decay=1e-4, lambda_domain=0.1,
                 kl_weight=0.001, epochs=50, freeze_layers=8):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = ViTModel.from_pretrained(
            "google/vit-base-patch16-224", ignore_mismatched_sizes=True
        )
        self._freeze_vit_layers(freeze_layers)
        feat_dim = self.backbone.config.hidden_size  # 768

        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, self.NUM_CLASSES),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, self.NUM_NUTRITION),
        )
        self.vae = VAEBranch(in_dim=feat_dim, latent_dim=32)
        self.grl = GradientReversalLayer(lambda_=lambda_domain)
        self.domain_head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

        self.loss_cls = nn.CrossEntropyLoss()
        self.loss_reg = nn.HuberLoss(delta=1.0)
        self.loss_domain = nn.CrossEntropyLoss()

    def _freeze_vit_layers(self, n):
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        for i, layer in enumerate(self.backbone.encoder.layer):
            if i < n:
                for p in layer.parameters():
                    p.requires_grad = False

    def _extract_features(self, pixel_values):
        return self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0, :]

    def forward(self, pixel_values):
        f = self._extract_features(pixel_values)
        return (self.cls_head(f), self.reg_head(f),
                *self.vae(f), self.domain_head(self.grl(f)))

    def _vae_loss(self, weight_pred, weight_true, mu, logvar):
        recon = F.mse_loss(weight_pred, weight_true)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + self.hparams.kl_weight * kl

    def _shared_step(self, batch):
        images, labels, nutrition, portion, domain = batch
        cls_logits, nutrition_pred, weight_pred, mu, logvar, domain_logits = self(images)
        l_cls = self.loss_cls(cls_logits, labels)
        l_reg = self.loss_reg(nutrition_pred, nutrition)
        l_vae = self._vae_loss(weight_pred, portion, mu, logvar)
        l_domain = self.loss_domain(domain_logits, domain)
        total = l_cls + l_reg + l_vae - self.hparams.lambda_domain * l_domain
        return total, l_cls, l_reg, l_vae, l_domain, cls_logits, labels

    def training_step(self, batch, batch_idx):
        total, l_cls, l_reg, l_vae, l_domain, logits, labels = self._shared_step(batch)
        acc = (logits.argmax(-1) == labels).float().mean()
        self.log_dict({"train/loss": total, "train/cls": l_cls, "train/reg": l_reg,
                       "train/vae": l_vae, "train/domain": l_domain, "train/acc": acc},
                      prog_bar=True, on_step=True, on_epoch=True)
        return total

    def validation_step(self, batch, batch_idx):
        total, l_cls, l_reg, l_vae, l_domain, logits, labels = self._shared_step(batch)
        acc = (logits.argmax(-1) == labels).float().mean()
        self.log_dict({"val/loss": total, "val/cls": l_cls, "val/reg": l_reg,
                       "val/vae": l_vae, "val/domain": l_domain, "val/acc": acc},
                      prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
        )
        return {"optimizer": opt,
                "lr_scheduler": CosineAnnealingLR(opt, T_max=self.hparams.epochs)}

    # ── Inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, image: Image.Image, n_samples: int = 50):
        """
        Run full inference on a PIL image.
        Returns a dict with dish name, macros, and portion weight interval.
        """
        self.eval()
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        x = tf(image.convert("RGB")).unsqueeze(0).to(self.device)
        features = self._extract_features(x)

        cls_idx = self.cls_head(features).argmax(-1).item()
        dish_name = DISH_CLASSES[cls_idx]

        nutrition = self.reg_head(features).squeeze(0).cpu().numpy()
        macros = {
            "calories":  round(float(nutrition[0]), 1),
            "protein_g": round(float(nutrition[1]), 1),
            "fat_g":     round(float(nutrition[2]), 1),
            "carbs_g":   round(float(nutrition[3]), 1),
        }

        # VAE uncertainty via reparameterization sampling
        self.vae.train()
        weights = [self.vae(features)[0].item() for _ in range(n_samples)]
        self.vae.eval()

        w_mean = float(np.mean(weights))
        w_std  = float(np.std(weights))

        return {
            "dish":        dish_name,
            "macros":      macros,
            "portion_g":   round(w_mean, 1),
            "portion_std": round(w_std, 1),
            "interval_95": (round(w_mean - 1.96 * w_std, 1),
                            round(w_mean + 1.96 * w_std, 1)),
        }


# ---------------------------------------------------------------------------
# DataModule  (auto-downloads or falls back to synthetic)
# ---------------------------------------------------------------------------

class MMFoodDataModule(pl.LightningDataModule):
    HF_DATASET_ID = "Kaludi/food-101-indian"   # public HF dataset; swap for MM-Food-100K when available

    def __init__(self, batch_size=32, num_workers=4, val_split=0.1,
                 dataset_id: str = None, force_synthetic: bool = False):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_id = dataset_id or self.HF_DATASET_ID
        self.force_synthetic = force_synthetic

    def _try_load_hf(self):
        try:
            from datasets import load_dataset
            print(f"[DataModule] Downloading dataset: {self.dataset_id} ...")
            raw = load_dataset(self.dataset_id)
            print("[DataModule] Dataset loaded from Hugging Face.")
            return raw
        except Exception as e:
            print(f"[DataModule] HF load failed ({e}). Using synthetic data.")
            return None

    def _make_hf_datasets(self, raw):
        if "train" in raw and "validation" not in raw:
            split = raw["train"].train_test_split(test_size=self.hparams.val_split, seed=42)
            return split["train"], split["test"]
        return raw["train"], raw.get("validation", raw.get("test", raw["train"]))

    def setup(self, stage=None):
        if self.force_synthetic:
            raw = None
        else:
            raw = self._try_load_hf()

        if raw is not None:
            train_hf, val_hf = self._make_hf_datasets(raw)
            self.train_ds = MMFood100KDataset(train_hf, domain=0)
            self.val_ds   = MMFood100KDataset(val_hf,   domain=0)
        else:
            n = 2000
            self.train_ds = SyntheticFoodDataset(size=int(n * 0.9))
            self.val_ds   = SyntheticFoodDataset(size=int(n * 0.1))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size,
                          shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)


# ---------------------------------------------------------------------------
# Gradio UI  (camera + file upload)
# ---------------------------------------------------------------------------

def build_gradio_app(model: IndianFoodCalorieModel):
    """
    Launches a Gradio web app with:
      - Tab 1: Upload a photo from disk
      - Tab 2: Live camera capture
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Run: pip install gradio")

    model.eval()

    def _format_result(result: dict) -> tuple:
        m = result["macros"]
        lo, hi = result["interval_95"]
        nutrition_text = (
            f"🍽️  Dish: {result['dish'].replace('_', ' ').title()}\n\n"
            f"🔥 Calories:      {m['calories']} kcal\n"
            f"💪 Protein:       {m['protein_g']} g\n"
            f"🧈 Fat:           {m['fat_g']} g\n"
            f"🌾 Carbohydrates: {m['carbs_g']} g\n\n"
            f"⚖️  Estimated Portion: {result['portion_g']} g\n"
            f"📊 95% Interval:  [{lo} g – {hi} g]"
        )
        return result["dish"].replace("_", " ").title(), nutrition_text

    def predict_from_image(img):
        if img is None:
            return "No image provided", ""
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        result = model.predict(img)
        dish, info = _format_result(result)
        return dish, info

    with gr.Blocks(title="🍛 Indian Food Calorie Estimator") as demo:
        gr.Markdown(
            "# 🍛 Indian Food Calorie Estimator\n"
            "Upload a photo or use your camera to detect the dish and estimate calories."
        )

        with gr.Tabs():
            # ── Tab 1: File Upload ─────────────────────────────────────────
            with gr.Tab("📁 Upload Photo"):
                with gr.Row():
                    upload_input = gr.Image(
                        type="pil", label="Upload food image",
                        sources=["upload"], height=300,
                    )
                    with gr.Column():
                        upload_dish  = gr.Textbox(label="Detected Dish", interactive=False)
                        upload_info  = gr.Textbox(label="Nutrition Info", lines=10,
                                                  interactive=False)
                upload_btn = gr.Button("Estimate Calories", variant="primary")
                upload_btn.click(predict_from_image,
                                 inputs=upload_input,
                                 outputs=[upload_dish, upload_info])

            # ── Tab 2: Camera ──────────────────────────────────────────────
            with gr.Tab("📷 Camera"):
                with gr.Row():
                    camera_input = gr.Image(
                        type="pil", label="Take a photo",
                        sources=["webcam"], height=300,
                    )
                    with gr.Column():
                        camera_dish = gr.Textbox(label="Detected Dish", interactive=False)
                        camera_info = gr.Textbox(label="Nutrition Info", lines=10,
                                                 interactive=False)
                camera_btn = gr.Button("Estimate Calories", variant="primary")
                camera_btn.click(predict_from_image,
                                 inputs=camera_input,
                                 outputs=[camera_dish, camera_info])

        gr.Markdown(
            "---\n"
            "*Nutrition values are estimates. Actual values vary by recipe and portion.*"
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "ui", "both"], default="both",
                        help="train: train model | ui: launch Gradio | both: train then launch UI")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .ckpt file to load for UI mode")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force synthetic dataset (skip HF download)")
    parser.add_argument("--dataset_id", type=str, default=None,
                        help="Hugging Face dataset repo id (overrides default)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    # ── Training ────────────────────────────────────────────────────────
    if args.mode in ("train", "both"):
        model = IndianFoodCalorieModel(epochs=args.epochs)
        dm = MMFoodDataModule(
            batch_size=args.batch_size,
            force_synthetic=args.synthetic,
            dataset_id=args.dataset_id,
        )
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="auto",
            devices="auto",
            precision="16-mixed",
            log_every_n_steps=10,
            gradient_clip_val=1.0,
        )
        trainer.fit(model, datamodule=dm)
        trainer.save_checkpoint("indian_food_model.ckpt")
        print("[Training] Saved checkpoint: indian_food_model.ckpt")

    # ── UI ───────────────────────────────────────────────────────────────
    if args.mode in ("ui", "both"):
        try:
            import gradio as gr
        except ImportError:
            raise ImportError("Run: pip install gradio")

        ckpt = args.checkpoint or "indian_food_model.ckpt"
        if os.path.exists(ckpt):
            print(f"[UI] Loading checkpoint: {ckpt}")
            model = IndianFoodCalorieModel.load_from_checkpoint(ckpt)
        elif args.mode == "ui":
            print("[UI] No checkpoint found — loading untrained model (demo only).")
            model = IndianFoodCalorieModel()

        demo = build_gradio_app(model)
        print(f"[UI] Launching Gradio on http://localhost:{args.port}")
        demo.launch(server_port=args.port, share=False)
n   