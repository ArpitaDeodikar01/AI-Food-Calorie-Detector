"""
Model 2 — ViT Multi-task Research Model
Architecture: ViT-B/16 backbone → Classification + Regression + VAE + Domain Discriminator

Dataset: Auto-downloads Food-101 from HuggingFace (no manual setup needed).
         Falls back to synthetic data if download fails.

Run:
    python food_ai/model2_research.py --epochs 10 --batch_size 16
"""

import os
import sys
import random
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from food_ai.nutrition_db import NUTRITION_DB, PORTION_G, DISH_NAMES

logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    from transformers import ViTModel
    _VIT_AVAILABLE = True
except ImportError:
    _VIT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

AUGMENT = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Dataset — HuggingFace Food-101 (auto-download)
# ---------------------------------------------------------------------------

class Food101HFDataset(Dataset):
    """
    Wraps HuggingFace food101 dataset.
    Nutrition labels are looked up from nutrition_db by class name.
    Portion sizes are sampled with realistic noise.
    """

    # Food-101 label index → nutrition_db key mapping (best-effort)
    FOOD101_TO_DB = {
        "bibimbap": "rice",          "fried_rice": "rice",
        "chicken_curry": "korma",    "chicken_wings": "tandoori_chicken",
        "samosa": "samosa",          "gulab_jamun": "gulab_jamun",
        "naan": "naan",              "dal_makhani": "dal makhani",
        "idli": "idli",              "dosa": "dosa",
        "pizza": "pizza",            "hamburger": "burger",
        "spaghetti_bolognese": "pasta", "caesar_salad": "salad",
        "omelette": "egg",           "grilled_salmon": "fish",
        "greek_salad": "salad",      "bread_pudding": "bread",
        "waffles": "bread",          "pancakes": "bread",
    }

    def __init__(self, hf_split, augment: bool = False):
        self.data    = hf_split
        self.augment = augment
        self.tf      = AUGMENT if augment else TRANSFORM
        # Build label list from dataset features
        self.id2label = hf_split.features["label"].int2str \
            if hasattr(hf_split.features["label"], "int2str") \
            else lambda x: str(x)

    def __len__(self):
        return len(self.data)

    def _get_nutrition(self, label_name: str):
        # Map Food-101 name to our DB
        key = self.FOOD101_TO_DB.get(label_name, None)
        if key is None:
            # Try direct match (replace _ with space)
            key = label_name.replace("_", " ")
        if key not in NUTRITION_DB:
            # Fallback: random realistic values
            return [random.uniform(100, 400),
                    random.uniform(3, 25),
                    random.uniform(2, 20),
                    random.uniform(10, 60)]
        return list(NUTRITION_DB[key])

    def __getitem__(self, idx):
        sample     = self.data[idx]
        image      = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image      = self.tf(image.convert("RGB"))

        label      = int(sample["label"])
        label_name = self.id2label(label)

        nutrition_per100 = self._get_nutrition(label_name)
        portion   = random.uniform(150, 400)                    # grams
        nutrition = torch.tensor(
            [v * portion / 100.0 for v in nutrition_per100],
            dtype=torch.float32,
        )
        portion_t = torch.tensor([portion], dtype=torch.float32)
        domain    = torch.tensor(0, dtype=torch.long)           # source domain

        return image, label, nutrition, portion_t, domain


# ---------------------------------------------------------------------------
# Synthetic fallback dataset
# ---------------------------------------------------------------------------

class SyntheticFoodDataset(Dataset):
    """Random images + realistic nutrition — used when HF download fails."""

    def __init__(self, size: int = 2000, num_classes: int = 101):
        self.size        = size
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        label     = idx % self.num_classes
        dish      = DISH_NAMES[label % len(DISH_NAMES)]
        per100    = list(NUTRITION_DB[dish])
        portion   = random.uniform(150, 400)
        nutrition = torch.tensor(
            [v * portion / 100.0 + random.gauss(0, 2) for v in per100],
            dtype=torch.float32,
        )
        portion_t = torch.tensor([portion], dtype=torch.float32)
        domain    = torch.tensor(0, dtype=torch.long)
        arr       = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image     = TRANSFORM(Image.fromarray(arr))
        return image, label, nutrition, portion_t, domain


# ---------------------------------------------------------------------------
# DataLoader builder — auto-downloads Food-101 or falls back to synthetic
# ---------------------------------------------------------------------------

def get_dataloaders(batch_size: int = 32, num_workers: int = 2,
                    force_synthetic: bool = False):
    """
    Returns (train_loader, val_loader, num_classes, class_names).
    Tries HuggingFace food101 first, falls back to synthetic.
    """
    if not force_synthetic:
        try:
            from datasets import load_dataset
            print("[Model 2] Downloading Food-101 from HuggingFace...")
            ds = load_dataset("food101", trust_remote_code=True)
            print(f"[Model 2] Food-101 loaded — "
                  f"train: {len(ds['train'])}, val: {len(ds['validation'])}")

            train_ds = Food101HFDataset(ds["train"],      augment=True)
            val_ds   = Food101HFDataset(ds["validation"], augment=False)
            num_classes  = ds["train"].features["label"].num_classes
            class_names  = ds["train"].features["label"].names

        except Exception as e:
            print(f"[Model 2] HF download failed ({e}). Using synthetic data.")
            force_synthetic = True

    if force_synthetic:
        num_classes = 101
        class_names = [f"class_{i}" for i in range(num_classes)]
        train_ds    = SyntheticFoodDataset(size=2000, num_classes=num_classes)
        val_ds      = SyntheticFoodDataset(size=400,  num_classes=num_classes)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, num_classes, class_names


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambda_ * grad, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 0.1):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFn.apply(x, self.lambda_)


# ---------------------------------------------------------------------------
# VAE Branch
# ---------------------------------------------------------------------------

class VAEBranch(nn.Module):
    def __init__(self, in_dim: int = 768, latent_dim: int = 32):
        super().__init__()
        self.fc_mu     = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)
        self.decoder   = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Softplus(),
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def forward(self, x):
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


# ---------------------------------------------------------------------------
# Full Multi-task Model
# ---------------------------------------------------------------------------

class FoodMultiTaskModel(nn.Module):
    """
    Shared ViT-B/16 backbone → 4 heads:
      1. Classification  (dish name, num_classes)
      2. Regression      (macros: cal, protein, fat, carbs)
      3. VAE             (portion size + 95% uncertainty interval)
      4. Domain Disc.    (domain-invariant features via GRL)
    """

    def __init__(self, num_classes: int = 101, num_domains: int = 2,
                 lambda_grl: float = 0.1):
        super().__init__()
        if not _VIT_AVAILABLE:
            raise ImportError("pip install transformers")

        self.num_classes = num_classes
        self.backbone    = ViTModel.from_pretrained(
            "google/vit-base-patch16-224", ignore_mismatched_sizes=True
        )
        # Freeze first 8 layers, fine-tune last 4
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        for i, layer in enumerate(self.backbone.encoder.layer):
            if i < 8:
                for p in layer.parameters():
                    p.requires_grad = False

        D = 768
        self.cls_head = nn.Sequential(
            nn.Linear(D, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(D, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 4),
        )
        self.vae = VAEBranch(in_dim=D, latent_dim=32)
        self.grl = GradientReversalLayer(lambda_=lambda_grl)
        self.domain_head = nn.Sequential(
            nn.Linear(D, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_domains),
        )

        self.loss_cls    = nn.CrossEntropyLoss()
        self.loss_reg    = nn.HuberLoss(delta=1.0)
        self.loss_domain = nn.CrossEntropyLoss()

    def _features(self, x):
        return self.backbone(pixel_values=x).last_hidden_state[:, 0, :]

    def forward(self, x):
        f = self._features(x)
        return (self.cls_head(f), self.reg_head(f),
                *self.vae(f), self.domain_head(self.grl(f)))

    def compute_loss(self, batch):
        images, labels, macros, portions, domains = batch
        cls_l, macro_p, w_p, mu, logvar, dom_l = self(images)

        l_cls    = self.loss_cls(cls_l, labels)
        l_reg    = self.loss_reg(macro_p, macros)
        recon    = F.mse_loss(w_p, portions)
        kl       = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        l_vae    = recon + 0.001 * kl
        l_domain = self.loss_domain(dom_l, domains)
        total    = l_cls + l_reg + l_vae - 0.1 * l_domain

        acc = (cls_l.argmax(-1) == labels).float().mean().item()
        return total, {"cls": l_cls.item(), "reg": l_reg.item(),
                       "vae": l_vae.item(), "domain": l_domain.item(), "acc": acc}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, train_loader, val_loader=None,
          epochs: int = 50, lr: float = 1e-4, device: str = "cpu",
          save_path: str = "food_ai/model2.pt"):

    model = model.to(device)
    opt   = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        t_loss, t_acc = 0.0, 0.0
        for batch in train_loader:
            batch = [b.to(device) for b in batch]
            loss, info = model.compute_loss(batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            t_loss += loss.item()
            t_acc  += info["acc"]

        scheduler.step()
        n = len(train_loader)
        print(f"Epoch {epoch+1:>3}/{epochs} | "
              f"Loss: {t_loss/n:.4f} | Acc: {t_acc/n*100:.1f}%", end="")

        # ── Validate ───────────────────────────────────────────────────
        if val_loader:
            model.eval()
            v_loss, v_acc = 0.0, 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = [b.to(device) for b in batch]
                    loss, info = model.compute_loss(batch)
                    v_loss += loss.item()
                    v_acc  += info["acc"]
            nv = len(val_loader)
            print(f" | Val Loss: {v_loss/nv:.4f} | Val Acc: {v_acc/nv*100:.1f}%", end="")

            if v_loss / nv < best_val_loss:
                best_val_loss = v_loss / nv
                torch.save(model.state_dict(), save_path)
                print(" ✓ saved", end="")

        print()

    print(f"\n[Model 2] Training complete. Best checkpoint: {save_path}")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def inference(model, image: Image.Image, class_names: list = None,
              n_samples: int = 50, device: str = "cpu") -> dict:
    model.eval().to(device)
    x = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        f          = model._features(x)
        cls_logits = model.cls_head(f)
        macro_pred = model.reg_head(f).squeeze(0).cpu().numpy()

    cls_idx   = cls_logits.argmax(-1).item()
    dish_name = class_names[cls_idx] if class_names else f"class_{cls_idx}"

    model.vae.train()
    weights = []
    with torch.no_grad():
        for _ in range(n_samples):
            w, _, _ = model.vae(f)
            weights.append(w.item())
    model.vae.eval()

    w_mean  = float(np.mean(weights))
    w_std   = float(np.std(weights))
    trained = w_std > 0.5
    note    = "" if trained else " [UNTRAINED]"

    return {
        "dish":        dish_name + note,
        "calories":    round(float(macro_pred[0]), 1),
        "protein_g":   round(float(macro_pred[1]), 1),
        "fat_g":       round(float(macro_pred[2]), 1),
        "carbs_g":     round(float(macro_pred[3]), 1),
        "portion_g":   round(w_mean, 1),
        "portion_std": round(w_std, 1),
        "interval_95": (round(w_mean - 1.96 * w_std, 1),
                        round(w_mean + 1.96 * w_std, 1)),
        "trained":     trained,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--synthetic",  action="store_true",
                        help="Skip HF download, use synthetic data")
    parser.add_argument("--infer_only", action="store_true",
                        help="Skip training, just run a sample inference")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Model 2] Device: {device}")

    if args.infer_only:
        # Quick inference demo with untrained model
        model = FoodMultiTaskModel(num_classes=101)
        dummy = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype="uint8"))
        result = inference(model, dummy)
        print("Sample inference:", result)
    else:
        # Auto-download dataset + train
        train_loader, val_loader, num_classes, class_names = get_dataloaders(
            batch_size=args.batch_size,
            force_synthetic=args.synthetic,
        )
        print(f"[Model 2] Classes: {num_classes} | "
              f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

        model = FoodMultiTaskModel(num_classes=num_classes)
        model = train(model, train_loader, val_loader,
                      epochs=args.epochs, lr=args.lr, device=device)

        # Sample inference after training
        print("\n[Model 2] Running sample inference on random image...")
        dummy  = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype="uint8"))
        result = inference(model, dummy, class_names=class_names, device=device)
        print("Result:", result)
