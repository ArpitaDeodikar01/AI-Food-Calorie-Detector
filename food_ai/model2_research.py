"""
Model 2 — ViT Multi-task Research Model
Architecture: ViT-B/16 backbone → Classification + Regression + VAE + Domain Discriminator
This is the blueprint/research version. Requires training on Food-101 / Nutrition5K.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    from transformers import ViTModel
    _VIT_AVAILABLE = True
except ImportError:
    _VIT_AVAILABLE = False


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
# VAE Branch — portion size with uncertainty
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
      1. Classification  (dish name)
      2. Regression      (macros: cal, protein, fat, carbs)
      3. VAE             (portion size + uncertainty)
      4. Domain Disc.    (domain-invariant features via GRL)
    """

    def __init__(self, num_classes: int = 101, num_domains: int = 2,
                 lambda_grl: float = 0.1):
        super().__init__()
        if not _VIT_AVAILABLE:
            raise ImportError("pip install transformers")

        self.backbone = ViTModel.from_pretrained(
            "google/vit-base-patch16-224", ignore_mismatched_sizes=True
        )
        # Freeze first 8 layers, fine-tune last 4
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        for i, layer in enumerate(self.backbone.encoder.layer):
            if i < 8:
                for p in layer.parameters():
                    p.requires_grad = False

        D = 768  # ViT hidden dim

        self.cls_head = nn.Sequential(
            nn.Linear(D, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(D, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 4),   # cal, protein, fat, carbs
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
        cls_logits    = self.cls_head(f)
        macro_pred    = self.reg_head(f)
        weight, mu, logvar = self.vae(f)
        domain_logits = self.domain_head(self.grl(f))
        return cls_logits, macro_pred, weight, mu, logvar, domain_logits

    def compute_loss(self, batch):
        images, labels, macros, portions, domains = batch
        cls_l, macro_p, w_p, mu, logvar, dom_l = self(images)

        l_cls    = self.loss_cls(cls_l, labels)
        l_reg    = self.loss_reg(macro_p, macros)
        recon    = F.mse_loss(w_p, portions)
        kl       = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        l_vae    = recon + 0.001 * kl
        l_domain = self.loss_domain(dom_l, domains)

        total = l_cls + l_reg + l_vae - 0.1 * l_domain
        return total, {"cls": l_cls, "reg": l_reg, "vae": l_vae, "domain": l_domain}


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train(model, dataloader, epochs: int = 50, lr: float = 1e-4, device: str = "cpu"):
    """
    Train the multi-task model.
    dataloader must yield: (images, labels, macros, portions, domains)
    """
    model = model.to(device)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            batch = [t.to(device) for t in batch]
            loss, _ = model.compute_loss(batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")

    return model


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def inference(model, image: Image.Image, class_names: list = None,
              n_samples: int = 50, device: str = "cpu") -> dict:
    """
    Run inference on a PIL image.
    If model is untrained, output is clearly labeled as placeholder.
    """
    model.eval().to(device)
    x = _TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        f = model._features(x)
        cls_logits = model.cls_head(f)
        macro_pred = model.reg_head(f).squeeze(0).cpu().numpy()

    cls_idx   = cls_logits.argmax(-1).item()
    dish_name = class_names[cls_idx] if class_names else f"class_{cls_idx}"

    # VAE uncertainty sampling
    model.vae.train()
    weights = []
    with torch.no_grad():
        for _ in range(n_samples):
            w, _, _ = model.vae(f)
            weights.append(w.item())
    model.vae.eval()

    w_mean = float(np.mean(weights))
    w_std  = float(np.std(weights))

    # Check if model has been trained (heuristic: non-trivial weight variance)
    trained = w_std > 0.5
    note = "" if trained else " [UNTRAINED — needs dataset: Food-101 / Nutrition5K]"

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


if __name__ == "__main__":
    print("Model 2 — ViT Multi-task Research Model")
    print("To train: call train(model, dataloader) with Food-101 / Nutrition5K dataset")
    print("To infer: call inference(model, pil_image, class_names)")
    model = FoodMultiTaskModel(num_classes=101)
    dummy = Image.fromarray(
        __import__("numpy").random.randint(0, 255, (224, 224, 3), dtype="uint8")
    )
    result = inference(model, dummy)
    print("Sample inference (untrained):", result)
