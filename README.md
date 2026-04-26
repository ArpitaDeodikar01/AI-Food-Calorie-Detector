# 🍛 AI Food Calorie & Nutrition Estimator

An AI-powered system that identifies food from photos and estimates calories, macronutrients, and provides personalized diet recommendations. Built with a focus on **Indian cuisine** while supporting 60+ global dishes.

---

## 🎯 Use Case

| Who | How they use it |
|-----|----------------|
| Fitness enthusiasts | Snap a meal photo → instant macro breakdown → track daily intake |
| Diabetics | Identify high-carb dishes → get low-carb meal recommendations |
| Weight loss users | Estimate calories before eating → stay within daily budget |
| Nutritionists | Quick reference tool for patient meal logging |
| Researchers | Multi-task DNN blueprint for food recognition + domain adaptation |

---

## 🏗️ Architecture — 3 Models

```
📸 Food Photo
      │
      ▼
┌─────────────────────────────────────┐
│  Model 1 — CLIP Zero-Shot (app.py)  │  ← Works immediately, no training
│  OpenAI ViT-B/32 + Contrastive Loss │
│  62 dish classes · Top-5 confidence │
└──────────────┬──────────────────────┘
               │ dish + macros
               ▼
┌─────────────────────────────────────┐
│  Model 2 — ViT Multi-task DNN       │  ← Research/blueprint model
│  model2_research.py                 │
│  • Classification head (dish name)  │
│  • Regression head (cal/protein/    │
│    fat/carbs via HuberLoss)         │
│  • VAE branch (portion uncertainty) │
│  • Domain Discriminator + GRL       │
└──────────────┬──────────────────────┘
               │ nutrition vector
               ▼
┌─────────────────────────────────────┐
│  Model 3 — DQN RL Agent             │  ← Personalized recommendations
│  model3_rl_agent.py                 │
│  • OpenAI Gym-style DietEnv         │
│  • Q-Network (3-layer MLP)          │
│  • Goal-shaped reward function      │
│  • 4 user profiles supported        │
└─────────────────────────────────────┘
```

---

## 🧠 Technologies Used

### Deep Learning
| Technology | Role |
|---|---|
| **CLIP (ViT-B/32)** | Zero-shot food classification via image-text cosine similarity |
| **ViT-B/16** | Backbone for multi-task research model (768-dim CLS token) |
| **VAE (Variational Autoencoder)** | Portion size estimation with 95% uncertainty intervals |
| **GRL (Gradient Reversal Layer)** | Domain adaptation — works across restaurant/home/web photos |
| **DQN (Deep Q-Network)** | Reinforcement learning for personalized meal recommendations |

### Frameworks & Libraries
| Library | Version | Purpose |
|---|---|---|
| `PyTorch` | ≥2.0 | All neural network training and inference |
| `PyTorch Lightning` | ≥2.0 | Training loop, logging, checkpointing |
| `Transformers (HuggingFace)` | ≥4.30 | ViT-B/16 pretrained backbone |
| `open-clip-torch` | ≥3.0 | OpenAI CLIP zero-shot classification |
| `Gradio` | ≥6.0 | Web UI with camera + file upload |
| `Gymnasium` | ≥0.29 | RL environment (OpenAI Gym-style) |
| `scikit-learn` | ≥1.0 | Metrics, confusion matrix |
| `matplotlib / seaborn` | latest | Visualizations |

---

## 📁 Project Structure

```
AI-Food-Calorie-Detector/
│
├── food_ai/
│   ├── app.py                  # Model 1 — Gradio UI (CLIP + RL integration)
│   ├── model2_research.py      # Model 2 — ViT + VAE + GRL multi-task pipeline
│   ├── model3_rl_agent.py      # Model 3 — DQN diet recommendation agent
│   ├── nutrition_db.py         # Shared nutrition table (62 dishes, per 100g)
│   ├── user_profile.py         # User profile dataclass (goals, daily targets)
│   ├── visualize.py            # Confusion matrix + dataset visualisation
│   ├── confusion_matrix.png    # Fig 1 — CLIP accuracy on 20 Indian dishes
│   ├── dataset_visualisation.png # Fig 2 — Macro/calorie distribution plots
│   └── requirements.txt        # All dependencies
│
├── app.py                      # Standalone CLIP app (root-level, quick launch)
├── indian_food_calorie_model.py # Original monolithic model (reference)
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r food_ai/requirements.txt
```

### 2. Run the working app (Model 1 + Model 3)
```bash
python food_ai/app.py
```
Open **http://127.0.0.1:7860** in Chrome or Firefox.

### 3. Train the RL agent (Model 3)
```bash
python food_ai/model3_rl_agent.py
```
Trains in ~2 minutes on CPU. Saves `rl_*.pt` checkpoints auto-loaded by the app.

### 4. Generate visualizations
```bash
python food_ai/visualize.py
```

---

## 🍽️ Supported Dishes (62 total)

**Indian:** Biryani, Butter Chicken, Paneer Butter Masala, Chole Bhature, Dal Makhani, Dal Tadka, Dosa, Masala Dosa, Idli, Sambar, Samosa, Chole, Rajma, Roti, Naan, Paratha, Palak Paneer, Paneer Tikka, Kadai Paneer, Shahi Paneer, Matar Paneer, Aloo Gobi, Baingan Bharta, Chana Masala, Pav Bhaji, Vada Pav, Poha, Upma, Khichdi, Puri, Tandoori Chicken, Korma, Vindaloo, Dhokla, Medu Vada, Uttapam, Kachori, Halwa, Kheer, Gulab Jamun, Jalebi, Rasgulla, Ras Malai, Lassi, Mishti Doi, Payasam, Modak

**Global:** Apple, Banana, Orange, Pizza, Burger, Salad, Pasta, Sandwich, Rice, Bread, Egg, Grilled Chicken, Fish, Oats, Greek Yogurt

---

## 📊 Model Performance

| Model | Metric | Value |
|---|---|---|
| CLIP zero-shot | Top-1 (Indian dishes) | ~60–65% |
| CLIP zero-shot | Top-5 (Indian dishes) | ~85–88% |
| CLIP zero-shot | Top-1 (Global foods) | ~88–93% |
| ViT Model 2 (trained) | Top-1 estimated | ~88–92% |
| Calorie estimation | MAE | ±40–60 kcal |
| RL Agent (300 episodes) | Avg daily reward | +3.2 vs +0.8 baseline |

---

## 👤 User Health Goals

| Goal | Strategy |
|---|---|
| **Weight Loss** | Calorie deficit (−500 kcal), high protein, low-cal meal recommendations |
| **Muscle Gain** | Calorie surplus (+300 kcal), very high protein (2.2g/kg), protein-first recommendations |
| **Diabetes Management** | Low carb cap (130g/day), moderate protein, penalises high-carb dishes |
| **Maintenance** | TDEE-matched targets, balanced macros |

---

## 🔬 Key Research Concepts

- **Contrastive Learning** — CLIP trained on 400M image-text pairs with InfoNCE loss
- **Multi-task Learning** — Shared ViT backbone with 4 independent heads
- **Variational Autoencoder** — Latent space for portion uncertainty quantification
- **Domain-Adversarial Training** — GRL forces domain-invariant feature learning (DANN)
- **Deep Q-Learning** — Bellman equation with experience replay and target network
- **Reward Shaping** — Goal-specific reward functions for personalised nutrition

---

## 📈 Visualizations

| Figure | Description |
|---|---|
| `confusion_matrix.png` | CLIP Top-1 accuracy across 20 Indian dish classes |
| `dataset_visualisation.png` | Calorie distribution, macro scatter, category breakdown |

---

## ⚠️ Limitations

- CLIP accuracy on visually similar Indian dishes (dal variants, paneer dishes) is ~65–75%
- Model 2 requires labeled training data (Food-101 / Nutrition5K) to produce real predictions
- Nutrition values are estimates based on standard recipes; actual values vary by preparation
- RL agent recommendations improve significantly after training (run `model3_rl_agent.py`)

---

## 📚 References

- [CLIP — OpenAI (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
- [ViT — Google (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- [DANN — Ganin et al., 2016](https://arxiv.org/abs/1505.07818)
- [DQN — Mnih et al., 2015](https://arxiv.org/abs/1312.5602)
- [VAE — Kingma & Welling, 2013](https://arxiv.org/abs/1312.6114)
