"""
Model 3 — DQN-based Personalized Diet Recommendation Agent
State:  current meal nutrition + user profile (remaining budget, goal)
Action: recommend one of N meals from MEAL_DB
Reward: shaped by goal type (weight loss / muscle gain / diabetes)
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import gymnasium as gym
from gymnasium import spaces

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from food_ai.nutrition_db import NUTRITION_DB, PORTION_G, get_nutrition
from food_ai.user_profile import UserProfile

# ---------------------------------------------------------------------------
# Meal database for RL actions (subset of nutrition_db — balanced meals)
# ---------------------------------------------------------------------------

MEAL_DB = [
    "idli", "dosa", "poha", "upma", "khichdi",
    "dal tadka", "dal makhani", "rajma", "chole", "sambar",
    "palak paneer", "paneer tikka", "kadai paneer", "matar paneer",
    "aloo gobi", "baingan bharta", "chana masala",
    "tandoori chicken", "grilled chicken", "fish",
    "biryani", "roti", "rice", "paratha", "naan",
    "oats", "greek yogurt", "egg", "salad", "lassi",
]
N_MEALS = len(MEAL_DB)


# ---------------------------------------------------------------------------
# Gym Environment
# ---------------------------------------------------------------------------

class DietEnv(gym.Env):
    """
    OpenAI Gym-style environment for diet recommendation.

    State (8-dim):
      [cal_eaten_norm, protein_eaten_norm, fat_eaten_norm, carbs_eaten_norm,
       cal_remaining_norm, protein_remaining_norm, meals_count_norm, goal_encoding]

    Action: integer index into MEAL_DB
    Reward: shaped by goal
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, user_profile: UserProfile):
        super().__init__()
        self.profile = user_profile
        self.observation_space = spaces.Box(
            low=0.0, high=2.0, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_MEALS)
        self._goal_enc = {
            "weight_loss":         0.0,
            "muscle_gain":         0.33,
            "diabetes_management": 0.66,
            "maintenance":         1.0,
        }
        self.reset()

    def _get_obs(self) -> np.ndarray:
        p = self.profile
        return np.array([
            p.calories_eaten  / max(p.calorie_target, 1),
            p.protein_eaten   / max(p.protein_target, 1),
            p.fat_eaten       / max(p.fat_target, 1),
            p.carbs_eaten     / max(p.carbs_target, 1),
            p.remaining_calories / max(p.calorie_target, 1),
            p.remaining_protein  / max(p.protein_target, 1),
            min(len(p.meals_today) / 5.0, 1.0),
            self._goal_enc.get(p.goal, 0.5),
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.profile.reset_day()
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        meal = MEAL_DB[action]
        n = get_nutrition(meal)
        if not n:
            return self._get_obs(), -1.0, False, False, {}

        self.profile.log_meal(meal, n["calories"], n["protein_g"],
                              n["fat_g"], n["carbs_g"])
        reward = self._compute_reward(meal, n)
        self.steps += 1
        done = self.steps >= 5  # 5 meals per episode (day)
        return self._get_obs(), reward, done, False, {"meal": meal, "nutrition": n}

    def _compute_reward(self, meal: str, n: dict) -> float:
        p = self.profile
        reward = 0.0

        # Base: staying within calorie budget
        if p.calories_eaten <= p.calorie_target:
            reward += 1.0
        else:
            over = (p.calories_eaten - p.calorie_target) / p.calorie_target
            reward -= over * 2.0

        # Goal-specific bonuses
        if p.goal == "muscle_gain":
            if n["protein_g"] >= 20:
                reward += 1.5
            protein_ratio = p.protein_eaten / max(p.protein_target, 1)
            reward += min(protein_ratio, 1.0)

        elif p.goal == "weight_loss":
            if n["calories"] < 300:
                reward += 1.0
            if n["protein_g"] >= 15:
                reward += 0.5

        elif p.goal == "diabetes_management":
            if n["carbs_g"] < 30:
                reward += 1.5
            elif n["carbs_g"] > 60:
                reward -= 1.5

        # Diversity penalty — discourage repeating meals
        if meal in p.meals_today[:-1]:
            reward -= 1.0

        return float(reward)

    def render(self):
        print(self.profile.summary())


# ---------------------------------------------------------------------------
# DQN Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 1e-3, gamma: float = 0.95,
                 buffer_size: int = 10_000, batch_size: int = 64):
        self.action_dim = action_dim
        self.gamma      = gamma
        self.batch_size = batch_size

        self.q_net      = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory    = deque(maxlen=buffer_size)
        self.steps     = 0

    def act(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            return self.q_net(s).argmax(dim=1).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s  = torch.FloatTensor(np.array(states))
        a  = torch.LongTensor(actions).unsqueeze(1)
        r  = torch.FloatTensor(rewards).unsqueeze(1)
        ns = torch.FloatTensor(np.array(next_states))
        d  = torch.FloatTensor(dones).unsqueeze(1)

        q_vals  = self.q_net(s).gather(1, a)
        with torch.no_grad():
            next_q = self.target_net(ns).max(1, keepdim=True)[0]
        target  = r + self.gamma * next_q * (1 - d)

        loss = nn.MSELoss()(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % 100 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_agent(profile: UserProfile, episodes: int = 500,
                save_path: str = None) -> DQNAgent:
    env   = DietEnv(profile)
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
    )

    epsilon     = 1.0
    eps_min     = 0.05
    eps_decay   = 0.995
    rewards_log = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_r  = 0.0
        done     = False

        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            agent.remember(state, action, reward, next_state, float(done))
            agent.replay()
            state    = next_state
            total_r += reward

        epsilon = max(eps_min, epsilon * eps_decay)
        rewards_log.append(total_r)

        if (ep + 1) % 100 == 0:
            avg = np.mean(rewards_log[-100:])
            print(f"Episode {ep+1}/{episodes} | Avg Reward: {avg:.2f} | ε: {epsilon:.3f}")

    if save_path:
        torch.save(agent.q_net.state_dict(), save_path)
        print(f"[Model 3] Agent saved to {save_path}")

    return agent


# ---------------------------------------------------------------------------
# Recommend next meal (integration hook for Model 1 / Model 2 output)
# ---------------------------------------------------------------------------

def recommend_next_meal(current_nutrition: dict, user_profile: UserProfile,
                        agent: DQNAgent = None) -> dict:
    """
    Takes Model 1/2 output + user profile → recommends next meal.

    current_nutrition: dict with keys calories, protein_g, fat_g, carbs_g
    """
    # Log the current meal into profile
    user_profile.log_meal(
        current_nutrition.get("dish", "unknown"),
        current_nutrition.get("calories", 0),
        current_nutrition.get("protein_g", 0),
        current_nutrition.get("fat_g", 0),
        current_nutrition.get("carbs_g", 0),
    )

    env   = DietEnv(user_profile)
    state = env._get_obs()

    if agent is None:
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
        )

    action    = agent.act(state, epsilon=0.0)
    meal      = MEAL_DB[action]
    nutrition = get_nutrition(meal)

    return {
        "recommended_meal": meal,
        "nutrition":        nutrition,
        "reason":           f"Optimised for goal: {user_profile.goal}",
        "profile_summary":  user_profile.summary(),
    }


# ---------------------------------------------------------------------------
# Entry point — train agent for all 4 goal types
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from food_ai.user_profile import PRESET_PROFILES

    for goal, profile in PRESET_PROFILES.items():
        print(f"\n=== Training RL agent for: {goal} ===")
        agent = train_agent(profile, episodes=300,
                            save_path=f"food_ai/rl_{goal}.pt")

    # Demo recommendation
    profile = PRESET_PROFILES["muscle_gain"]
    dummy_meal = {"dish": "biryani", "calories": 450, "protein_g": 18,
                  "fat_g": 15, "carbs_g": 66}
    rec = recommend_next_meal(dummy_meal, profile)
    print("\nRecommendation:", rec)
