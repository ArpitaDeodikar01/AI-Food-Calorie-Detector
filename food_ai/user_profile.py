"""
User profile dataclass — stores health goals, daily targets, and meal history.
"""

from dataclasses import dataclass, field
from typing import List, Literal


GoalType = Literal["weight_loss", "muscle_gain", "diabetes_management", "maintenance"]


@dataclass
class UserProfile:
    name: str = "User"
    age: int = 25
    weight_kg: float = 70.0
    height_cm: float = 170.0
    goal: GoalType = "maintenance"

    # Daily macro targets (auto-set by goal if not overridden)
    calorie_target: float = 0.0
    protein_target: float = 0.0
    fat_target: float = 0.0
    carbs_target: float = 0.0

    # Running totals for today
    calories_eaten: float = 0.0
    protein_eaten: float = 0.0
    fat_eaten: float = 0.0
    carbs_eaten: float = 0.0
    meals_today: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.calorie_target == 0.0:
            self._set_targets_from_goal()

    def _set_targets_from_goal(self):
        # Mifflin-St Jeor BMR (sedentary activity factor 1.4)
        bmr = 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age + 5
        tdee = bmr * 1.4

        if self.goal == "weight_loss":
            self.calorie_target = tdee - 500
            self.protein_target = self.weight_kg * 1.8   # high protein
            self.fat_target     = self.calorie_target * 0.25 / 9
            self.carbs_target   = (self.calorie_target - self.protein_target * 4
                                   - self.fat_target * 9) / 4

        elif self.goal == "muscle_gain":
            self.calorie_target = tdee + 300
            self.protein_target = self.weight_kg * 2.2   # very high protein
            self.fat_target     = self.calorie_target * 0.25 / 9
            self.carbs_target   = (self.calorie_target - self.protein_target * 4
                                   - self.fat_target * 9) / 4

        elif self.goal == "diabetes_management":
            self.calorie_target = tdee
            self.carbs_target   = 130.0                  # low carb cap
            self.protein_target = self.weight_kg * 1.2
            self.fat_target     = (self.calorie_target - self.protein_target * 4
                                   - self.carbs_target * 4) / 9

        else:  # maintenance
            self.calorie_target = tdee
            self.protein_target = self.weight_kg * 1.2
            self.fat_target     = self.calorie_target * 0.30 / 9
            self.carbs_target   = (self.calorie_target - self.protein_target * 4
                                   - self.fat_target * 9) / 4

    @property
    def remaining_calories(self) -> float:
        return max(0.0, self.calorie_target - self.calories_eaten)

    @property
    def remaining_protein(self) -> float:
        return max(0.0, self.protein_target - self.protein_eaten)

    def log_meal(self, dish: str, calories: float, protein: float,
                 fat: float, carbs: float):
        self.meals_today.append(dish)
        self.calories_eaten += calories
        self.protein_eaten  += protein
        self.fat_eaten      += fat
        self.carbs_eaten    += carbs

    def reset_day(self):
        self.calories_eaten = 0.0
        self.protein_eaten  = 0.0
        self.fat_eaten      = 0.0
        self.carbs_eaten    = 0.0
        self.meals_today    = []

    def summary(self) -> str:
        return (
            f"Goal: {self.goal} | "
            f"Calories: {self.calories_eaten:.0f}/{self.calorie_target:.0f} kcal | "
            f"Protein: {self.protein_eaten:.1f}/{self.protein_target:.1f}g | "
            f"Remaining: {self.remaining_calories:.0f} kcal"
        )


# ── Preset profiles for demo ──────────────────────────────────────────────

PRESET_PROFILES = {
    "weight_loss":         UserProfile(goal="weight_loss",         weight_kg=80, age=30),
    "muscle_gain":         UserProfile(goal="muscle_gain",         weight_kg=75, age=25),
    "diabetes_management": UserProfile(goal="diabetes_management", weight_kg=85, age=50),
    "maintenance":         UserProfile(goal="maintenance",         weight_kg=70, age=28),
}
