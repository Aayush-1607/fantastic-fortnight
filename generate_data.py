"""Generate a realistic synthetic student dataset (replace with real data if you have it)."""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def generate(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    study_hours = np.clip(rng.normal(14, 6, n), 0, 40)            # per week
    attendance = np.clip(rng.normal(82, 13, n), 30, 100)          # %
    previous_grade = np.clip(rng.normal(65, 15, n), 10, 100)      # %
    assignments = np.clip(rng.normal(78, 18, n), 0, 100)          # % completed
    sleep_hours = np.clip(rng.normal(6.8, 1.2, n), 3, 10)         # per night
    participation = np.clip(rng.normal(6, 2.2, n), 0, 10)         # 0-10
    parental_support = rng.integers(1, 6, n)                      # 1-5
    extracurricular = rng.integers(0, 6, n)                       # activities/week
    internet_access = rng.choice([0, 1], n, p=[0.12, 0.88])
    tutoring = rng.choice([0, 1], n, p=[0.72, 0.28])

    sleep_effect = -2.0 * np.abs(sleep_hours - 7.5)
    extra_effect = np.where(extracurricular > 3, -1.5 * (extracurricular - 3), 0.8 * extracurricular)
    score = (
        -10 + 0.9 * study_hours
        + 0.22 * attendance
        + 0.32 * previous_grade
        + 0.10 * assignments
        + 0.9 * participation
        + 1.2 * parental_support
        + extra_effect
        + sleep_effect
        + 2.0 * internet_access
        + 3.0 * tutoring
        + rng.normal(0, 4, n)
    )
    score = np.clip(score, 0, 100)

    df = pd.DataFrame({
        "study_hours": study_hours.round(1),
        "attendance": attendance.round(1),
        "previous_grade": previous_grade.round(1),
        "assignments_completed": assignments.round(1),
        "sleep_hours": sleep_hours.round(1),
        "participation": participation.round(1),
        "parental_support": parental_support,
        "extracurricular": extracurricular,
        "internet_access": internet_access,
        "tutoring": tutoring,
        "final_score": score.round(1),
    })
    df["passed"] = (df["final_score"] >= 50).astype(int)
    return df


if __name__ == "__main__":
    out = ROOT / "data" / "students.csv"
    out.parent.mkdir(exist_ok=True)
    generate().to_csv(out, index=False)
    print(f"Saved dataset -> {out}")
