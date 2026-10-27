"""Train regression (final score) + classification (pass/fail) models and save them."""
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix)
from generate_data import generate

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "students.csv"
MODELS = ROOT / "models"

FEATURES = ["study_hours", "attendance", "previous_grade", "assignments_completed",
            "sleep_hours", "participation", "parental_support", "extracurricular",
            "internet_access", "tutoring"]


def main():
    if not DATA.exists():
        DATA.parent.mkdir(exist_ok=True)
        generate().to_csv(DATA, index=False)
    df = pd.read_csv(DATA)
    X, y_reg, y_clf = df[FEATURES], df["final_score"], df["passed"]
    Xtr, Xte, ytr_r, yte_r, ytr_c, yte_c = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    reg = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
    reg.fit(Xtr, ytr_r)
    clf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=5, class_weight="balanced", random_state=42)
    clf.fit(Xtr, ytr_c)

    pr, pc = reg.predict(Xte), clf.predict(Xte)
    metrics = {
        "regression": {
            "mae": round(float(mean_absolute_error(yte_r, pr)), 3),
            "r2": round(float(r2_score(yte_r, pr)), 3),
            "cv_r2": round(float(cross_val_score(reg, X, y_reg, cv=5, scoring="r2").mean()), 3),
        },
        "classification": {
            "accuracy": round(float(accuracy_score(yte_c, pc)), 3),
            "precision": round(float(precision_score(yte_c, pc)), 3),
            "recall": round(float(recall_score(yte_c, pc)), 3),
            "f1": round(float(f1_score(yte_c, pc)), 3),
            "confusion_matrix": confusion_matrix(yte_c, pc).tolist(),
        },
        "feature_importance": dict(sorted(
            zip(FEATURES, [round(float(v), 4) for v in reg.feature_importances_]),
            key=lambda kv: -kv[1])),
        "dataset": {"rows": int(len(df)), "pass_rate": round(float(df["passed"].mean()), 3),
                    "avg_score": round(float(df["final_score"].mean()), 2)},
        "features": FEATURES,
    }
    MODELS.mkdir(exist_ok=True)
    joblib.dump(reg, MODELS / "regressor.joblib")
    joblib.dump(clf, MODELS / "classifier.joblib")
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
