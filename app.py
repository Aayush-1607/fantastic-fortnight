"""EduPredict AI - Flask backend. Serves the API and the frontend."""
import io
import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
try:
    from flask_cors import CORS
except ImportError:  # CORS only needed if the frontend is hosted on another origin
    CORS = lambda app: None

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
FRONTEND = ROOT / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND), static_url_path="")
CORS(app)

# Auto-train on first run
if not (MODELS / "regressor.joblib").exists():
    import train_model
    train_model.main()

reg = joblib.load(MODELS / "regressor.joblib")
clf = joblib.load(MODELS / "classifier.joblib")
metrics = json.loads((MODELS / "metrics.json").read_text())
FEATURES = metrics["features"]
BASELINE = pd.read_csv(ROOT / "data" / "students.csv")[FEATURES].mean()

RANGES = {
    "study_hours": (0, 40), "attendance": (0, 100), "previous_grade": (0, 100),
    "assignments_completed": (0, 100), "sleep_hours": (0, 12), "participation": (0, 10),
    "parental_support": (1, 5), "extracurricular": (0, 10), "internet_access": (0, 1),
    "tutoring": (0, 1),
}


def validate(data):
    row = {}
    for f in FEATURES:
        if f not in data or data[f] in ("", None):
            raise ValueError(f"Missing field: {f}")
        try:
            v = float(data[f])
        except (TypeError, ValueError):
            raise ValueError(f"Invalid number for {f}")
        lo, hi = RANGES[f]
        if not lo <= v <= hi:
            raise ValueError(f"{f} must be between {lo} and {hi}")
        row[f] = v
    return row


def grade(score):
    return "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 60 else "D" if score >= 50 else "F"


def risk(prob_pass):
    return "Low" if prob_pass >= 0.8 else "Medium" if prob_pass >= 0.5 else "High"


def recommendations(r):
    tips = []
    if r["study_hours"] < 10: tips.append("Increase weekly study time to at least 10-15 hours.")
    if r["attendance"] < 75: tips.append("Improve class attendance - aim for 85% or higher.")
    if r["assignments_completed"] < 70: tips.append("Complete more assignments; they strongly reinforce learning.")
    if r["sleep_hours"] < 6 or r["sleep_hours"] > 9: tips.append("Aim for 7-8 hours of sleep for better focus and memory.")
    if r["participation"] < 4: tips.append("Participate more in class discussions and ask questions.")
    if r["extracurricular"] > 4: tips.append("Balance extracurricular activities with study time.")
    if r["tutoring"] == 0 and r["previous_grade"] < 60: tips.append("Consider tutoring or peer study groups.")
    if r["internet_access"] == 0: tips.append("Seek access to online learning resources at school or library.")
    if not tips: tips.append("Great habits! Keep up the consistent effort.")
    return tips


def predict_row(row):
    X = pd.DataFrame([row])[FEATURES]
    score = float(max(0, min(100, reg.predict(X)[0])))
    p_pass = float(clf.predict_proba(X)[0][1])
    return {
        "predicted_score": round(score, 1),
        "grade": grade(score),
        "pass_probability": round(p_pass * 100, 1),
        "result": "Pass" if p_pass >= 0.5 else "Fail",
        "risk_level": risk(p_pass),
        "recommendations": recommendations(row),
    }


@app.get("/")
def index():
    return send_from_directory(FRONTEND, "index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok")


@app.get("/api/metrics")
def get_metrics():
    return jsonify(metrics)


@app.post("/api/predict")
def predict():
    try:
        row = validate(request.get_json(force=True, silent=True) or {})
    except ValueError as e:
        return jsonify(error=str(e)), 400
    result = predict_row(row)
    # Factor impact: how much each input moves the score vs. an average student
    raw = float(reg.predict(pd.DataFrame([row])[FEATURES])[0])
    impacts = {}
    for f in FEATURES:
        alt = dict(row)
        alt[f] = float(BASELINE[f])
        impacts[f] = round(raw - float(reg.predict(pd.DataFrame([alt])[FEATURES])[0]), 2)
    result["factor_impact"] = impacts
    return jsonify(result)


@app.post("/api/batch")
def batch():
    if "file" not in request.files:
        return jsonify(error="Upload a CSV file with field name 'file'"), 400
    try:
        df = pd.read_csv(io.BytesIO(request.files["file"].read()))
    except Exception:
        return jsonify(error="Could not read CSV"), 400
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        return jsonify(error=f"Missing columns: {', '.join(missing)}"), 400
    try:
        X = df[FEATURES].astype(float)
    except ValueError:
        return jsonify(error="All feature columns must be numeric"), 400
    if X.isna().any().any():
        return jsonify(error="CSV contains empty values"), 400
    scores = reg.predict(X).clip(0, 100)
    probs = clf.predict_proba(X)[:, 1]
    out = []
    for i in range(len(df)):
        out.append({
            "row": i + 1,
            "name": str(df["name"].iloc[i]) if "name" in df.columns else f"Student {i + 1}",
            "predicted_score": round(float(scores[i]), 1),
            "grade": grade(scores[i]),
            "pass_probability": round(float(probs[i]) * 100, 1),
            "risk_level": risk(probs[i]),
        })
    summary = {"total": len(out),
               "high_risk": sum(o["risk_level"] == "High" for o in out),
               "medium_risk": sum(o["risk_level"] == "Medium" for o in out),
               "low_risk": sum(o["risk_level"] == "Low" for o in out),
               "avg_score": round(float(scores.mean()), 1)}
    return jsonify(summary=summary, results=out)


@app.get("/api/sample-csv")
def sample_csv():
    df = pd.read_csv(ROOT / "data" / "students.csv")[FEATURES].head(15)
    df.insert(0, "name", [f"Student {i+1}" for i in range(len(df))])
    return app.response_class(df.to_csv(index=False), mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_students.csv"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
