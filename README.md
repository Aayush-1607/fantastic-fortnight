# 🎓 EduPredict AI
AI-powered student performance prediction using machine learning (scikit-learn) with a Flask backend and a web frontend.

## Features
- **Score prediction** (Gradient Boosting Regressor) and **pass probability** (Random Forest Classifier)
- **Risk level** (Low / Medium / High), letter grade, personalised recommendations
- **Factor impact** chart showing which inputs raise or lower a student's predicted score
- **Batch analysis**: upload a CSV for a whole class, filter by risk, export results
- **Model insights**: accuracy, F1, R², feature importance, confusion matrix

## Quick start
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd backend
python app.py
```
Open **http://localhost:5000**. (Models auto-train on first run; or run `python train_model.py` manually.)
Shortcut: `./run.sh` (Mac/Linux) or `run.bat` (Windows).

## Project structure
```
edupredict-ai/
├── backend/
│   ├── app.py            # Flask API + serves the frontend
│   ├── train_model.py    # Trains and evaluates both models
│   └── generate_data.py  # Creates the synthetic dataset
├── frontend/             # index.html, style.css, app.js (no build step)
├── data/students.csv     # Training data
├── models/               # Saved models + metrics.json
└── requirements.txt
```

## API
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/predict` | JSON with the 10 features → score, grade, pass %, risk, tips, factor impact |
| POST | `/api/batch` | multipart CSV (`file`) → per-student predictions + summary |
| GET | `/api/metrics` | Model metrics and feature importance |
| GET | `/api/sample-csv` | Downloadable sample CSV |

Features: `study_hours, attendance, previous_grade, assignments_completed, sleep_hours, participation, parental_support, extracurricular, internet_access, tutoring`

## Using your own data
Replace `data/students.csv` with real records using the same columns plus `final_score` and `passed` (1 if score ≥ 50), then run `python backend/train_model.py`.

> **Note:** The bundled dataset is synthetic, for demonstration. Predictions on real students should be validated on real data and used to support, not replace, educator judgment.
