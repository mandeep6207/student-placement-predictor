from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
TRAIN_SCRIPT_PATH = BASE_DIR / "utils" / "model_train.py"

app = Flask(__name__)


def load_model() -> Any:
    """Load the trained model from disk."""
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def get_suggestions(cgpa: float, skills: int, internship: int) -> list[str]:
    suggestions: list[str] = []

    if cgpa < 7.0:
        suggestions.append("Improve academic performance to raise your CGPA.")
    if skills < 5:
        suggestions.append("Learn more technical skills through courses and practice projects.")
    if internship == 0:
        suggestions.append("Apply for internships to gain practical industry experience.")

    if not suggestions:
        suggestions.append("Great profile. Keep refining projects and communication for top opportunities.")

    return suggestions


def get_result_label(probability: float) -> tuple[str, str]:
    if probability >= 75:
        return "High Chance", "success"
    if probability >= 40:
        return "Medium Chance", "warning"
    return "Low Chance", "danger"


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict() -> tuple[Any, int] | Any:
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided."}), 400

        cgpa = float(data.get("cgpa", 0))
        skills = int(data.get("skills", 0))
        internship = 1 if str(data.get("internship", "no")).lower() in {"yes", "1", "true"} else 0
        projects = int(data.get("projects", 0))
        communication = int(data.get("communication", 0))

        if not (0 <= cgpa <= 10):
            return jsonify({"error": "CGPA must be between 0 and 10."}), 400
        if skills < 0 or projects < 0:
            return jsonify({"error": "Skills and projects must be non-negative."}), 400
        if not (1 <= communication <= 10):
            return jsonify({"error": "Communication rating must be between 1 and 10."}), 400

        model = load_model()
        features = [[cgpa, skills, internship, projects, communication]]
        probability = float(model.predict_proba(features)[0][1]) * 100

        label, badge = get_result_label(probability)
        suggestions = get_suggestions(cgpa, skills, internship)

        return jsonify(
            {
                "probability": round(probability, 2),
                "label": label,
                "badge": badge,
                "suggestions": suggestions,
            }
        )
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input. Please check entered values."}), 400
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Unexpected server error: {exc}"}), 500


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print("model.pkl not found. Training model first...")
        os.system(f'python "{TRAIN_SCRIPT_PATH}"')

    app.run(debug=True)
