from __future__ import annotations

import os
import pickle
import subprocess
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, session, url_for

from utils.database import (
    add_history_record,
    create_user,
    get_history_record,
    get_user_by_id,
    get_user_by_username,
    get_user_history,
    init_db,
)
from utils.report_generator import build_prediction_report

try:
    from werkzeug.security import check_password_hash, generate_password_hash
except ImportError:
    raise RuntimeError("Werkzeug is required for password hashing.")


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
TRAIN_SCRIPT_PATH = BASE_DIR / "utils" / "model_train.py"
HIGH_CHANCE_THRESHOLD = 75
MEDIUM_CHANCE_THRESHOLD = 40

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
init_db()


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapper


def load_model() -> Any:
    with MODEL_PATH.open("rb") as model_file:
        bundle = pickle.load(model_file)

    if isinstance(bundle, dict) and "model" in bundle:
        return bundle

    # Backward compatibility for old plain-model pickles.
    return {
        "model": bundle,
        "feature_names": ["CGPA", "Skills", "Internship", "Projects", "Communication"],
        "selected_model": "Legacy Logistic Regression",
        "model_metrics": [],
        "best_metrics": {},
        "feature_importance": {
            "CGPA": 0.28,
            "Skills": 0.22,
            "Internship": 0.20,
            "Projects": 0.16,
            "Communication": 0.14,
        },
        "cgpa_distribution": [],
        "calibrated": False,
    }


def get_suggestions(
    cgpa: float,
    skills: int,
    internship: int,
    projects: int,
    communication: int,
    probability: float,
) -> list[str]:
    suggestions: list[str] = []

    if cgpa < 6.2 and internship == 0:
        suggestions.append(
            "Your profile shows both low CGPA and no internship exposure. Prioritize a short internship while improving semester scores."
        )
    elif cgpa < 7.0:
        suggestions.append("Boost academic consistency by targeting a CGPA above 7.5 over the next two terms.")

    if skills >= 7 and communication < 6:
        suggestions.append(
            "Your technical foundation is strong. Add mock interviews and presentations to improve communication clarity."
        )
    elif skills < 5:
        suggestions.append(
            "Increase your technical breadth with 2-3 role-focused skills and apply them in practical mini-projects."
        )

    if projects < 2:
        suggestions.append("Build at least two portfolio projects with clear README files and deployment links.")

    if internship == 0:
        suggestions.append("Apply for internships or remote freelance tasks to gain real-world problem-solving experience.")

    if communication < 5:
        suggestions.append("Practice concise storytelling: explain one project per day in under 90 seconds.")

    if probability >= 75 and not suggestions:
        suggestions.append(
            "Excellent momentum. Keep polishing projects and interview communication to target top-tier opportunities."
        )
    elif probability < 40:
        suggestions.append("Focus on one improvement plan for 6 weeks and re-assess with updated inputs.")

    return suggestions[:5]


def get_result_label(probability: float) -> tuple[str, str]:
    if probability >= HIGH_CHANCE_THRESHOLD:
        return "High Chance", "success"
    if probability >= MEDIUM_CHANCE_THRESHOLD:
        return "Medium Chance", "warning"
    return "Low Chance", "danger"


def parse_internship_flag(value: Any) -> int:
    return 1 if str(value).strip().lower() in {"yes", "1", "true"} else 0


def parse_and_validate_inputs(data: dict[str, Any]) -> dict[str, Any]:
    cgpa = float(data.get("cgpa", 0))
    skills = int(data.get("skills", 0))
    internship = parse_internship_flag(data.get("internship", "no"))
    projects = int(data.get("projects", 0))
    communication = int(data.get("communication", 0))

    if not (0 <= cgpa <= 10):
        raise ValueError("CGPA must be between 0 and 10.")
    if skills < 0 or projects < 0:
        raise ValueError("Skills and projects must be non-negative.")
    if not (1 <= communication <= 10):
        raise ValueError("Communication rating must be between 1 and 10.")

    return {
        "cgpa": cgpa,
        "skills": skills,
        "internship": internship,
        "projects": projects,
        "communication": communication,
    }


def get_current_user() -> dict[str, Any] | None:
    user_id = session.get("user_id")
    if not user_id:
        return None
    return get_user_by_id(int(user_id))


@app.route("/")
def index() -> str:
    if get_current_user():
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/signup", methods=["GET", "POST"])
def signup() -> str | Any:
    if request.method == "POST":
        username = str(request.form.get("username", "")).strip().lower()
        password = str(request.form.get("password", "")).strip()

        if len(username) < 3 or len(password) < 6:
            flash("Username must be at least 3 chars and password at least 6 chars.", "danger")
            return render_template("signup.html")

        created = create_user(username, generate_password_hash(password))
        if not created:
            flash("Username already exists. Try another one.", "danger")
            return render_template("signup.html")

        flash("Account created. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login() -> str | Any:
    if request.method == "POST":
        username = str(request.form.get("username", "")).strip().lower()
        password = str(request.form.get("password", "")).strip()
        user = get_user_by_username(username)

        if not user or not check_password_hash(str(user["password_hash"]), password):
            flash("Invalid username or password.", "danger")
            return render_template("login.html")

        session["user_id"] = int(user["id"])
        session["username"] = str(user["username"])
        flash("Welcome back.", "success")
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
def logout() -> Any:
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


@app.route("/dashboard")
@login_required
def dashboard() -> str:
    return render_template("dashboard.html")


@app.route("/history")
@login_required
def history() -> str:
    user_id = int(session["user_id"])
    records = get_user_history(user_id=user_id, limit=150)
    return render_template("history.html", records=records)


@app.route("/api/model-insights")
@login_required
def model_insights() -> Any:
    bundle = load_model()

    cgpa_values = np.array(bundle.get("cgpa_distribution", []), dtype=float)
    if cgpa_values.size:
        counts, edges = np.histogram(cgpa_values, bins=[0, 5, 6, 7, 8, 9, 10])
        histogram = {
            "labels": ["0-5", "5-6", "6-7", "7-8", "8-9", "9-10"],
            "values": counts.tolist(),
        }
    else:
        histogram = {"labels": [], "values": []}

    return jsonify(
        {
            "selected_model": bundle.get("selected_model", "Unknown"),
            "model_metrics": bundle.get("model_metrics", []),
            "feature_importance": bundle.get("feature_importance", {}),
            "cgpa_histogram": histogram,
        }
    )


@app.route("/predict", methods=["POST"])
@login_required
def predict() -> tuple[Any, int] | Any:
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided."}), 400

        parsed = parse_and_validate_inputs(data)
        bundle = load_model()
        model = bundle["model"]

        features = [
            [
                parsed["cgpa"],
                parsed["skills"],
                parsed["internship"],
                parsed["projects"],
                parsed["communication"],
            ]
        ]
        probability = float(model.predict_proba(features)[0][1]) * 100

        label, badge = get_result_label(probability)
        suggestions = get_suggestions(
            cgpa=parsed["cgpa"],
            skills=parsed["skills"],
            internship=parsed["internship"],
            projects=parsed["projects"],
            communication=parsed["communication"],
            probability=probability,
        )

        record_id = add_history_record(
            user_id=int(session["user_id"]),
            inputs=parsed,
            probability=round(probability, 2),
            label=label,
            suggestions=suggestions,
        )

        return jsonify(
            {
                "probability": round(probability, 2),
                "label": label,
                "badge": badge,
                "suggestions": suggestions,
                "feature_importance": bundle.get("feature_importance", {}),
                "selected_model": bundle.get("selected_model", "Unknown"),
                "prediction_id": record_id,
                "probability_breakdown": {
                    "placed": round(probability, 2),
                    "not_placed": round(100 - probability, 2),
                },
            }
        )
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc) or "Invalid input. Please check entered values."}), 400
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Unexpected server error: {exc}"}), 500


@app.route("/report/<int:record_id>")
@login_required
def download_report(record_id: int) -> Any:
    user_id = int(session["user_id"])
    record = get_history_record(record_id=record_id, user_id=user_id)
    if not record:
        flash("Report not found.", "danger")
        return redirect(url_for("history"))

    bundle = load_model()
    report_buffer = build_prediction_report(
        username=str(session.get("username", "student")),
        record=record,
        feature_importance=bundle.get("feature_importance", {}),
        model_name=str(bundle.get("selected_model", "Unknown")),
    )

    return send_file(
        report_buffer,
        as_attachment=True,
        download_name=f"placement_report_{record_id}.pdf",
        mimetype="application/pdf",
    )


if __name__ == "__main__":
    init_db()

    if not MODEL_PATH.exists():
        print("model.pkl not found. Training model first...")
        subprocess.run(["python", str(TRAIN_SCRIPT_PATH)], check=False)

    app.run(debug=True)
