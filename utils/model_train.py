from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "model.pkl"
FEATURE_NAMES = ["CGPA", "Skills", "Internship", "Projects", "Communication"]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def generate_synthetic_data(rows: int = 600, seed: int = RANDOM_STATE) -> pd.DataFrame:
    np.random.seed(seed)

    cgpa = np.round(np.random.uniform(4.2, 10.0, rows), 2)
    skills = np.random.randint(1, 11, rows)
    internship = np.random.binomial(1, 0.48, rows)
    projects = np.random.randint(0, 9, rows)
    communication = np.random.randint(1, 11, rows)

    score = (
        0.34 * (cgpa / 10)
        + 0.22 * (skills / 10)
        + 0.18 * internship
        + 0.14 * (projects / 8)
        + 0.12 * (communication / 10)
        + np.random.normal(0, 0.05, rows)
    )

    placed = (score > 0.60).astype(int)

    return pd.DataFrame(
        {
            "CGPA": cgpa,
            "Skills": skills,
            "Internship": internship,
            "Projects": projects,
            "Communication": communication,
            "Placed": placed,
        }
    )


def _get_candidate_models() -> list[tuple[str, Any]]:
    candidates: list[tuple[str, Any]] = [
        ("Logistic Regression", LogisticRegression(max_iter=1500, random_state=42)),
        (
            "Random Forest",
            RandomForestClassifier(n_estimators=300, min_samples_split=4, random_state=42),
        ),
    ]

    try:
        from xgboost import XGBClassifier  # type: ignore

        candidates.append(
            (
                "XGBoost",
                XGBClassifier(
                    n_estimators=250,
                    learning_rate=0.08,
                    max_depth=4,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                ),
            )
        )
    except Exception:
        candidates.append(
            (
                "Gradient Boosting",
                GradientBoostingClassifier(random_state=42),
            )
        )

    return candidates


def _extract_feature_importance(model: Any, feature_names: list[str]) -> dict[str, float]:
    importances: np.ndarray

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.asarray(model.coef_[0], dtype=float))
    else:
        importances = np.ones(len(feature_names), dtype=float)

    total = float(importances.sum()) or 1.0
    normalized = (importances / total).tolist()
    return {name: round(val, 6) for name, val in zip(feature_names, normalized)}


def _evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    }


def train_and_save_model() -> None:
    data = generate_synthetic_data(rows=600)

    X = data[FEATURE_NAMES]
    y = data["Placed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    comparison_rows: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None

    for model_name, model in _get_candidate_models():
        model.fit(X_train, y_train)

        calibrated_model = CalibratedClassifierCV(model, method="sigmoid", cv=5)
        calibrated_model.fit(X_train, y_train)

        metrics = _evaluate_model(calibrated_model, X_test, y_test)
        importances = _extract_feature_importance(model, FEATURE_NAMES)

        row = {"model": model_name, **metrics}
        comparison_rows.append(row)

        candidate_payload = {
            "model": calibrated_model,
            "base_model": model,
            "model_name": model_name,
            "metrics": metrics,
            "feature_importance": importances,
        }

        if best_payload is None:
            best_payload = candidate_payload
            continue

        if (
            metrics["f1"],
            metrics["accuracy"],
        ) > (
            best_payload["metrics"]["f1"],
            best_payload["metrics"]["accuracy"],
        ):
            best_payload = candidate_payload

    assert best_payload is not None

    bundle = {
        "model": best_payload["model"],
        "feature_names": FEATURE_NAMES,
        "selected_model": best_payload["model_name"],
        "model_metrics": comparison_rows,
        "best_metrics": best_payload["metrics"],
        "feature_importance": best_payload["feature_importance"],
        "cgpa_distribution": data["CGPA"].round(2).tolist(),
        "calibrated": True,
    }

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(bundle, model_file)

    print("Model comparison complete:")
    for row in comparison_rows:
        print(
            f"- {row['model']}: "
            f"accuracy={row['accuracy']:.4f}, "
            f"precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, "
            f"f1={row['f1']:.4f}"
        )
    print(f"Selected model: {best_payload['model_name']}")
    print(f"Model bundle saved at: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()
