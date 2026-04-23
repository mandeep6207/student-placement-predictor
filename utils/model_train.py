from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "model.pkl"


def generate_synthetic_data(rows: int = 350, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    cgpa = np.round(np.random.uniform(4.5, 10.0, rows), 2)
    skills = np.random.randint(1, 11, rows)
    internship = np.random.binomial(1, 0.45, rows)
    projects = np.random.randint(0, 8, rows)
    communication = np.random.randint(1, 11, rows)

    # Weighted score with small noise to create realistic placement outcomes.
    score = (
        0.30 * (cgpa / 10)
        + 0.20 * (skills / 10)
        + 0.20 * internship
        + 0.15 * (projects / 7)
        + 0.15 * (communication / 10)
        + np.random.normal(0, 0.05, rows)
    )

    placed = (score > 0.58).astype(int)

    df = pd.DataFrame(
        {
            "CGPA": cgpa,
            "Skills": skills,
            "Internship": internship,
            "Projects": projects,
            "Communication": communication,
            "Placed": placed,
        }
    )
    return df


def train_and_save_model() -> None:
    data = generate_synthetic_data(rows=350)

    X = data[["CGPA", "Skills", "Internship", "Projects", "Communication"]]
    y = data["Placed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)

    print(f"Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()
