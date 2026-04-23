# Student Placement Predictor

A lightweight, production-ready Flask web application that predicts a student's placement probability using a machine learning model.

## Overview

This project uses a trained Logistic Regression model on synthetic student profile data to estimate placement chances and provide actionable improvement suggestions.

## Features

- Placement probability prediction (in percentage)
- Result categories:
  - High Chance (>= 75%)
  - Medium Chance (40% to < 75%)
  - Low Chance (< 40%)
- Personalized suggestions based on profile gaps
- Clean Bootstrap single-page UI
- Reset button and loading spinner
- Input validation on frontend and backend
- Automatic model training if model file is missing

## Tech Stack

- Backend: Flask (Python)
- ML: scikit-learn (Logistic Regression)
- Data: pandas, numpy (synthetic dataset generation)
- Frontend: HTML, CSS, Bootstrap 5

## Project Structure

```
project/
|-- app.py
|-- model.pkl
|-- README.md
|-- requirements.txt
|-- static/
|   `-- style.css
|-- templates/
|   `-- index.html
`-- utils/
    `-- model_train.py
```

## Input Features

The model predicts placement probability from:

- CGPA (0-10)
- Number of Skills
- Internship Experience (Yes/No)
- Projects Completed
- Communication Skills Rating (1-10)

## API

### POST /predict

Request body (JSON):

```json
{
  "cgpa": 8.2,
  "skills": 6,
  "internship": "yes",
  "projects": 4,
  "communication": 8
}
```

Response (JSON):

```json
{
  "probability": 86.37,
  "label": "High Chance",
  "badge": "success",
  "suggestions": [
    "Great profile. Keep refining projects and communication for top opportunities."
  ]
}
```

## Local Setup

### 1. Clone repository

```bash
git clone https://github.com/mandeep6207/Predictor.git
cd Predictor
```

### 2. (Optional) Create virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

App runs at:

- http://127.0.0.1:5000

## Model Training Details

- Synthetic dataset size: 350 rows
- Features: CGPA, Skills, Internship, Projects, Communication
- Target: Placed (0 or 1)
- Train/test split: 80/20
- Model: Logistic Regression
- Accuracy is printed in console during training

To retrain manually:

```bash
python utils/model_train.py
```

## Validation Rules

- CGPA must be between 0 and 10
- Skills and Projects must be non-negative
- Communication must be between 1 and 10

## Improvement Suggestions Logic

- Low CGPA: focus on academic performance
- Low skills: improve technical skill set
- No internship: apply for internships for practical exposure

## Future Improvements

- Use real placement dataset
- Add model explainability (feature importance/SHAP)
- Add authentication and user history tracking
- Deploy on Render or Azure App Service

## License

This project is for learning and portfolio use.
