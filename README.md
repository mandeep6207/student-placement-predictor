# Student Placement Predictor (PlacementIQ)

An advanced Flask-based intelligent placement prediction system with multi-model ML comparison, calibrated probabilities, user authentication, database-backed history, PDF reporting, and interactive dashboard charts.

## Highlights

- Multiple ML models trained and compared automatically:
  - Logistic Regression
  - Random Forest
  - XGBoost (if available) or Gradient Boosting fallback
- Metrics comparison for each model:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Best model auto-selected and saved
- Probability calibration using sigmoid calibration
- Dynamic, personalized suggestion engine
- SQLite user authentication and prediction history
- Downloadable PDF prediction report (with visual summaries)
- Interactive charts via Chart.js:
  - Feature importance bar chart
  - Placement probability pie chart
  - Synthetic CGPA histogram

## Tech Stack

- Backend: Flask, SQLite
- ML: scikit-learn (+ optional xgboost)
- Data: pandas, numpy
- PDF: reportlab
- Frontend: Bootstrap 5, Chart.js, custom CSS

## Project Structure

```text
project/
|-- app.py
|-- model.pkl
|-- database.db
|-- README.md
|-- requirements.txt
|-- static/
|   |-- style.css
|   `-- charts.js
|-- templates/
|   |-- base.html
|   |-- index.html
|   |-- dashboard.html
|   |-- history.html
|   |-- login.html
|   `-- signup.html
`-- utils/
    |-- model_train.py
    |-- database.py
    `-- report_generator.py
```

## Input Features

The model predicts placement probability using:

- CGPA (0-10)
- Number of Skills
- Internship Experience (Yes/No)
- Projects Completed
- Communication Skills (1-10)

## API Endpoints

### POST /predict (authenticated)

Request body:

```json
{
  "cgpa": 8.1,
  "skills": 7,
  "internship": "yes",
  "projects": 4,
  "communication": 8
}
```

Response body:

```json
{
  "probability": 88.62,
  "label": "High Chance",
  "badge": "success",
  "suggestions": [
    "Excellent momentum. Keep polishing projects and interview communication to target top-tier opportunities."
  ],
  "feature_importance": {
    "CGPA": 0.33,
    "Skills": 0.22,
    "Internship": 0.18,
    "Projects": 0.15,
    "Communication": 0.12
  },
  "selected_model": "Logistic Regression",
  "prediction_id": 12,
  "probability_breakdown": {
    "placed": 88.62,
    "not_placed": 11.38
  }
}
```

### GET /api/model-insights (authenticated)

Returns selected model name, model metrics table, feature importance map, and CGPA histogram data.

### GET /report/<record_id> (authenticated)

Downloads a PDF report for that prediction record.

## Local Setup

### 1. Clone repository

```bash
git clone https://github.com/mandeep6207/Predictor.git
cd Predictor
```

### 2. Create and activate virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train model (optional, app auto-trains if missing)

```bash
python utils/model_train.py
```

### 5. Run app

```bash
python app.py
```

Open: http://127.0.0.1:5000

## Authentication Flow

- Sign up at `/signup`
- Login at `/login`
- Access:
  - Dashboard: `/dashboard`
  - History: `/history`

Each user sees only their own prediction history and reports.

## Model Training Notes

- Synthetic dataset generated programmatically
- Train/test split with stratification
- Candidate models are trained and calibrated
- Best model selected by F1 score (with accuracy tie-break)
- Bundle saved in `model.pkl` with:
  - model object
  - selected model metadata
  - metrics comparison
  - feature importance
  - synthetic CGPA distribution

## Validation Rules

- CGPA: 0 to 10
- Skills and projects: non-negative
- Communication: 1 to 10

## Deployment Notes

- Set a strong `SECRET_KEY` environment variable in production.
- Disable Flask debug mode in production.
- This project is deployable to services like Render, Azure App Service, or a VM/container.

## License

This project is for learning and portfolio purposes.
