# Diabetes Diagnosis AI-ML – Web App

This project uses a Support Vector Machine (SVM) model to predict diabetes risk based on 8 health metrics. The model is served through a Flask web application with a professional UI.
<br>
## Features
- Predicts diabetes risk using a trained SVM model (~77% accuracy)
- Web interface built with Flask and modern HTML/CSS
- Educational information about diabetes and risk factors

## Live Demo
[Insert Render deployment URL here]

## Local Setup
1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run the app: `python app.py`
6. Open `http://127.0.0.1:5000` in your browser

## Files
- `train_model.py` – trains and saves the model
- `app.py` – Flask web application
- `templates/` – HTML pages
- `scaler.joblib` & `svm_model.joblib` – pre‑trained model files

## Author
MUBASSHIR ALI
