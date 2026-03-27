import numpy as np
import joblib
from flask import Flask, render_template, request

# Load the model and scaler using joblib
scaler = joblib.load('scaler.joblib')
model = joblib.load('svm_model.joblib')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form inputs
    features = [
        float(request.form['Pregnancies']),
        float(request.form['Glucose']),
        float(request.form['BloodPressure']),
        float(request.form['SkinThickness']),
        float(request.form['Insulin']),
        float(request.form['BMI']),
        float(request.form['DiabetesPedigreeFunction']),
        float(request.form['Age'])
    ]
    
    # Preprocess
    input_array = np.array(features).reshape(1, -1)
    scaled = scaler.transform(input_array)
    
    # Predict
    pred = model.predict(scaled)[0]
    result = 'Diabetic' if pred == 1 else 'Not Diabetic'
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)