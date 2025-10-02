from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "loan_model.joblib")
model = joblib.load(model_path)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    loan_amount = int(data.get("loan_amount", 0))
    applicant_income = int(data.get("applicant_income", 0))
    coapplicant_income = int(data.get("coapplicant_income", 0))  # ✅ include this
    loan_term = int(data.get("loan_term", 0))
    age = int(data.get("age", 0))
    employment_type = int(data.get("employment_type", 0))
    education = int(data.get("education", 0))
    married = int(data.get("married", 0))
    property_area = int(data.get("property_area", 0))

    # Prepare input with 3 features
    features = np.array([[ loan_amount, applicant_income, coapplicant_income, employment_type, education, property_area]])

    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return jsonify({
        "loan_prediction": int(prediction),
        "loan_probability": round(probability, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
