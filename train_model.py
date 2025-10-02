import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("loan_data.csv")

# Map categorical feature to integers
data["employment_type"] = data["employment_type"].map({"Unemployed": 0, "Self-Employed": 1, "Salaried": 2})
data["education"] = data["education"].map({"Not Graduate": 0, "Graduate": 1})
data["married"] = data["married"].map({"No": 0, "Yes": 1})
data["property_area"] = data["property_area"].map({"Rural": 0, "Semiurban": 1, "Urban": 2})

# Features and target
feature_columns = ['loan_amount', 'applicant_income', 'coapplicant_income', 'employment_type', 'education', 'property_area']

X = data[feature_columns]
y = data["loan_status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "loan_model.joblib")
print("💾 Model saved as loan_model.joblib")
