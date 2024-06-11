import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

data = pd.read_csv("Bank Customer Churn Prediction.csv")

X = data.drop(columns=["customer_id", "churn"])
y = data["churn"]

label_encoder = LabelEncoder()
X["gender"] = label_encoder.fit_transform(X["gender"])
X = pd.get_dummies(X, columns=["country"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'modelo_entrenado.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Evaluar el modelo
predicciones = model.predict(X_test)
accuracy = accuracy_score(y_test, predicciones)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, predicciones))
