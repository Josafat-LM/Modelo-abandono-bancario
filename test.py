import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import streamlit as st

# Cargar el modelo y el escalador
model = load_model('modelo entrenado.keras')
df = pd.read_csv('Bank Customer Churn Prediction.csv')
df.dropna(inplace=True)
X = df.drop(columns=['customer_id', 'churn'])

# Preprocesar datos categóricos
label_encoder_country = LabelEncoder()
label_encoder_gender = LabelEncoder()
X['country'] = label_encoder_country.fit_transform(X['country'])
X['gender'] = label_encoder_gender.fit_transform(X['gender'])

scaler = StandardScaler()
scaler.fit(X)

# Función para preprocesar los datos del usuario
def preprocesar_datos_usuario(credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary):
    country = label_encoder_country.transform([country])[0]
    gender = label_encoder_gender.transform([gender])[0]
    
    user_data = np.array([[credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary]])
    user_data_scaled = scaler.transform(user_data)
    
    return user_data_scaled

# Función para realizar la predicción
def predecir_churn(user_data):
    prediction = model.predict(user_data)
    predicted_churn = (prediction > 0.5).astype("int32")
    return predicted_churn

# Función principal de la aplicación
def main():
    # Configuración de la página
    st.set_page_config(page_title="Modelo predictivo de abandono de clientes bancarios", layout="centered")
    st.title("Modelo predictivo de abandono de clientes bancarios")
    st.write("Ingrese los datos del cliente para predecir:")

    # Cargar el archivo CSS
    with open('styles.css') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Dividir la entrada en dos columnas
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.text_input("Puntaje de crédito:", value="500")
        age = st.text_input("Edad:", value="30")
        balance = st.text_input("Saldo en la cuenta:", value="0.0")
        products_number = st.text_input("Número de productos:", value="1")
        estimated_salary = st.text_input("Salario estimado:", value="50000.0")
    
    with col2:
        country = st.selectbox("País:", ["France", "Spain", "Germany"])
        gender = st.selectbox("Género:", ["Male", "Female"])
        tenure = st.text_input("Antigüedad en años:", value="5")
        credit_card = st.selectbox("¿Tiene tarjeta de crédito?", [1, 0])
        active_member = st.selectbox("¿Es miembro activo?", [1, 0])

    # Botón para realizar la predicción
    if st.button('PREDECIR'):
        try:
            # Convertir entradas a tipo numérico
            credit_score = float(credit_score)
            age = int(age)
            balance = float(balance)
            products_number = int(products_number)
            estimated_salary = float(estimated_salary)
            tenure = int(tenure)

            user_data = preprocesar_datos_usuario(credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary)
            predicted_churn = predecir_churn(user_data)
            
            # Mostrar la predicción
            if predicted_churn == 0:
                st.success("El cliente probablemente NO abandonará.")
            else:
                st.success("El cliente probablemente abandonará.")
        except ValueError:
            st.error("Por favor, ingrese valores válidos en todos los campos.")

if __name__ == '__main__':
    main()
