import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import streamlit as st

# Configuración de la página
st.set_page_config(page_title="Modelo predictivo bancario", layout="centered")
st.title("Modelo predictivo de abandono de clientes bancarios")

# Cargar el archivo CSS
with open('styles.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV", type=["csv"])

if uploaded_file is not None:
    user_dataa = pd.read_csv(uploaded_file)
    st.write("Lista de datos cargada")
else:
    # Dividir la entrada en dos columnas
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.text_input("Puntaje de crédito:", value="502")
        country = st.selectbox("País:", ["France", "Spain", "Germany"])
        gender = st.selectbox("Género:", ["Male", "Female"])
        age = st.text_input("Edad:", value="42")
        tenure = st.text_input("Antigüedad en años:", value="8")        
    with col2:
        balance = st.text_input("Saldo en la cuenta:", value="159660.8")
        products_number = st.text_input("Número de productos:", value="3")
        credit_card = st.selectbox("¿Tiene tarjeta de crédito?", [1, 0])
        active_member = st.selectbox("¿Es miembro activo?", [1, 0])
        estimated_salary = st.text_input("Salario estimado:", value="113931.57")

#=================================================
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

# Función para preprocesar un DataFrame de usuarios
def preprocesar_datos_dataframe(df):
    df['country'] = label_encoder_country.transform(df['country'])
    df['gender'] = label_encoder_gender.transform(df['gender'])
    return scaler.transform(df)

# Cargar el modelo
model = load_model('modelo entrenado2.keras')

def predecir_churn(user_data):
    prediction = model.predict(user_data)
    predicted_churn = (prediction > 0.5).astype("int32")
    probability = prediction[:, 0] * 100
    return predicted_churn, probability

# Botón para realizar la predicción
if uploaded_file is not None:
    user_data = preprocesar_datos_dataframe(user_dataa)
    predicted_churn, probability = predecir_churn(user_data)
    resultados = pd.DataFrame({
                    'Cliente': range(1, len(predicted_churn) + 1),
                    'Abandonara': ['Sí' if pc == 1 else 'No' for pc in predicted_churn],
                    'Probabilidad (%)': probability
                })
    opcion_filtrado = st.radio(
        'Filtrar por:',
        ('Mostrar todos', 'Mostrar solo clientes que abandonarán', 'Mostrar solo clientes que NO abandonarán')
    )
    mayor_50=st.checkbox('Mostrar solo predicciones con propabilidades mayores al 50%', value=False)
    # Mostrar la predicción 
    if st.button('PREDECIR'):
        resultados_filtrados = resultados.copy()       
        if opcion_filtrado== 'Mostrar solo clientes que abandonarán':
            resultados_filtrados = resultados_filtrados[resultados_filtrados['Abandonara'] == 'Sí']
            
        elif opcion_filtrado == 'Mostrar solo clientes que NO abandonarán':
            resultados_filtrados = resultados_filtrados[resultados_filtrados['Abandonara'] == 'No']
            
        if mayor_50:
            resultados_filtrados = resultados_filtrados[resultados_filtrados['Probabilidad (%)'] > 50.0]
            
        st.write("Tabla de predicciones:")
        st.write(resultados_filtrados)
                 

else:
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
            
            predicted_churn, probability = predecir_churn(user_data)
            
            # Mostrar la predicción
           
            if predicted_churn[0] == 0:
                col5, col6 = st.columns(2)

                with col5:
                    st.success(f"El cliente abandonará")
                with col6:
                    st.success(f"Probabilidad: {probability[0]:.2f}%")
            else:
                col5, col6 = st.columns(2)

                with col5:
                    st.success(f"El cliente NO abandonará")
                with col6:
                    st.success(f"Probabilidad: {probability[0]:.2f}%")
        except ValueError:
            st.error("Por favor, ingrese valores válidos en todos los campos")

