import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Cargar el modelo y el escalador
modelo = joblib.load('modelo_entrenado.pkl')
scaler = joblib.load('scaler.pkl')

# Inicializar el LabelEncoder y ajustar para el género
label_encoder = LabelEncoder()
label_encoder.fit(["Male", "Female"])  # Ajustar el encoder con los valores conocidos

# Función para preprocesar los datos
def preprocesar_datos(datos):
    # Codificar el género
    datos[2] = label_encoder.transform([datos[2]])[0]
    
    # Convertir el país a variables dummy
    countries = ['France', 'Spain', 'Germany']
    country_dummies = pd.get_dummies([datos[1]], drop_first=False).reindex(columns=countries, fill_value=0).values[0]

    # Quitar las columnas originales categóricas y añadir las dummies
    datos_numericos = datos[:1] + datos[3:]  # Excluir country y gender
    datos_preprocesados = np.concatenate((datos_numericos, country_dummies, [datos[2]]))
    
    # Convertir a un array de numpy y escalar
    datos_preprocesados = np.array(datos_preprocesados, dtype=float).reshape(1, -1)
    datos_preprocesados = scaler.transform(datos_preprocesados)
    
    return datos_preprocesados

# Función para realizar la predicción
def predecir_churn(datos):
    # Preprocesar los datos
    datos_preprocesados = preprocesar_datos(datos)
    
    # Realizar la predicción
    prediccion = modelo.predict(datos_preprocesados)
    
    # Devolver la predicción
    return prediccion[0]

# Función principal de la aplicación
def main():

    # Cargar el archivo CSS
    with open('styles.css') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    
    st.title("Modelo predictivo de avandono de clientes bancarios")
    st.write("Ingrese los datos del cliente para predecir:")

    # Dividir la entrada en dos columnas
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Puntaje de crédito:", min_value=0, max_value=1000, value=500)
        age = st.number_input("Edad:", min_value=18, max_value=100, value=30)
        balance = st.number_input("Saldo en la cuenta:", min_value=0.0, max_value=1e6, value=0.0)
        products_number = st.number_input("Número de productos:", min_value=1, max_value=5, value=1)
        estimated_salary = st.number_input("Salario estimado:", min_value=0.0, max_value=1e6, value=50000.0)

    with col2:
        country = st.selectbox("País:", ["France", "Spain", "Germany"])
        gender = st.selectbox("Género:", ["Male", "Female"])
        tenure = st.number_input("Antigüedad en años:", min_value=0, max_value=10, value=5)
        credit_card = st.selectbox("¿Tiene tarjeta de crédito?", [1, 0])
        active_member = st.selectbox("¿Es miembro activo?", [1, 0])


    col3, col4 = st.columns(2)
    with col3:
        # Botón para realizar la predicción
        if st.button("PREDECIR"):
            datos = [credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary]
            churn_predicho = predecir_churn(datos)
        
            # Imprimir la predicción
            if churn_predicho == 1:
                st.success("El churn predicho es 1: Posible abandono de cliente.")
            else:
                st.success("El churn predicho es 0: No hay riesgo de avandono de cliente.")
    with col4:
        # Botón para resetear los valores
        if st.button("RESETEAR VALORES"):
            st.experimental_rerun()


if __name__ == '__main__':
    main()
