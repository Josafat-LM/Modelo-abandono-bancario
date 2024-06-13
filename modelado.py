import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv('Bank Customer Churn Prediction.csv')
df.head()

# Verificar valores faltantes
df.isnull().sum()

# Eliminar filas con valores faltantes (si existen)
df.dropna(inplace=True)

# Inicializar el codificador
label_encoder = LabelEncoder()

# Codificar las variables categóricas
df['country'] = label_encoder.fit_transform(df['country'])
df['gender'] = label_encoder.fit_transform(df['gender'])

X = df.drop(columns=['customer_id', 'churn'])
y = df['churn']

# Dividir los datos en conjuntos de entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Inicializar el escalador
scaler = StandardScaler()

# Ajustar el escalador solo en los datos de entrenamiento
X_train_scaled = scaler.fit_transform(X_train)

# Aplicar el escalador en los datos de prueba
X_test_scaled = scaler.transform(X_test)
#=================================================

# Definir el modelo
model = Sequential()

# Capa de entrada y primera capa oculta
model.add(Dense(units=16, activation='relu', input_shape=(X_train_scaled.shape[1],)))

# Segunda capa oculta
model.add(Dense(units=8, activation='relu'))

# Capa de salida
model.add(Dense(units=1, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=50, batch_size=32)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Generar predicciones
y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")

# Reporte de clasificación
print(classification_report(y_test, y_pred))

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

#guardar modelo
model.save('modelo entrenado.keras')