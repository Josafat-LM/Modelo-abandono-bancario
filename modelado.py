import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC


df = pd.read_csv('Bank Customer Churn Prediction.csv')
df.head()

# Verificar valores faltantes
df.isnull().sum()

# Calcular la mediana de los balances no nulos
median_balance = df[df['balance'] > 0]['balance'].median()

# Reemplazar los balances en 0 con la mediana y agregar ruido
np.random.seed(42)
noise = np.random.normal(0, 5000, df[df['balance'] == 0].shape[0])
df.loc[df['balance'] == 0, 'balance'] = median_balance + noise

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
#==================RNA====================================
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
#model.save('modelo entrenado.keras')

#==================SVM====================================
# Entrenar un SVM
svm_model = SVC(random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Evaluar el modelo
y_pred_svm = svm_model.predict(X_test_scaled)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))
print("Características utilizadas para el aprendizaje:")
print(X.columns)

#==================DEEP LEARNING====================================
# Ejemplo de modificación del modelo con más neuronas y capas
dnn_model = Sequential()
dnn_model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
dnn_model.add(Dropout(0.5))
dnn_model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
dnn_model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
dnn_model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo con una tasa de aprendizaje diferente
dnn_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Entrenar el modelo con diferentes parámetros
history = dnn_model.fit(X_train_scaled, y_train, epochs=150, batch_size=16, validation_split=0.2, verbose=1)


# Evaluar el modelo
dnn_loss, dnn_accuracy = dnn_model.evaluate(X_test_scaled, y_test)
print(f'DNN Test Loss: {dnn_loss}')
print(f'DNN Test Accuracy: {dnn_accuracy}')

# Predicciones y evaluación detallada
y_pred_dnn = (dnn_model.predict(X_test_scaled) > 0.5).astype("int32")

from sklearn.metrics import confusion_matrix, classification_report

print("DNN Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dnn))
print("DNN Classification Report:\n", classification_report(y_test, y_pred_dnn))

dnn_model.save('modelo entrenado2.keras')
