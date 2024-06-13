import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Cargar el dataset
df = pd.read_csv('Bank Customer Churn Prediction.csv')

# Mostrar las primeras filas del dataset
df.head()

# Describir las características numéricas
df.describe()

# Graficar la distribución de las características numéricas
num_features = ['credit_score', 'age', 'tenure', 'balance', 'estimated_salary']

plt.figure(figsize=(10, 8))
for i, feature in enumerate(num_features):
    plt.subplot(3, 2, i + 1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribución de {feature}')
plt.tight_layout()
plt.show()

# Graficar la distribución de las características categóricas
cat_features = ['country', 'gender', 'products_number', 'credit_card', 'active_member', 'churn']

plt.figure(figsize=(10, 8))
for i, feature in enumerate(cat_features):
    plt.subplot(3, 2, i + 1)
    sns.countplot(data=df, x=feature)
    plt.title(f'Distribución de {feature}')
plt.tight_layout()
plt.show()

# Graficar la distribución de la variable objetivo 'churn'
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='churn')
plt.title('Distribución de la variable objetivo: Churn')
plt.show()
