import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
import streamlit as st

##########################################################################################

# Importar base de datos
file_path = "base_final .xlsx"
data = pd.read_excel(file_path)

# Preprar base de datos para el modelo


data = data.dropna(subset=['grade']) 

# Definir variables 
X = data.drop('grade', axis=1)
y = data['grade']

# Convertir el ingreso anual a formato numerico 
X['annual_inc'] = pd.to_numeric(X['annual_inc'], errors='coerce')

# Agrupar las categorías poco comunes en 'title'
frecuencia_titulos = X['title'].value_counts()
titulos_comunes = frecuencia_titulos[frecuencia_titulos > 10].index
X['title'] = X['title'].where(X['title'].isin(titulos_comunes), 'Otros')

# Definir variables categoricas
columnas_categoricas = ['term', 'emp_length', 'title', "home_ownership"]

# Convertir en variables dummies
X = pd.get_dummies(X, columns=columnas_categoricas, drop_first=True)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Verificar que no hayan Na´s 
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test) 

# Entrenar el modelo SVM
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# Hacer predicciones y evaluar el modelo
y_pred = clf.predict(X_test)


##########################################################################################

# Iniciar la interfaz
st.title("Modelo de predicción de Calificación Crediticia")

st.subheader("Estadisticas descriptivas: ")

# Graficos y estadistica descriptiva
fig, ax = plt.subplots()
ax.hist(data["loan_amnt"], bins = 20, color = "blue", edgecolor = "black")
ax.set_title("Histograma de los montos de creditos")
ax.set_xlabel("Monto")
ax.set_ylabel("Frecuencia")
st.pyplot(fig)

fig1, ax = plt.subplots()
ax.boxplot(data["loan_amnt"])
ax.set_title("Box-Plot de los montos de credito")
ax.set_xlabel("Monto de credito")
st.pyplot(fig1)

# Definir los rangos y las etiquetas
bins = [0, 1000, 2000, 5000, 10000, 20000, 30000, 40000]
labels = ['<1000', '1000-2000', '2000-5000', '5000-10000', '10000-20000', '20000-30000', '30000-40000']

# Crear una nueva columna con los rangos categorizados
data['loan_range'] = pd.cut(data['loan_amnt'], bins=bins, labels=labels, right=False)
# Calcular la cantidad de personas en cada rango
counts = data['loan_range'].value_counts(sort=False)

# Calcular el porcentaje
percentages = (counts / counts.sum()) * 100

# Crear un DataFrame con los resultados
percentage_table = pd.DataFrame({
    'Range': labels,
    'Percentage': percentages
}).fillna(0)  # Llenar con 0 si hay rangos sin valores

st.write("Tabla de porcentajes de creditos otorgados")
st.table(percentage_table)

# Mostrar la tasa de interes por cada score crediticio
tasas = data.groupby("grade")["int_rate"].agg(['max', 'min', 'mean']).reset_index()
tasas.columns = ["grade", "max_int_rate", "min_int_rate", "mean_int_rate"]
st.write("Tasas de interés cobradas a cada grado de calificación crediticia")
st.table(tasas)

# tasa de presicion del modelo
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Classification Report:\n", classification_report(y_test, y_pred))

#############################################################################################

# ingresar nuevos datos
loan_amnt = st.sidebar.number_input("Monto del credito (en USD): ", value = 5000)
annual_inc = st.sidebar.number_input("ingresos anuales: ", value = 75000)
int_rate = st.sidebar.number_input("Tasa de interes (%): ", value = 10)

meses = ["36 months", "60 months"]
term = st.sidebar.selectbox("Ingresar plazo del credito (en meses)", meses)

periodo_trabajo = ["<1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years",
                   "7 years", "8 years", "9 years", "10+ years"]
emp_length = st.sidebar.selectbox("Ingresa cuantos años lleva en su actual trabajo: ", periodo_trabajo)

proposito = ["Business", "Credit card refinancing", "Debt consolidation", "Car financing", "Major purchase",
             "Home improvement", "Home buying", "Medical expenses","Vacation", "Other"]
title = st.sidebar.selectbox("Proposito del credito: ", proposito)

# Data Frame con los datos del usuario
data_input = pd.DataFrame({
    "loan_amnt": [loan_amnt],
    'annual_inc': [annual_inc],
    'int_rate': [int_rate],
    'term': [term],
    'emp_length': [emp_length],
    'title': [title]
})

data_input = pd.get_dummies(data_input, columns=['term', 'emp_length', 'title'], drop_first=True)

# Asegurar que las columnas coincidan
missing_cols = set(X.columns) - set(data_input.columns)
for col in missing_cols:
    data_input[col] = 0
data_input = data_input[X.columns]

# Imputar y escalar los datos
data_input = imputer.transform(data_input)
data_input = scaler.transform(data_input)

# Hacer la predicción
prediccion = clf.predict(data_input)

# Mostrar la predicción
st.write("La calificación crediticia predicha es:", prediccion[0])

#def prediccion_credito(prediccion, loan_amnt, annual_inc):
rangos = {
        "A": 0.6,
        "B": 0.45,
        "C": 0.4,
        "D": 0.35,
        "E": 0.3,
        "F": 0.25,
        "G": 0.2   
    }
    
umbral = rangos.get(prediccion[0], 0) 
    
ratio = loan_amnt / annual_inc 
st.write(f"El ratio deuda/ingreso es: {ratio}")

credito_recomendado = ratio * loan_amnt
if ratio <= umbral:
    st.write("Se recomienda dar credito")
else:
    st.write("No se recomienda dar credito a esta tasa")
    st.write(f"Se recomienda dar credito en un monto máximo de: ${credito_recomendado}")








# python -m streamlit run "C:\Users\gadyh\OneDrive\Documentos\UNISABANA\proyecto_quant_str.py"
