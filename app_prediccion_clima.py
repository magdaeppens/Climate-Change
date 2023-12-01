### APP PREDICCION DE CLIMA ###

### PASO 1: Importar librerias
import streamlit as st
import pandas as pd
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

### PASO 2: Preparación de la base

# CARGAMOS EL DATASET #
df = pd.read_csv('ClimateChange.csv')

# ESTANDARIZAMOS LAS FECHAS #
def separar_fecha(df, columna):
    for index, fila in df.iterrows():
        fecha = fila[columna]
        if "-" in fecha:
            fecha_obj = datetime.strptime(fecha, '%Y-%d-%m')
        else:
            fecha_obj = datetime.strptime(fecha, '%d/%m/%Y')
        df.at[index, "dia"] = int(fecha_obj.day)
        df.at[index, "mes"] = int(fecha_obj.month)
        df.at[index, "anio"] = int(fecha_obj.year)
        df.at[index, "dt"] = fecha_obj.strftime('%d/%m/%Y')
    return df


# CONVERTIR LAS COORDENADAS #
def convertir_coordenadas(coordenada):
    deg = float(coordenada[:-1])
    direction = coordenada[-1]
    decimal_coord = deg if direction.upper() in ['N', 'E'] else -deg if direction.upper() in ['S', 'W'] else None
    return decimal_coord

df = separar_fecha(df, "dt")
df['Latitude'] = df['Latitude'].apply(convertir_coordenadas)
df['Longitude'] = df['Longitude'].apply(convertir_coordenadas)
df['dt'] = pd.to_datetime(df['dt'])
df = df[df['dt'].dt.year >= 2000]

# IMPUTACION DE MISSINGS #

df['dt'] = pd.to_datetime(df['dt'])
city_monthly_mean = df.groupby(['City', df['dt'].dt.month])['AverageTemperature'].mean()
city_monthly_mean1 = df.groupby(['City', df['dt'].dt.month])['AverageTemperatureUncertainty'].mean()

for index, row in df.iterrows():
    city = row['City']
    month = row['dt'].month
    if pd.isnull(row['AverageTemperature']):
        mean_value = city_monthly_mean.get((city, month))
        df.at[index, 'AverageTemperature'] = mean_value
    if pd.isnull(row['AverageTemperatureUncertainty']):
        mean_value1 = city_monthly_mean1.get((city, month))
        df.at[index, 'AverageTemperatureUncertainty'] = mean_value1

### PASO 3: FUNCIONES
def esEstacionaria(series):
    result = adfuller(series)
    return result[1] < 0.05

def prediccionTemperaturaPromedioARIMA(pais, cantmeses):
  dfnuevo = df[df['Country'] == pais]
  dfnuevo['anio_mes'] = dfnuevo['dt'].dt.strftime('%Y-%m')
  dfmundo = dfnuevo.groupby('anio_mes')['AverageTemperature'].mean().reset_index()
  dfmundo['anio_mes'] = pd.to_datetime(dfmundo['anio_mes'])
  dfmundo = dfmundo.set_index('anio_mes')
  y = dfmundo['AverageTemperature'].resample('MS').mean()
  dfmundoSERIES = dfmundo.iloc[:, 0]
  X = dfmundoSERIES.values
  d = 0 if esEstacionaria(dfmundoSERIES) else 1
  history = [x for x in dfmundoSERIES.values]
  predictions = []
  order = (3,d,10)
  print(order)
  model = ARIMA(history, order=order)
  model_fit = model.fit()
  predictions = model_fit.forecast(steps=cantmeses)
  fecha_inicio = dfmundoSERIES.index[-1]
  fechas = pd.date_range(start=fecha_inicio, periods=cantmeses + 1, freq='MS')[1:]
  df_predicciones = pd.DataFrame({'Fecha': fechas, 'Prediccion_Temperatura': predictions})
  return df_predicciones

st.title('App de predicción de temperatura')

### PASO 4: Preparación de la app
image = Image.open('foto.jpg')
st.image(image)

countries = [
    "Côte D'Ivoire", "Ethiopia", "India", "Syria", "Egypt", "Turkey",
    "Iraq", "Thailand", "Brazil", "Germany", "Colombia", "South Africa", "Morocco",
    "China", "United States", "Senegal", "Tanzania", "Bangladesh", "Pakistan",
    "Zimbabwe", "Vietnam", "Nigeria", "Indonesia", "Saudi Arabia", "Afghanistan",
    "Ukraine", "Congo (Democratic Republic Of The)", "Peru", "United Kingdom",
    "Angola", "Spain", "Philippines", "Iran", "Australia", "Mexico", "Somalia",
    "Canada", "Russia", "Japan", "Kenya", "France", "Burma", "Italy", "Chile",
    "Dominican Republic", "South Korea", "Singapore", "Taiwan", "Sudan"
]

# PARA SELECCIONAR EL PAIS
selected_country = st.selectbox("Elegí un país:", countries)

# PARA SELECCIONAR EL MES
st.text("")
st.text('Mes')
selected_months=st.slider('<-- Deslizá a los costados -->', min_value=0, max_value=12, value=6, step=1)

# PREDICCION
st.title("Resultado" )
st.write(f"Predicciones de temperatura los próximos {selected_months} meses en {selected_country}:")
predictions_df=prediccionTemperaturaPromedioARIMA(selected_country, selected_months)
st.dataframe(predictions_df, height=300)
