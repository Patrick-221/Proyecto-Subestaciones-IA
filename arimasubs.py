import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Cargar el dataset
data = pd.read_csv('electricity-consumption-processed.csv', sep=';', parse_dates=['datetime'])
data.set_index('datetime', inplace=True)

# Agrupar por día y calcular el consumo máximo
daily_max_consumption = data.groupby(data.index.date)['consumption'].max().reset_index()
daily_max_consumption.columns = ['date', 'max_consumption']
daily_max_consumption.set_index('date', inplace=True)

# Filtrar para quedarte solo con los consumos máximos por día
daily_max_consumption = daily_max_consumption[daily_max_consumption['max_consumption'].notna()]

# Establecer la frecuencia del índice a diaria
daily_max_consumption = daily_max_consumption.asfreq('D')

# Ajustar el modelo ARIMA
model = ARIMA(daily_max_consumption['max_consumption'], order=(1, 1, 1))
results = model.fit()

# Realizar predicciones para todos los días en el dataset
pred = results.get_prediction(start=daily_max_consumption.index[0], end=daily_max_consumption.index[-1])
pred_mean = pred.predicted_mean
pred_conf = pred.conf_int()

# Graficar la comparación
plt.figure(figsize=(12, 6))

# Graficar los valores reales
plt.plot(daily_max_consumption.index, daily_max_consumption['max_consumption'], label='Consumo Máximo Diario Real', color='blue')

# Graficar las predicciones
plt.plot(pred_mean.index, pred_mean, label='Predicción', color='orange')

# Rellenar la zona de confianza
plt.fill_between(pred_conf.index, pred_conf.iloc[:, 0], pred_conf.iloc[:, 1], color='pink', alpha=0.3)

# Configurar el gráfico
plt.title('Comparación entre Consumo Máximo Diario Real y Predicciones (ARIMA)')
plt.xlabel('Fecha')
plt.ylabel('Consumo')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()

# Mostrar gráfico
plt.show()
