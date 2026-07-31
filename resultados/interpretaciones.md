# Interpretaciones — Laboratorio 2 (LSTM)

## Resultados del tuneo — Total mensual

Se evaluaron 4 configuraciones distintas. La configuración con el menor RMSE de validación fue **LSTM_1**, con un RMSE de **110768.93** y un MAE de **75504.63**. Esta configuración utilizó un `look_back` de **6**, **32** unidades en la primera capa LSTM, un dropout de **0.1** y una tasa de aprendizaje de **0.001**.

## Resultados del tuneo — Frontera 01 La Aurora

Se evaluaron 4 configuraciones distintas. La configuración con el menor RMSE de validación fue **LSTM_1**, con un RMSE de **43493.34** y un MAE de **31458.35**. El resultado muestra que **LSTM_1** (look_back=6, capas=1) fue la más estable en validación.

## Comparación e interpretación

Para la serie **Total mensual**, el mejor modelo fue **LSTM_1**. Obtuvo un RMSE de prueba de **50794.23**, un MAE de **40554.45** y un MAPE de **18.05%**.

Para la serie **Frontera 01 La Aurora**, el mejor modelo fue **LSTM_1**, con un RMSE de **19295.30**.

Al comparar ambas series (escalas distintas), el modelo predijo mejor **Total mensual** según MAPE de prueba (18.05%). No se utilizó el conjunto de prueba para seleccionar los hiperparámetros; la selección se realizó exclusivamente con el error de validación.

## Comparación con el laboratorio anterior

El mejor modelo del laboratorio anterior para **Total mensual** fue **HoltWinters(trend=add,seasonal=mul,damped=False)**, con un RMSE de **146605.58**. El modelo LSTM obtuvo un RMSE de **50794.23**. Por lo tanto, **LSTM** produjo el mejor resultado sobre el mismo conjunto de prueba.

En **Frontera 01 La Aurora**, el mejor modelo previo fue **ARIMA(1, 1, 1)x(0, 0, 1, 12)** (RMSE **27986.19**) frente a LSTM (RMSE **19295.30**). El ganador es **LSTM**.

## Conclusiones

1. Se construyeron cuatro configuraciones LSTM por serie y se modificaron `look_back`, unidades, capas, dropout, tasa de aprendizaje y batch size.
2. El mejor modelo para **Total mensual** fue **LSTM_1** (RMSE prueba **50794.23**).
3. El mejor modelo para **Frontera 01 La Aurora** fue **LSTM_1** (RMSE prueba **19295.30**).
4. La serie **Total mensual** fue predicha con mayor precisión según las métricas de prueba.
5. Respecto al laboratorio anterior, LSTM **superó** a los modelos clásicos; la comparación usa el mismo conjunto de prueba.
