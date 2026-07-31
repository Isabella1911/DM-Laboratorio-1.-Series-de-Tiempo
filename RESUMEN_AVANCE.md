# Resumen de avance

## Laboratorio 1

- Logica centralizada en `src/lab.py`, `src/models.py` y `src/comparacion.py`.
- Siete series mensuales en `outputs/parte1/series/` con split cronologico 70/30.
- Modelos clasicos (ARIMA/SARIMA, Prophet, Holt-Winters, etc.) y comparacion en el informe LaTeX.

## Laboratorio 2 (LSTM) — avance

- Notebook: `notebooks/laboratorio_2_lstm.ipynb`
- Codigo: `src/lstm_lab.py`
- Series usadas (mismos train/test del lab 1):
  - **Total mensual**
  - **Frontera 01 La Aurora**
- 4 configuraciones LSTM por serie; seleccion por RMSE de validacion (no prueba).
- Mejor modelo en ambas: **LSTM_1** (`look_back=6`, 32 unidades, dropout 0.1).
- Prueba (aprox.):
  - Total: RMSE ≈ 50 794, MAPE ≈ 18.1 %
  - La Aurora: RMSE ≈ 19 295, MAPE ≈ 18.8 %
- LSTM mejoro frente a Holt-Winters (total) y ARIMA (La Aurora) del laboratorio anterior.
- Entorno: Keras 3 + backend Torch (Python 3.14 no tiene TensorFlow).

Regenerar:

```bash
export KERAS_BACKEND=torch
python -m src.lstm_lab
```
