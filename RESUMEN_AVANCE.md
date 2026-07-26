# Resumen de avance

- Se centralizo la logica en `src/lab.py` para evitar repetir el mismo analisis siete veces.
- El pipeline genera tablas de calidad de datos, EDA, series mensuales, split train/test, diagnostico de estacionariedad y `manifest.md`.
- Las siete series requeridas quedan exportadas en `outputs/parte1/series/`.
- El notebook `notebooks/01_eda_y_series.ipynb` ya funciona como frontend liviano del pipeline reutilizable.
