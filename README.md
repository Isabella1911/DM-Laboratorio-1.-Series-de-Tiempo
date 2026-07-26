# DM-Laboratorio-1.-Series-de-Tiempo

Refactor del analisis exploratorio y construccion de series mensuales para el Laboratorio 1 de series de tiempo.

## Estructura minima

- `notebooks/01_eda_y_series.ipynb`: notebook ejecutable de principio a fin.
- `src/lab.py`: pipeline reutilizable para carga, calidad de datos, EDA, construccion de series y diagnostico.
- `data/Base_Migracion_2009-2026jun.xlsx`: dataset fuente.
- `outputs/parte1/`: tablas, series CSV, figuras y `manifest.md`.

## Ejecucion

Usando el entorno virtual del proyecto:

```bash
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -c "from src import lab as L; L.run()"
```

El notebook `notebooks/01_eda_y_series.ipynb` consume el mismo pipeline y regenera las mismas salidas.

## Series generadas

- Total mensual.
- Top 3 fronteras por volumen acumulado en todo el periodo filtrado.
- Vias de ingreso: Aerea, Terrestre y Maritima.

## Notas metodologicas

- Solo se consideran `Turista` y `Excursionista`.
- La agregacion mensual usa suma sobre `Viajero`.
- El corte train/test es cronologico con proporcion `0.70/0.30`.
- El ranking Top 3 de fronteras se calcula con todo el periodo disponible.
- La variable `Pais` puede perder comparabilidad despues de 2023 por cambios a agrupaciones de mercado.
