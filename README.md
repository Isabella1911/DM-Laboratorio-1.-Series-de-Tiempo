# DM-Laboratorio-1.-Series-de-Tiempo

Laboratorio 1 de CC3084 (Data Science, UVG): analisis exploratorio, construccion de series
mensuales, diagnostico de estacionariedad, modelos de pronostico y comparacion estadistica
sobre el ingreso de viajeros internacionales a Guatemala (2009-2026).

> **Nota de uso academico**: el dataset (`data/Base_Migracion_2009-2026jun.xlsx`) es solo para
> fines academicos; no corresponde a datos oficiales del INGUAT ni del Instituto Guatemalteco
> de Migracion.

## Objetivo

Construir una base reproducible para describir, modelar y pronosticar el flujo mensual de
viajeros, cubriendo la serie total y dos categorias adicionales (fronteras principales y vias
de ingreso), con el fin de generar hallazgos utiles para la toma de decisiones del INGUAT.

## Estructura

```text
data/                              Dataset fuente (xlsx)
notebooks/
  01_eda_y_series.ipynb            EDA, series mensuales, split train/test, estacionariedad
  02_modelos_y_pronosticos.ipynb   ARIMA/SARIMA, Prophet, Holt-Winters, exp. smoothing, naive
  03_comparacion_final.ipynb       Comparacion estadistica, hallazgos INGUAT, exportacion
  laboratorio_2_lstm.ipynb         Laboratorio 2: tuneo LSTM y prediccion (2 series)
src/
  lab.py                           Pipeline Parte 1 (carga, calidad, EDA, series, diagnostico)
  models.py                        Pipeline Parte 2 (modelos, diagnostico, pronosticos)
  comparacion.py                   Pipeline Parte 3 (comparacion, hallazgos, exportacion)
  lstm_lab.py                      Pipeline Laboratorio 2 (LSTM + Keras/Torch)
resultados/                        Metricas, predicciones e interpretaciones LSTM
modelos/                           Mejores modelos `.keras` (ignorados en git)
outputs/                           Generado localmente (no versionado, ver .gitignore)
  parte1/{tablas,series,figuras}, manifest.md
  parte2/{pronosticos,residuos,tablas_latex,figuras}, metricas_modelos.csv, mejores_modelos.csv
  parte3/{tablas_latex,figuras}, comparacion_*.csv, hallazgos_inguat.csv, control_calidad.txt
report/
  main.tex                         Informe LaTeX (no incluye codigo, solo resultados)
  figures/, tables/, values/       Copias finales exportadas por la Parte 3 para el informe
```

## Entorno

Requiere Python 3.11 (`pmdarima`/`prophet` no siempre tienen ruedas para versiones mas nuevas).

```bash
py -3.11 -m venv .venv
```

## Instalar dependencias

```bash
./.venv/Scripts/pip install -r requirements.txt      # Windows
./.venv/bin/pip install -r requirements.txt           # Linux/Mac
```

Prophet requiere el backend `cmdstan`. Si no esta instalado, se descarga con:

```bash
./.venv/Scripts/python -m cmdstanpy.install_cmdstan    # Windows
./.venv/bin/python -m cmdstanpy.install_cmdstan         # Linux/Mac
```

Si `prophet` no esta disponible, el pipeline de la Parte 2 lo omite automaticamente y lo deja
documentado en `outputs/parte2/manifest.md` (no interrumpe el resto de modelos).

## Orden de ejecucion

1. `notebooks/01_eda_y_series.ipynb` — genera `outputs/parte1/`.
2. `notebooks/02_modelos_y_pronosticos.ipynb` — lee `outputs/parte1/`, genera `outputs/parte2/`.
3. `notebooks/03_comparacion_final.ipynb` — lee `outputs/parte1/` y `outputs/parte2/`, genera
   `outputs/parte3/` y exporta a `report/{figures,tables,values}`.
4. Compilar `report/main.tex`.

Tambien puede ejecutarse directamente desde Python (equivalente a correr los tres notebooks):

```bash
./.venv/Scripts/python -c "from src import lab as L; L.run()"
./.venv/Scripts/python -c "from src import models as M; M.ejecutar_parte2()"
./.venv/Scripts/python -c "from src import comparacion as C; C.ejecutar_parte3()"
```

### Laboratorio 2 (LSTM)

Requiere `outputs/parte1/series/`. En Python 3.14 use Keras con backend Torch (no hay TensorFlow):

```bash
export KERAS_BACKEND=torch
./.venv/bin/pip install keras torch scikit-learn
./.venv/bin/python -m src.lstm_lab
# o notebooks/laboratorio_2_lstm.ipynb
```

Salidas en `resultados/` y `modelos/`.

## Compilar el informe

```bash
cd report
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

(Se compila dos veces para resolver referencias cruzadas de tablas y figuras.)

## Series generadas

- Total mensual (obligatoria).
- Top 3 fronteras por volumen acumulado en todo el periodo filtrado.
- Vias de ingreso: Aerea, Terrestre y Maritima.

## Notas metodologicas

- Solo se consideran `Turista` y `Excursionista` para mantener comparabilidad 2009-2026.
- La agregacion mensual usa suma sobre `Viajero`.
- El corte train/test es cronologico con proporcion `0.70/0.30` (nunca aleatorio).
- El ranking Top 3 de fronteras se calcula con todo el periodo disponible, no con un ano especifico.
- La variable `Pais` puede perder comparabilidad despues de 2023 por cambios a agrupaciones de mercado.
- Los modelos ARIMA/SARIMA usan la transformacion (log, log1p o Box-Cox aproximada) sugerida en
  la Parte 1 cuando la varianza no es estable; las metricas siempre se calculan revirtiendo la
  transformacion a la escala original.
