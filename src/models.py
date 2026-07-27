from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.special import inv_boxcox
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src import lab as L

try:
    from pmdarima import auto_arima as _pmd_auto_arima

    HAS_PMDARIMA = True
except Exception:  # pragma: no cover - optional dependency
    HAS_PMDARIMA = False

try:
    from prophet import Prophet

    HAS_PROPHET = True
except Exception:  # pragma: no cover - optional dependency
    HAS_PROPHET = False

ROOT = L.ROOT
PARTE1_DIR = ROOT / "outputs" / "parte1"
PARTE2_DIR = ROOT / "outputs" / "parte2"
FIGURAS_DIR = PARTE2_DIR / "figuras"
RESIDUOS_DIR = PARTE2_DIR / "residuos"
PRONOSTICOS_DIR = PARTE2_DIR / "pronosticos"
TABLAS_LATEX_DIR = PARTE2_DIR / "tablas_latex"

PERIODO_ESTACIONAL = 12
PANDEMIA_INICIO = pd.Timestamp("2020-03-01")
PANDEMIA_FIN = pd.Timestamp("2021-12-01")

P_MAX = 3
Q_MAX = 3
PQ_ESTACIONAL_MAX = 1


def _ensure_dirs() -> None:
    for path in (PARTE2_DIR, FIGURAS_DIR, RESIDUOS_DIR, PRONOSTICOS_DIR, TABLAS_LATEX_DIR):
        path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Carga de datos generados en la Parte 1
# ---------------------------------------------------------------------------


def cargar_configuracion_series() -> pd.DataFrame:
    path = PARTE1_DIR / "tablas" / "resultados_estacionariedad.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro {path}. Ejecute primero 'from src import lab as L; L.run()'."
        )
    return pd.read_csv(path)


def cargar_serie_train_test(nombre_serie: str) -> tuple[pd.Series, pd.Series]:
    slug = L.slugify(nombre_serie)
    train_path = PARTE1_DIR / "series" / f"{slug}_train.csv"
    test_path = PARTE1_DIR / "series" / f"{slug}_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"No se encontraron series para '{nombre_serie}' en {PARTE1_DIR / 'series'}."
        )
    train_df = pd.read_csv(train_path, parse_dates=["fecha"]).set_index("fecha")
    test_df = pd.read_csv(test_path, parse_dates=["fecha"]).set_index("fecha")
    train = train_df["valor"].asfreq(L.FRECUENCIA)
    test = test_df["valor"].asfreq(L.FRECUENCIA)
    train.name = nombre_serie
    test.name = nombre_serie
    return train, test


# ---------------------------------------------------------------------------
# Transformaciones (heredadas del diagnostico de la Parte 1)
# ---------------------------------------------------------------------------


@dataclass
class Transformacion:
    nombre: str
    forward: Callable[[pd.Series], pd.Series]
    inverse: Callable[[np.ndarray], np.ndarray]


def construir_transformacion(nombre: str, serie_train: pd.Series) -> Transformacion:
    clean = serie_train.dropna()
    if nombre == "log" and (clean > 0).all():
        return Transformacion("log", lambda s: np.log(s), lambda a: np.exp(a))
    if nombre == "log1p" and (clean >= 0).all():
        return Transformacion("log1p", lambda s: np.log1p(s), lambda a: np.expm1(a))
    if nombre == "boxcox_aproximada" and (clean > 0).all():
        _, lmbda = scipy_stats.boxcox(clean.values)

        def _fwd(s: pd.Series, lmbda=lmbda) -> pd.Series:
            return pd.Series(scipy_stats.boxcox(s.values, lmbda=lmbda), index=s.index)

        def _inv(a: np.ndarray, lmbda=lmbda) -> np.ndarray:
            return inv_boxcox(a, lmbda)

        return Transformacion("boxcox_aproximada", _fwd, _inv)
    return Transformacion("sin_transformacion", lambda s: s.copy(), lambda a: a)


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------


def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
    error = y_true - y_pred
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    nonzero = y_true != 0
    mape = float(np.mean(np.abs(error[nonzero] / y_true[nonzero])) * 100) if nonzero.any() else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


# ---------------------------------------------------------------------------
# ARIMA / SARIMA
# ---------------------------------------------------------------------------


def crear_candidatos_arima(
    d: int, D: int, p_max: int = P_MAX, q_max: int = Q_MAX, s: int = PERIODO_ESTACIONAL
) -> list[dict[str, tuple[int, ...]]]:
    candidatos = []
    for p in range(p_max + 1):
        for q in range(q_max + 1):
            candidatos.append({"order": (p, d, q), "seasonal_order": (0, 0, 0, 0)})
    for P in range(PQ_ESTACIONAL_MAX + 1):
        for Q in range(PQ_ESTACIONAL_MAX + 1):
            if P == 0 and Q == 0:
                continue
            candidatos.append({"order": (1, d, 1), "seasonal_order": (P, D, Q, s)})
    return candidatos


def ajustar_sarima(
    serie_train: pd.Series, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]
) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = SARIMAX(
            serie_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return modelo.fit(disp=False)


def ajustar_arima(serie_train: pd.Series, order: tuple[int, int, int]) -> Any:
    return ajustar_sarima(serie_train, order, (0, 0, 0, 0))


def sugerir_auto_arima(serie_train: pd.Series, d: int, D: int, s: int = PERIODO_ESTACIONAL) -> dict[str, Any] | None:
    if not HAS_PMDARIMA:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo = _pmd_auto_arima(
                serie_train,
                d=d,
                D=D,
                seasonal=True,
                m=s,
                max_p=P_MAX,
                max_q=Q_MAX,
                max_P=PQ_ESTACIONAL_MAX,
                max_Q=PQ_ESTACIONAL_MAX,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                trace=False,
            )
        return {
            "order": modelo.order,
            "seasonal_order": modelo.seasonal_order,
            "aic": float(modelo.aic()),
            "bic": float(modelo.bic()),
        }
    except Exception:
        return None


def diagnosticar_residuos(
    fitted: Any, nombre_serie: str, modelo_nombre: str, output_dir: Path = RESIDUOS_DIR
) -> dict[str, Any]:
    _ensure_dirs()
    slug_serie = L.slugify(nombre_serie)
    slug_modelo = L.slugify(modelo_nombre)
    resid = pd.Series(fitted.resid).dropna()

    csv_path = output_dir / f"residuos_{slug_serie}_{slug_modelo}.csv"
    resid.rename("residuo").rename_axis("fecha").reset_index().to_csv(csv_path, index=False)

    ljung = acorr_ljungbox(resid, lags=[12], return_df=True)
    ljung_pvalue = float(ljung["lb_pvalue"].iloc[0])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(resid.index, resid.values, color="#1f77b4", linewidth=1.2)
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set_title("Residuos en el tiempo")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].hist(resid.values, bins=20, color="#4c78a8", edgecolor="white")
    axes[0, 1].set_title("Histograma de residuos")

    qqplot(resid, line="s", ax=axes[1, 0])
    axes[1, 0].set_title("Q-Q plot")

    plot_acf(resid, ax=axes[1, 1], lags=min(24, max(1, len(resid) // 2 - 1)), zero=False)
    axes[1, 1].set_title("ACF de residuos")

    fig.suptitle(f"Diagnostico de residuos - {nombre_serie} - {modelo_nombre}")
    fig.tight_layout()
    fig_path = FIGURAS_DIR / f"residuos_{slug_serie}_{slug_modelo}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "media_residuo": float(resid.mean()),
        "std_residuo": float(resid.std()),
        "ljung_box_pvalue": ljung_pvalue,
        "residuos_csv": csv_path.name,
        "residuos_png": fig_path.name,
    }


def pronosticar_modelo(fitted: Any, n_periods: int) -> tuple[pd.Series, pd.DataFrame | None]:
    forecast_res = fitted.get_forecast(steps=n_periods)
    media = forecast_res.predicted_mean
    try:
        intervalo = forecast_res.conf_int(alpha=0.05)
    except Exception:
        intervalo = None
    return media, intervalo


# ---------------------------------------------------------------------------
# Modelos alternativos obligatorios
# ---------------------------------------------------------------------------


def ajustar_holt_winters(
    serie_train: pd.Series, trend: str = "add", seasonal: str = "add", damped: bool = False
) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = ExponentialSmoothing(
            serie_train,
            trend=trend,
            damped_trend=damped,
            seasonal=seasonal,
            seasonal_periods=PERIODO_ESTACIONAL,
            initialization_method="estimated",
        )
        return modelo.fit(optimized=True)


def ajustar_suavizamiento_exponencial(serie_train: pd.Series, variante: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if variante == "simple":
            return SimpleExpSmoothing(serie_train, initialization_method="estimated").fit()
        if variante == "holt":
            return Holt(serie_train, initialization_method="estimated").fit()
        if variante == "holt_amortiguado":
            return Holt(
                serie_train, damped_trend=True, initialization_method="estimated"
            ).fit(damping_trend=0.9)
        raise ValueError(f"Variante no soportada: {variante}")


def ajustar_prophet(
    serie_train: pd.Series, serie_test: pd.Series, usar_log: bool
) -> tuple[Any, pd.Series, pd.DataFrame | None]:
    if not HAS_PROPHET:
        raise RuntimeError("prophet no esta instalado")

    train_df = pd.DataFrame({"ds": serie_train.index, "y": serie_train.values})
    if usar_log:
        train_df["y"] = np.log1p(train_df["y"].clip(lower=0))
    train_df["pandemia"] = (
        (train_df["ds"] >= PANDEMIA_INICIO) & (train_df["ds"] <= PANDEMIA_FIN)
    ).astype(int)

    modelo = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    modelo.add_regressor("pandemia")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo.fit(train_df)

    future = pd.DataFrame({"ds": serie_test.index})
    future["pandemia"] = (
        (future["ds"] >= PANDEMIA_INICIO) & (future["ds"] <= PANDEMIA_FIN)
    ).astype(int)
    forecast = modelo.predict(future)

    pred = forecast.set_index("ds")["yhat"]
    intervalo = forecast.set_index("ds")[["yhat_lower", "yhat_upper"]]
    if usar_log:
        pred = np.expm1(pred)
        intervalo = np.expm1(intervalo)
    pred.index.name = "fecha"
    return modelo, pred, intervalo


def pronostico_seasonal_naive(
    serie_train: pd.Series, n_periods: int, period: int = PERIODO_ESTACIONAL
) -> pd.Series:
    ultimo_ciclo = serie_train.dropna().iloc[-period:].values
    repeticiones = int(np.ceil(n_periods / period))
    valores = np.tile(ultimo_ciclo, repeticiones)[:n_periods]
    return pd.Series(valores, name="seasonal_naive")


# ---------------------------------------------------------------------------
# Orquestacion por serie
# ---------------------------------------------------------------------------


def _guardar_pronostico(
    nombre_serie: str,
    modelo_nombre: str,
    test: pd.Series,
    pred: np.ndarray,
    intervalo: pd.DataFrame | None,
) -> Path:
    _ensure_dirs()
    slug_serie = L.slugify(nombre_serie)
    slug_modelo = L.slugify(modelo_nombre)
    df = pd.DataFrame({"fecha": test.index, "real": test.values, "prediccion": pred})
    if intervalo is not None:
        cols = list(intervalo.columns)
        df["limite_inferior"] = np.asarray(intervalo[cols[0]])
        df["limite_superior"] = np.asarray(intervalo[cols[1]])
    path = PRONOSTICOS_DIR / f"pronostico_{slug_serie}_{slug_modelo}.csv"
    df.to_csv(path, index=False)
    return path


def evaluar_todos_los_modelos(
    nombre_serie: str, configuracion: pd.Series, output_dir: Path = PARTE2_DIR
) -> dict[str, Any]:
    _ensure_dirs()
    train, test = cargar_serie_train_test(nombre_serie)
    d = int(configuracion["d_sugerido"])
    D = int(configuracion["D_sugerido"])
    transformacion_sugerida = str(configuracion["transformacion"])

    transformacion = construir_transformacion(transformacion_sugerida, train)
    train_transformado = transformacion.forward(train.dropna())

    registros_metricas: list[dict[str, Any]] = []
    registros_candidatos: list[dict[str, Any]] = []
    notas: list[str] = []

    sin_observaciones_prueba = test.dropna().empty
    if sin_observaciones_prueba:
        ultimo_valido = train.dropna().index.max()
        notas.append(
            f"'{nombre_serie}' no tiene observaciones validas en el conjunto de prueba "
            f"(ultimo dato disponible: {ultimo_valido.date() if pd.notna(ultimo_valido) else 'N/D'}). "
            "MAE/RMSE/MAPE no son evaluables; el mejor modelo se selecciona por AIC en entrenamiento."
        )

    # --- Auto ARIMA (apoyo, no sustituto de la busqueda manual) ---
    auto = sugerir_auto_arima(train_transformado, d, D)
    if auto is not None:
        notas.append(
            f"auto_arima sugirio orden={auto['order']} estacional={auto['seasonal_order']} "
            f"AIC={auto['aic']:.2f} BIC={auto['bic']:.2f}."
        )
    else:
        notas.append("auto_arima no disponible o no convergio; se uso solo la rejilla manual.")

    candidatos = crear_candidatos_arima(d, D)
    if auto is not None:
        candidato_auto = {"order": auto["order"], "seasonal_order": auto["seasonal_order"]}
        if candidato_auto not in candidatos:
            candidatos.append(candidato_auto)

    resultados_arima: list[dict[str, Any]] = []
    for cand in candidatos:
        order, seasonal_order = cand["order"], cand["seasonal_order"]
        etiqueta = f"ARIMA{order}x{seasonal_order}"
        registro = {
            "serie": nombre_serie,
            "modelo": etiqueta,
            "orden": str(order),
            "orden_estacional": str(seasonal_order),
            "convergio": False,
            "aic": np.nan,
            "bic": np.nan,
        }
        try:
            fitted = ajustar_sarima(train_transformado, order, seasonal_order)
            registro["convergio"] = bool(fitted.mle_retvals.get("converged", True))
            registro["aic"] = float(fitted.aic)
            registro["bic"] = float(fitted.bic)
            resultados_arima.append({**cand, "etiqueta": etiqueta, "fitted": fitted})
        except Exception as exc:  # pragma: no cover - depende de datos
            registro["nota"] = f"error de ajuste: {exc}"[:200]
        registros_candidatos.append(registro)

    resultados_arima = [r for r in resultados_arima if np.isfinite(r["fitted"].aic)]
    resultados_arima.sort(key=lambda r: r["fitted"].aic)
    finalistas = resultados_arima[:3]

    for cand in finalistas:
        fitted = cand["fitted"]
        etiqueta = cand["etiqueta"]
        diag = diagnosticar_residuos(fitted, nombre_serie, etiqueta)
        pred_transformada, intervalo_transformado = pronosticar_modelo(fitted, len(test))
        pred = transformacion.inverse(np.asarray(pred_transformada))
        pred = np.clip(pred, a_min=0, a_max=None)
        intervalo = None
        if intervalo_transformado is not None:
            cols = list(intervalo_transformado.columns)
            intervalo = pd.DataFrame(
                {
                    cols[0]: np.clip(transformacion.inverse(intervalo_transformado[cols[0]].values), 0, None),
                    cols[1]: np.clip(transformacion.inverse(intervalo_transformado[cols[1]].values), 0, None),
                },
                index=intervalo_transformado.index,
            )
        metricas = calcular_metricas(test.values, pred)
        _guardar_pronostico(nombre_serie, etiqueta, test, pred, intervalo)
        registros_metricas.append(
            {
                "serie": nombre_serie,
                "modelo": etiqueta,
                "orden": str(cand["order"]),
                "orden_estacional": str(cand["seasonal_order"]),
                "MAE": metricas["MAE"],
                "RMSE": metricas["RMSE"],
                "MAPE": metricas["MAPE"],
                "AIC": float(fitted.aic),
                "BIC": float(fitted.bic),
                "ljung_box_pvalue": diag["ljung_box_pvalue"],
                "convergio": True,
                "observaciones": len(test),
                "familia": "ARIMA/SARIMA",
            }
        )

    # --- Prophet ---
    if HAS_PROPHET:
        try:
            usar_log = transformacion_sugerida in ("log", "log1p")
            _, pred_prophet, intervalo_prophet = ajustar_prophet(train, test, usar_log)
            pred_prophet = np.clip(pred_prophet.reindex(test.index).values, 0, None)
            metricas = calcular_metricas(test.values, pred_prophet)
            _guardar_pronostico(nombre_serie, "Prophet", test, pred_prophet, intervalo_prophet)
            registros_metricas.append(
                {
                    "serie": nombre_serie,
                    "modelo": "Prophet",
                    "orden": "",
                    "orden_estacional": "",
                    "MAE": metricas["MAE"],
                    "RMSE": metricas["RMSE"],
                    "MAPE": metricas["MAPE"],
                    "AIC": np.nan,
                    "BIC": np.nan,
                    "ljung_box_pvalue": np.nan,
                    "convergio": True,
                    "observaciones": len(test),
                    "familia": "Prophet",
                }
            )
        except Exception as exc:  # pragma: no cover
            notas.append(f"Prophet fallo para {nombre_serie}: {exc}"[:200])
    else:
        notas.append("prophet no esta instalado; se omitio para esta serie.")

    # --- Holt-Winters ---
    # ExponentialSmoothing/Holt no soportan NaN internos (a diferencia de SARIMAX, que usa
    # filtro de Kalman); se ajustan sobre el tramo valido y contiguo de entrenamiento.
    train_valido = train.dropna()
    valores_positivos = (train_valido > 0).all()
    variantes_hw = [("add", "add", False), ("add", "add", True)]
    if valores_positivos:
        variantes_hw.append(("add", "mul", False))
    for trend, seasonal, damped in variantes_hw:
        etiqueta = f"HoltWinters(trend={trend},seasonal={seasonal},damped={damped})"
        try:
            fitted_hw = ajustar_holt_winters(train_valido, trend=trend, seasonal=seasonal, damped=damped)
            pred_hw = np.clip(np.asarray(fitted_hw.forecast(len(test))), 0, None)
            metricas = calcular_metricas(test.values, pred_hw)
            _guardar_pronostico(nombre_serie, etiqueta, test, pred_hw, None)
            registros_metricas.append(
                {
                    "serie": nombre_serie,
                    "modelo": etiqueta,
                    "orden": "",
                    "orden_estacional": "",
                    "MAE": metricas["MAE"],
                    "RMSE": metricas["RMSE"],
                    "MAPE": metricas["MAPE"],
                    "AIC": float(getattr(fitted_hw, "aic", np.nan)),
                    "BIC": float(getattr(fitted_hw, "bic", np.nan)),
                    "ljung_box_pvalue": np.nan,
                    "convergio": True,
                    "observaciones": len(test),
                    "familia": "Holt-Winters",
                }
            )
        except Exception as exc:  # pragma: no cover
            notas.append(f"Holt-Winters ({etiqueta}) fallo: {exc}"[:200])

    # --- Suavizamiento exponencial (sin estacionalidad) ---
    for variante in ("simple", "holt", "holt_amortiguado"):
        etiqueta = f"SuavizamientoExponencial({variante})"
        try:
            fitted_se = ajustar_suavizamiento_exponencial(train_valido, variante)
            pred_se = np.clip(np.asarray(fitted_se.forecast(len(test))), 0, None)
            metricas = calcular_metricas(test.values, pred_se)
            _guardar_pronostico(nombre_serie, etiqueta, test, pred_se, None)
            registros_metricas.append(
                {
                    "serie": nombre_serie,
                    "modelo": etiqueta,
                    "orden": "",
                    "orden_estacional": "",
                    "MAE": metricas["MAE"],
                    "RMSE": metricas["RMSE"],
                    "MAPE": metricas["MAPE"],
                    "AIC": float(getattr(fitted_se, "aic", np.nan)),
                    "BIC": float(getattr(fitted_se, "bic", np.nan)),
                    "ljung_box_pvalue": np.nan,
                    "convergio": True,
                    "observaciones": len(test),
                    "familia": "SuavizamientoExponencial",
                }
            )
        except Exception as exc:  # pragma: no cover
            notas.append(f"Suavizamiento ({variante}) fallo: {exc}"[:200])

    # --- Seasonal naive ---
    pred_sn = pronostico_seasonal_naive(train_valido, len(test)).values
    metricas = calcular_metricas(test.values, pred_sn)
    _guardar_pronostico(nombre_serie, "SeasonalNaive", test, pred_sn, None)
    registros_metricas.append(
        {
            "serie": nombre_serie,
            "modelo": "SeasonalNaive",
            "orden": "",
            "orden_estacional": "",
            "MAE": metricas["MAE"],
            "RMSE": metricas["RMSE"],
            "MAPE": metricas["MAPE"],
            "AIC": np.nan,
            "BIC": np.nan,
            "ljung_box_pvalue": np.nan,
            "convergio": True,
            "observaciones": len(test),
            "familia": "SeasonalNaive",
        }
    )

    metricas_df = pd.DataFrame(registros_metricas)
    mejor = comparar_modelos(metricas_df)

    fig_path = _graficar_pronostico_comparativo(nombre_serie, train, test, metricas_df)

    return {
        "serie": nombre_serie,
        "metricas": metricas_df,
        "candidatos_arima": pd.DataFrame(registros_candidatos),
        "mejor_modelo": mejor,
        "notas": notas,
        "figura_comparativa": fig_path,
        "transformacion_usada": transformacion.nombre,
    }


def comparar_modelos(metricas_df: pd.DataFrame) -> dict[str, Any]:
    if metricas_df.empty:
        return {}
    validos = metricas_df[metricas_df["convergio"]].copy()
    if validos.empty:
        validos = metricas_df.copy()

    if validos["RMSE"].notna().any():
        autocorr_ok = validos["ljung_box_pvalue"].isna() | (validos["ljung_box_pvalue"] > 0.05)
        filtrados = validos[autocorr_ok & validos["RMSE"].notna()]
        if filtrados.empty:
            filtrados = validos.dropna(subset=["RMSE"])
        filtrados = filtrados.sort_values(["RMSE", "MAE"])
        fila = filtrados.iloc[0]
        justificacion = (
            f"Menor RMSE en prueba ({fila['RMSE']:.2f}) entre modelos convergentes"
            + (" con residuos sin autocorrelacion significativa" if autocorr_ok.any() else "")
            + f"; MAE={fila['MAE']:.2f}."
        )
        return {
            "serie": fila["serie"],
            "modelo_seleccionado": fila["modelo"],
            "configuracion": f"orden={fila.get('orden', '')} estacional={fila.get('orden_estacional', '')}",
            "MAE": fila["MAE"],
            "RMSE": fila["RMSE"],
            "AIC": fila.get("AIC", np.nan),
            "BIC": fila.get("BIC", np.nan),
            "justificacion": justificacion,
        }

    # Ninguna metrica de prueba es evaluable (serie sin observaciones reales en el
    # horizonte de prueba, p. ej. Via Maritima despues de 2016): elegir por AIC en
    # entrenamiento entre los modelos ARIMA/SARIMA ajustados, si existen.
    con_aic = validos.dropna(subset=["AIC"])
    base = con_aic if not con_aic.empty else validos
    fila = base.sort_values("AIC").iloc[0] if not con_aic.empty else base.iloc[0]
    justificacion = (
        "Sin observaciones validas en el conjunto de prueba para esta serie; "
        + (
            f"se selecciono por menor AIC en entrenamiento ({fila['AIC']:.2f})."
            if not con_aic.empty
            else "se reporta el primer modelo ajustado a falta de un criterio evaluable."
        )
    )
    return {
        "serie": fila["serie"],
        "modelo_seleccionado": fila["modelo"],
        "configuracion": f"orden={fila.get('orden', '')} estacional={fila.get('orden_estacional', '')}",
        "MAE": fila["MAE"],
        "RMSE": fila["RMSE"],
        "AIC": fila.get("AIC", np.nan),
        "BIC": fila.get("BIC", np.nan),
        "justificacion": justificacion,
    }


def _graficar_pronostico_comparativo(
    nombre_serie: str, train: pd.Series, test: pd.Series, metricas_df: pd.DataFrame
) -> Path:
    _ensure_dirs()
    slug = L.slugify(nombre_serie)
    top_modelos = metricas_df.sort_values("RMSE").head(4)

    fig, ax = plt.subplots(figsize=(12, 5))
    train.tail(36).plot(ax=ax, label="Train (ultimos 36m)", color="#555555", linewidth=1.3)
    test.plot(ax=ax, label="Real (prueba)", color="black", linewidth=2)

    colores = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for color, (_, fila) in zip(colores, top_modelos.iterrows()):
        modelo_slug = L.slugify(fila["modelo"])
        pron_path = PRONOSTICOS_DIR / f"pronostico_{slug}_{modelo_slug}.csv"
        if not pron_path.exists():
            continue
        pron_df = pd.read_csv(pron_path, parse_dates=["fecha"]).set_index("fecha")
        ax.plot(
            pron_df.index,
            pron_df["prediccion"],
            label=f"{fila['modelo']} (RMSE={fila['RMSE']:.0f})",
            color=color,
            linewidth=1.5,
            linestyle="--",
        )

    ax.set_title(f"Comparacion de pronosticos - {nombre_serie}")
    ax.set_ylabel("Viajeros")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig_path = FIGURAS_DIR / f"comparacion_modelos_{slug}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig_path


# ---------------------------------------------------------------------------
# Ejecucion completa de la Parte 2
# ---------------------------------------------------------------------------


def _dataframe_a_latex(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    cuerpo = df.to_latex(
        index=False,
        float_format="%.2f",
        na_rep="--",
        escape=True,
    )
    contenido = (
        "\\begin{table}[H]\n\\centering\n\\scriptsize\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"{cuerpo}\n\\end{{table}}\n"
    )
    path.write_text(contenido, encoding="utf-8")


def _build_manifest_parte2(
    mejores: pd.DataFrame, metricas: pd.DataFrame, notas_por_serie: dict[str, list[str]]
) -> Path:
    lines = ["# Manifest Parte 2", "", "## Mejor modelo por serie"]
    for _, fila in mejores.iterrows():
        lines.append(
            f"- {fila['serie']}: {fila['modelo_seleccionado']} "
            f"(RMSE={fila['RMSE']:.2f}, MAE={fila['MAE']:.2f}) — {fila['justificacion']}"
        )
    lines.extend(["", "## Notas y problemas de convergencia"])
    for serie, notas in notas_por_serie.items():
        if not notas:
            continue
        lines.append(f"- {serie}:")
        lines.extend([f"  - {nota}" for nota in notas])
    lines.extend(["", "## Archivos", "- metricas_modelos.csv", "- mejores_modelos.csv", "- candidatos_arima.csv"])
    path = PARTE2_DIR / "manifest.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def ejecutar_parte2() -> dict[str, Any]:
    _ensure_dirs()
    config = cargar_configuracion_series()

    todas_metricas = []
    todos_candidatos = []
    todos_mejores = []
    notas_por_serie: dict[str, list[str]] = {}

    for _, fila_config in config.iterrows():
        nombre_serie = fila_config["serie"]
        resultado = evaluar_todos_los_modelos(nombre_serie, fila_config)
        todas_metricas.append(resultado["metricas"])
        todos_candidatos.append(resultado["candidatos_arima"])
        if resultado["mejor_modelo"]:
            todos_mejores.append(resultado["mejor_modelo"])
        notas_por_serie[nombre_serie] = resultado["notas"]

    metricas_df = pd.concat(todas_metricas, ignore_index=True)
    candidatos_df = pd.concat(todos_candidatos, ignore_index=True)
    mejores_df = pd.DataFrame(todos_mejores)

    metricas_df.to_csv(PARTE2_DIR / "metricas_modelos.csv", index=False)
    candidatos_df.to_csv(PARTE2_DIR / "candidatos_arima.csv", index=False)
    mejores_df.to_csv(PARTE2_DIR / "mejores_modelos.csv", index=False)

    total_mask = metricas_df["serie"].str.contains("Total", case=False)
    frontera_mask = metricas_df["serie"].str.contains("Frontera", case=False)
    via_mask = metricas_df["serie"].str.contains("Vía|Via", case=False, regex=True)

    cols_tabla = ["serie", "modelo", "MAE", "RMSE", "MAPE", "AIC", "BIC"]
    _dataframe_a_latex(
        metricas_df.loc[total_mask, cols_tabla],
        TABLAS_LATEX_DIR / "tabla_metricas_total.tex",
        "Metricas de modelos para la serie total mensual.",
        "tab:metricas-total",
    )
    _dataframe_a_latex(
        metricas_df.loc[frontera_mask, cols_tabla],
        TABLAS_LATEX_DIR / "tabla_metricas_fronteras.tex",
        "Metricas de modelos para las series por frontera.",
        "tab:metricas-fronteras",
    )
    _dataframe_a_latex(
        metricas_df.loc[via_mask, cols_tabla],
        TABLAS_LATEX_DIR / "tabla_metricas_vias.tex",
        "Metricas de modelos para las series por via de ingreso.",
        "tab:metricas-vias",
    )

    manifest = _build_manifest_parte2(mejores_df, metricas_df, notas_por_serie)

    return {
        "metricas": metricas_df,
        "candidatos": candidatos_df,
        "mejores": mejores_df,
        "manifest": str(manifest),
        "notas_por_serie": notas_por_serie,
    }


if __name__ == "__main__":
    resumen = ejecutar_parte2()
    print(resumen["mejores"])
