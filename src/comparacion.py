from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from src import lab as L

ROOT = L.ROOT
PARTE1_DIR = ROOT / "outputs" / "parte1"
PARTE2_DIR = ROOT / "outputs" / "parte2"
PARTE3_DIR = ROOT / "outputs" / "parte3"
FIGURAS_DIR = PARTE3_DIR / "figuras"
TABLAS_LATEX_DIR = PARTE3_DIR / "tablas_latex"
REPORT_DIR = ROOT / "report"

PERIODO_ESTACIONAL = 12
PANDEMIA_INICIO = pd.Timestamp("2020-03-01")
PANDEMIA_FIN = pd.Timestamp("2021-12-01")

FRONTERAS_LABEL = "fronteras"
VIAS_LABEL = "vias"


def _ensure_dirs() -> None:
    for path in (
        PARTE3_DIR,
        FIGURAS_DIR,
        TABLAS_LATEX_DIR,
        REPORT_DIR / "figures",
        REPORT_DIR / "tables",
        REPORT_DIR / "values",
    ):
        path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Carga de series completas (train + test)
# ---------------------------------------------------------------------------


def cargar_serie_completa(nombre_serie: str) -> pd.Series:
    slug = L.slugify(nombre_serie)
    train = pd.read_csv(PARTE1_DIR / "series" / f"{slug}_train.csv", parse_dates=["fecha"])
    test = pd.read_csv(PARTE1_DIR / "series" / f"{slug}_test.csv", parse_dates=["fecha"])
    completa = pd.concat([train, test]).set_index("fecha")["valor"].asfreq(L.FRECUENCIA)
    completa.name = nombre_serie
    return completa


# ---------------------------------------------------------------------------
# Medidas comparativas
# ---------------------------------------------------------------------------


def _recortar_a_rango_valido(serie: pd.Series, max_hueco_interno: int = 3) -> pd.Series:
    """Recorta los NaN iniciales/finales (series que dejaron de reportarse, p. ej.
    Via Maritima despues de 2016) e interpola solo huecos internos cortos, para no
    fabricar tendencia/estacionalidad sobre tramos sin ningun dato real."""
    valida = serie.dropna()
    if valida.empty:
        return valida
    recortada = serie.loc[valida.index.min() : valida.index.max()]
    return recortada.interpolate(limit=max_hueco_interno, limit_direction="both")


def fuerza_estacionalidad(serie: pd.Series, periodo: int = PERIODO_ESTACIONAL) -> float:
    limpia = _recortar_a_rango_valido(serie).dropna()
    if len(limpia) < periodo * 2:
        return np.nan
    decomp = seasonal_decompose(limpia, model="additive", period=periodo)
    var_resid = np.nanvar(decomp.resid)
    var_seasonal_resid = np.nanvar(decomp.seasonal + decomp.resid)
    if var_seasonal_resid == 0:
        return np.nan
    return float(max(0.0, 1 - var_resid / var_seasonal_resid))


def pendiente_tendencia(serie: pd.Series, periodo: int = PERIODO_ESTACIONAL) -> float:
    limpia = _recortar_a_rango_valido(serie).dropna()
    if len(limpia) < periodo * 2:
        return np.nan
    decomp = seasonal_decompose(limpia, model="additive", period=periodo)
    trend = decomp.trend.dropna()
    if len(trend) < 2:
        return np.nan
    x = np.arange(len(trend))
    pendiente, _ = np.polyfit(x, trend.values, 1)
    return float(pendiente)


def crecimiento_ventanas(serie: pd.Series) -> dict[str, float]:
    anual = serie.groupby(serie.index.year).mean()
    anios = sorted(anual.index)
    primer_anio_completo = next(
        (a for a in anios if (serie.index.year == a).sum() == PERIODO_ESTACIONAL), anios[0]
    )
    promedio_2019 = float(anual.get(2019, np.nan))
    promedio_primer_anio = float(anual.get(primer_anio_completo, np.nan))
    crecimiento_pre = (
        (promedio_2019 / promedio_primer_anio - 1) * 100
        if promedio_primer_anio not in (0, np.nan) and not np.isnan(promedio_primer_anio)
        else np.nan
    )

    anios_completos_recientes = [
        a for a in anios if a > 2021 and (serie.index.year == a).sum() == PERIODO_ESTACIONAL
    ]
    ultimo_anio_completo = anios_completos_recientes[-1] if anios_completos_recientes else None
    minimo_2020 = float(serie[serie.index.year == 2020].min()) if 2020 in anios else np.nan
    if ultimo_anio_completo is not None and minimo_2020 not in (0, np.nan) and not np.isnan(minimo_2020):
        crecimiento_post = (float(anual[ultimo_anio_completo]) / minimo_2020 - 1) * 100
    else:
        crecimiento_post = np.nan

    return {
        "crecimiento_pre_pandemia": crecimiento_pre,
        "crecimiento_post_pandemia": crecimiento_post,
    }


def volatilidad(serie: pd.Series) -> dict[str, float]:
    limpia = serie.dropna()
    media = limpia.mean()
    std = limpia.std()
    diffs = limpia.diff().dropna()
    pct = limpia.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "desviacion_estandar": float(std),
        "coeficiente_variacion": float(std / media) if media else np.nan,
        "std_diferencias": float(diffs.std()),
        "std_cambio_porcentual": float(pct.std()) if not pct.empty else np.nan,
    }


def impacto_pandemia(serie: pd.Series) -> dict[str, Any]:
    datos_2019 = serie[serie.index.year == 2019].dropna()
    datos_2020 = serie[serie.index.year == 2020].dropna()
    if datos_2019.empty or datos_2020.empty:
        return {
            "promedio_2019": np.nan,
            "minimo_2020": np.nan,
            "caida_porcentual": np.nan,
            "meses_recuperacion": "sin_datos_suficientes",
        }
    promedio_2019 = float(datos_2019.mean())
    minimo_2020 = float(datos_2020.min())
    fecha_minimo = datos_2020.idxmin()
    caida_pct = (minimo_2020 / promedio_2019 - 1) * 100 if promedio_2019 else np.nan

    posterior = serie[serie.index > fecha_minimo]
    recuperacion = posterior[posterior >= promedio_2019]
    if recuperacion.empty:
        meses_recuperacion: Any = "no_recuperada_hasta_2026_06"
    else:
        fecha_recuperacion = recuperacion.index[0]
        meses_recuperacion = int(
            (fecha_recuperacion.year - fecha_minimo.year) * 12
            + (fecha_recuperacion.month - fecha_minimo.month)
        )

    return {
        "promedio_2019": promedio_2019,
        "minimo_2020": minimo_2020,
        "caida_porcentual": caida_pct,
        "meses_recuperacion": meses_recuperacion,
    }


def construir_fila_comparativa(nombre_serie: str, mejores_modelos: pd.DataFrame) -> dict[str, Any]:
    serie = cargar_serie_completa(nombre_serie)
    fila: dict[str, Any] = {"serie": nombre_serie}
    fila["fuerza_estacionalidad"] = fuerza_estacionalidad(serie)
    fila["pendiente_tendencia"] = pendiente_tendencia(serie)
    fila.update(crecimiento_ventanas(serie))
    fila.update(volatilidad(serie))
    fila.update(impacto_pandemia(serie))

    match = mejores_modelos[mejores_modelos["serie"] == nombre_serie]
    if not match.empty:
        fila["mejor_modelo"] = match.iloc[0]["modelo_seleccionado"]
        fila["MAE"] = match.iloc[0]["MAE"]
        fila["RMSE"] = match.iloc[0]["RMSE"]
    else:
        fila["mejor_modelo"] = np.nan
        fila["MAE"] = np.nan
        fila["RMSE"] = np.nan
    return fila


# ---------------------------------------------------------------------------
# Figuras comparativas
# ---------------------------------------------------------------------------


def _figura_series_superpuestas(nombres: list[str], titulo: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    for nombre in nombres:
        serie = cargar_serie_completa(nombre)
        ax.plot(serie.index, serie.values, label=nombre, linewidth=1.4)
    ax.axvspan(PANDEMIA_INICIO, PANDEMIA_FIN, color="#ffcc99", alpha=0.3, label="Pandemia")
    ax.set_title(titulo)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Viajeros")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = FIGURAS_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _figura_estacionalidad_mensual(nombres: list[str], titulo: str, filename: str) -> Path:
    fig, axes = plt.subplots(1, len(nombres), figsize=(5 * len(nombres), 4.2), sharey=False)
    if len(nombres) == 1:
        axes = [axes]
    for ax, nombre in zip(axes, nombres):
        serie = cargar_serie_completa(nombre)
        df = serie.to_frame("valor")
        df["mes"] = df.index.month
        df.boxplot(column="valor", by="mes", ax=ax)
        ax.set_title(nombre, fontsize=9)
        ax.set_xlabel("Mes")
        ax.set_ylabel("Viajeros")
    fig.suptitle(titulo)
    plt.suptitle(titulo)
    fig.tight_layout()
    path = FIGURAS_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _figura_caida_recuperacion(nombres: list[str], titulo: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    for nombre in nombres:
        serie = cargar_serie_completa(nombre)
        if serie[serie.index.year == 2019].dropna().empty:
            continue
        base = serie[serie.index.year == 2019].mean()
        indice = (serie / base) * 100
        ventana = indice[(indice.index >= "2019-01-01") & (indice.index <= "2023-12-01")]
        ax.plot(ventana.index, ventana.values, label=nombre, linewidth=1.4)
    ax.axhline(100, color="black", linestyle=":", linewidth=1, label="Nivel 2019 = 100")
    ax.axvspan(PANDEMIA_INICIO, PANDEMIA_FIN, color="#ffcc99", alpha=0.3, label="Pandemia")
    ax.set_title(titulo)
    ax.set_ylabel("Indice (2019=100)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = FIGURAS_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _figura_rmse_por_serie(mejores_modelos: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5))
    datos = mejores_modelos.sort_values("RMSE")
    ax.barh(datos["serie"], datos["RMSE"], color="#4c78a8")
    ax.set_xlabel("RMSE (conjunto de prueba)")
    ax.set_title("RMSE del mejor modelo por serie")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = FIGURAS_DIR / "rmse_por_serie.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _figura_mejor_pronostico_por_serie(mejores_modelos: pd.DataFrame) -> Path:
    n = len(mejores_modelos)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (_, fila) in zip(axes, mejores_modelos.iterrows()):
        nombre_serie = fila["serie"]
        modelo_slug = L.slugify(fila["modelo_seleccionado"])
        slug_serie = L.slugify(nombre_serie)
        pron_path = PARTE2_DIR / "pronosticos" / f"pronostico_{slug_serie}_{modelo_slug}.csv"
        if not pron_path.exists():
            ax.set_visible(False)
            continue
        pron_df = pd.read_csv(pron_path, parse_dates=["fecha"]).set_index("fecha")
        ax.plot(pron_df.index, pron_df["real"], label="Real", color="black", linewidth=1.6)
        ax.plot(
            pron_df.index,
            pron_df["prediccion"],
            label=fila["modelo_seleccionado"],
            color="#d62728",
            linestyle="--",
        )
        if {"limite_inferior", "limite_superior"}.issubset(pron_df.columns):
            ax.fill_between(
                pron_df.index,
                pron_df["limite_inferior"],
                pron_df["limite_superior"],
                color="#d62728",
                alpha=0.15,
            )
        ax.set_title(nombre_serie, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    for ax in axes[len(mejores_modelos):]:
        ax.set_visible(False)
    fig.suptitle("Mejor pronostico por serie (conjunto de prueba)")
    fig.tight_layout()
    path = FIGURAS_DIR / "mejor_pronostico_por_serie.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _figura_real_vs_predicho(mejores_modelos: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 7))
    colores = plt.cm.tab10(np.linspace(0, 1, len(mejores_modelos)))
    for color, (_, fila) in zip(colores, mejores_modelos.iterrows()):
        modelo_slug = L.slugify(fila["modelo_seleccionado"])
        slug_serie = L.slugify(fila["serie"])
        pron_path = PARTE2_DIR / "pronosticos" / f"pronostico_{slug_serie}_{modelo_slug}.csv"
        if not pron_path.exists():
            continue
        pron_df = pd.read_csv(pron_path)
        ax.scatter(pron_df["real"], pron_df["prediccion"], s=12, alpha=0.6, color=color, label=fila["serie"])
    lims = [0, ax.get_xlim()[1]]
    ax.plot(lims, lims, color="black", linestyle=":", linewidth=1, label="Ajuste perfecto")
    ax.set_xlabel("Real")
    ax.set_ylabel("Predicho")
    ax.set_title("Real vs. predicho — mejor modelo por serie")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = FIGURAS_DIR / "real_vs_predicho.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Hallazgos INGUAT
# ---------------------------------------------------------------------------


def construir_hallazgos(comparacion: pd.DataFrame, mejores_modelos: pd.DataFrame) -> pd.DataFrame:
    filas: list[dict[str, str]] = []

    fronteras = comparacion[comparacion["serie"].str.contains("Frontera")]
    if not fronteras.empty:
        peor_recuperacion = fronteras.loc[
            fronteras["meses_recuperacion"].apply(
                lambda v: v if isinstance(v, (int, float)) else np.inf
            ).idxmax()
        ]
        filas.append(
            {
                "hallazgo": "Recuperacion post-pandemia desigual entre fronteras",
                "evidencia": (
                    f"{peor_recuperacion['serie']} tardo "
                    f"{peor_recuperacion['meses_recuperacion']} meses en volver al nivel de 2019, "
                    "el mas lento entre las tres fronteras principales."
                ),
                "implicacion": "La capacidad de recuperacion no es homogenea por punto de ingreso.",
                "recomendacion": "Priorizar apoyo operativo y promocion diferenciada en esa frontera.",
                "series_relacionadas": peor_recuperacion["serie"],
            }
        )
        mayor_estacionalidad_f = fronteras.loc[fronteras["fuerza_estacionalidad"].idxmax()]
        filas.append(
            {
                "hallazgo": "Concentracion estacional marcada en una frontera",
                "evidencia": (
                    f"{mayor_estacionalidad_f['serie']} presenta la mayor fuerza de estacionalidad "
                    f"({mayor_estacionalidad_f['fuerza_estacionalidad']:.2f}) entre las fronteras analizadas."
                ),
                "implicacion": "La demanda en esa frontera se concentra en meses recurrentes especificos.",
                "recomendacion": "Planificar capacidad migratoria y de infraestructura para los meses pico.",
                "series_relacionadas": mayor_estacionalidad_f["serie"],
            }
        )

    vias = comparacion[comparacion["serie"].str.contains("Vía|Via", regex=True)]
    if not vias.empty:
        mas_volatil = vias.loc[vias["coeficiente_variacion"].idxmax()]
        filas.append(
            {
                "hallazgo": "Una via de ingreso concentra mayor incertidumbre",
                "evidencia": (
                    f"{mas_volatil['serie']} tiene el mayor coeficiente de variacion "
                    f"({mas_volatil['coeficiente_variacion']:.2f}) entre las vias analizadas."
                ),
                "implicacion": "Esa via es mas dificil de pronosticar y mas sensible a choques externos.",
                "recomendacion": "Usar intervalos de pronostico amplios y monitoreo mas frecuente para esa via.",
                "series_relacionadas": mas_volatil["serie"],
            }
        )
        mayor_caida = vias.loc[vias["caida_porcentual"].idxmin()]
        filas.append(
            {
                "hallazgo": "Concentracion del impacto de la pandemia por via",
                "evidencia": (
                    f"{mayor_caida['serie']} sufrio la mayor caida porcentual "
                    f"({mayor_caida['caida_porcentual']:.1f}%) frente al promedio de 2019."
                ),
                "implicacion": "El riesgo ante choques futuros no se distribuye igual entre vias.",
                "recomendacion": "Diversificar la dependencia de una sola via de ingreso en la planificacion.",
                "series_relacionadas": mayor_caida["serie"],
            }
        )

    if not mejores_modelos.empty:
        peor_rmse = mejores_modelos.loc[mejores_modelos["RMSE"].idxmax()]
        filas.append(
            {
                "hallazgo": "El desempeno de los modelos no es uniforme entre series",
                "evidencia": (
                    f"{peor_rmse['serie']} presenta el mayor RMSE en prueba "
                    f"({peor_rmse['RMSE']:.0f}) entre las series modeladas."
                ),
                "implicacion": "Los cambios estructurales recientes son mas dificiles de capturar en esa serie.",
                "recomendacion": "Revisar esa serie con mayor frecuencia y complementar con juicio experto.",
                "series_relacionadas": peor_rmse["serie"],
            }
        )

    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Exportacion LaTeX y control de calidad
# ---------------------------------------------------------------------------


def _dataframe_a_latex(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    cuerpo = df.round(2).to_latex(index=False, float_format="%.2f", na_rep="--", escape=True)
    contenido = (
        "\\begin{table}[H]\n\\centering\n\\scriptsize\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"{cuerpo}\n\\end{{table}}\n"
    )
    path.write_text(contenido, encoding="utf-8")


def _exportar_report(figuras: list[Path], tablas: list[Path]) -> None:
    for fig in figuras:
        if fig.exists():
            shutil.copy2(fig, REPORT_DIR / "figures" / fig.name)
    for tabla in tablas:
        if tabla.exists():
            shutil.copy2(tabla, REPORT_DIR / "tables" / tabla.name)


def _resultados_clave_tex(
    config: pd.DataFrame, mejores_modelos: pd.DataFrame, comparacion: pd.DataFrame
) -> Path:
    total = mejores_modelos[mejores_modelos["serie"].str.contains("Total")]
    fecha_corte = "2021-03"
    fila_config = config[config["serie"].str.contains("Total")]
    if not fila_config.empty and "fecha_final_train" in fila_config.columns:
        fecha_corte = str(fila_config.iloc[0]["fecha_final_train"])[:7]

    mejor_modelo_total = total.iloc[0]["modelo_seleccionado"] if not total.empty else "N/D"
    rmse_total = f"{total.iloc[0]['RMSE']:.2f}" if not total.empty else "N/D"
    mae_total = f"{total.iloc[0]['MAE']:.2f}" if not total.empty else "N/D"

    lines = [
        f"\\newcommand{{\\FechaCorteTrain}}{{{fecha_corte}}}",
        "\\newcommand{\\NumeroSeries}{7}",
        f"\\newcommand{{\\MejorModeloTotal}}{{{mejor_modelo_total}}}",
        f"\\newcommand{{\\RMSETotal}}{{{rmse_total}}}",
        f"\\newcommand{{\\MAETotal}}{{{mae_total}}}",
    ]
    path = REPORT_DIR / "values" / "resultados_clave.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _control_calidad(
    config: pd.DataFrame, metricas: pd.DataFrame, mejores_modelos: pd.DataFrame
) -> Path:
    checks: list[str] = []

    series_esperadas = set(config["serie"])
    checks.append(f"[{'OK' if len(series_esperadas) == 7 else 'FALLA'}] Existen 7 series: {len(series_esperadas)} encontradas.")

    series_sin_test: set[str] = set()
    for nombre in series_esperadas:
        try:
            slug = L.slugify(nombre)
            train_df = pd.read_csv(PARTE1_DIR / "series" / f"{slug}_train.csv", parse_dates=["fecha"])
            test_df = pd.read_csv(PARTE1_DIR / "series" / f"{slug}_test.csv", parse_dates=["fecha"])
            dup_train = train_df["fecha"].duplicated().sum()
            dup_test = test_df["fecha"].duplicated().sum()
            orden_ok = train_df["fecha"].max() < test_df["fecha"].min()
            checks.append(
                f"[{'OK' if dup_train == 0 and dup_test == 0 else 'FALLA'}] {nombre}: sin fechas duplicadas."
            )
            checks.append(f"[{'OK' if orden_ok else 'FALLA'}] {nombre}: train termina antes que test.")
            if test_df["valor"].dropna().empty:
                series_sin_test.add(nombre)
        except Exception as exc:
            checks.append(f"[FALLA] {nombre}: error validando series ({exc}).")

    if series_sin_test:
        checks.append(
            f"[OK] Series sin observaciones en prueba (excluidas del chequeo de metricas vacias, "
            f"metricas no evaluables por diseno): {sorted(series_sin_test)}."
        )

    convergentes = metricas[metricas["convergio"] & ~metricas["serie"].isin(series_sin_test)]
    metricas_vacias = convergentes[["MAE", "RMSE"]].isna().any(axis=1).sum()
    checks.append(
        f"[{'OK' if metricas_vacias == 0 else 'FALLA'}] Modelos convergentes (con datos de prueba) sin metricas vacias: {metricas_vacias} casos."
    )

    checks.append(
        f"[{'OK' if len(mejores_modelos) == len(series_esperadas) else 'FALLA'}] "
        f"Mejor modelo seleccionado para {len(mejores_modelos)}/{len(series_esperadas)} series."
    )

    main_tex_path = REPORT_DIR / "main.tex"
    if main_tex_path.exists():
        contenido = main_tex_path.read_text(encoding="utf-8")
        marcadores = [m for m in ("TODO", "TBD", "completar", "COMPLETAR") if m in contenido]
        checks.append(f"[{'OK' if not marcadores else 'FALLA'}] Sin marcadores pendientes en main.tex: {marcadores or 'ninguno'}.")
        rutas_absolutas = "C:\\Users" in contenido or "/home/" in contenido
        checks.append(f"[{'OK' if not rutas_absolutas else 'FALLA'}] Sin rutas absolutas en main.tex.")

    path = PARTE3_DIR / "control_calidad.txt"
    path.write_text("\n".join(checks) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Ejecucion completa de la Parte 3
# ---------------------------------------------------------------------------


def ejecutar_parte3() -> dict[str, Any]:
    _ensure_dirs()

    config = pd.read_csv(PARTE1_DIR / "tablas" / "resultados_estacionariedad.csv")
    metricas = pd.read_csv(PARTE2_DIR / "metricas_modelos.csv")
    mejores_modelos = pd.read_csv(PARTE2_DIR / "mejores_modelos.csv")

    comparacion = pd.DataFrame(
        [construir_fila_comparativa(nombre, mejores_modelos) for nombre in config["serie"]]
    )
    comparacion.to_csv(PARTE3_DIR / "comparacion_modelos_global.csv", index=False)

    fronteras_nombres = [n for n in config["serie"] if "Frontera" in n]
    vias_nombres = [n for n in config["serie"] if "Vía" in n or "Via" in n]

    comparacion_fronteras = comparacion[comparacion["serie"].isin(fronteras_nombres)]
    comparacion_vias = comparacion[comparacion["serie"].isin(vias_nombres)]
    comparacion_fronteras.to_csv(PARTE3_DIR / "comparacion_fronteras.csv", index=False)
    comparacion_vias.to_csv(PARTE3_DIR / "comparacion_vias.csv", index=False)

    cols_tabla = [
        "serie",
        "fuerza_estacionalidad",
        "pendiente_tendencia",
        "coeficiente_variacion",
        "caida_porcentual",
        "meses_recuperacion",
        "mejor_modelo",
        "MAE",
        "RMSE",
    ]
    _dataframe_a_latex(
        comparacion_fronteras[cols_tabla],
        TABLAS_LATEX_DIR / "comparacion_fronteras.tex",
        "Comparacion estadistica entre las tres fronteras principales.",
        "tab:comparacion-fronteras",
    )
    _dataframe_a_latex(
        comparacion_vias[cols_tabla],
        TABLAS_LATEX_DIR / "comparacion_vias.tex",
        "Comparacion estadistica entre las tres vias de ingreso.",
        "tab:comparacion-vias",
    )
    _dataframe_a_latex(
        mejores_modelos,
        TABLAS_LATEX_DIR / "mejores_modelos.tex",
        "Mejor modelo seleccionado para cada serie.",
        "tab:mejores-modelos",
    )

    figuras = [
        _figura_series_superpuestas(fronteras_nombres, "Series mensuales — tres fronteras principales", "fronteras_superpuestas.png"),
        _figura_series_superpuestas(vias_nombres, "Series mensuales — tres vias de ingreso", "vias_superpuestas.png"),
        _figura_estacionalidad_mensual(fronteras_nombres, "Estacionalidad mensual por frontera", "estacionalidad_fronteras.png"),
        _figura_estacionalidad_mensual(vias_nombres, "Estacionalidad mensual por via", "estacionalidad_vias.png"),
        _figura_caida_recuperacion(
            fronteras_nombres + vias_nombres, "Caida y recuperacion durante la pandemia (2019=100)", "caida_recuperacion_pandemia.png"
        ),
        _figura_rmse_por_serie(mejores_modelos),
        _figura_mejor_pronostico_por_serie(mejores_modelos),
        _figura_real_vs_predicho(mejores_modelos),
    ]

    hallazgos = construir_hallazgos(comparacion, mejores_modelos)
    hallazgos.to_csv(PARTE3_DIR / "hallazgos_inguat.csv", index=False)

    figuras_parte2 = sorted((PARTE2_DIR / "figuras").glob("comparacion_modelos_*.png"))
    residuos_destacados = [
        PARTE2_DIR / "figuras" / "residuos_total_mensual_arima_2_1_2_x_1_0_1_12.png",
        PARTE2_DIR / "figuras" / "residuos_frontera_01_la_aurora_arima_1_1_1_x_0_0_1_12.png",
        PARTE2_DIR / "figuras" / "residuos_via_maritima_arima_0_1_1_x_1_0_1_12.png",
    ]
    figuras += figuras_parte2 + [p for p in residuos_destacados if p.exists()]

    tablas_para_exportar = [
        TABLAS_LATEX_DIR / "comparacion_fronteras.tex",
        TABLAS_LATEX_DIR / "comparacion_vias.tex",
        TABLAS_LATEX_DIR / "mejores_modelos.tex",
        PARTE2_DIR / "tablas_latex" / "tabla_metricas_total.tex",
        PARTE2_DIR / "tablas_latex" / "tabla_metricas_fronteras.tex",
        PARTE2_DIR / "tablas_latex" / "tabla_metricas_vias.tex",
    ]
    _exportar_report(figuras, tablas_para_exportar)
    resultados_clave = _resultados_clave_tex(config, mejores_modelos, comparacion)
    control = _control_calidad(config, metricas, mejores_modelos)

    return {
        "comparacion": comparacion,
        "comparacion_fronteras": comparacion_fronteras,
        "comparacion_vias": comparacion_vias,
        "hallazgos": hallazgos,
        "figuras": figuras,
        "resultados_clave": str(resultados_clave),
        "control_calidad": str(control),
    }


if __name__ == "__main__":
    resumen = ejecutar_parte3()
    print(resumen["hallazgos"])
