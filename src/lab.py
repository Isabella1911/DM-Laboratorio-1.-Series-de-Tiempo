from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import boxcox_normmax
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Base_Migracion_2009-2026jun.xlsx"
SHEET_NAME = "Datos"
TIPOS_VALIDOS = ["Turista", "Excursionista"]
TRAIN_RATIO = 0.70
FRECUENCIA = "MS"
PERIODO_ESTACIONAL = 12
OUTPUT_DIR = ROOT / "outputs" / "parte1"
TABLAS_DIR = OUTPUT_DIR / "tablas"
SERIES_DIR = OUTPUT_DIR / "series"
FIGURAS_DIR = OUTPUT_DIR / "figuras"

EDA_WARNING = (
    "Advertencia: despues de 2023 la variable 'Pais' puede reflejar agrupaciones "
    "de mercado y no siempre paises individuales comparables."
)


@dataclass
class SeriesAnalysisResult:
    serie: str
    archivo_train: str
    archivo_test: str
    inicio: str
    fin: str
    frecuencia: str
    n_observaciones: int
    fecha_final_train: str
    fecha_inicial_test: str
    transformacion: str
    justificacion_transformacion: str
    adf_nivel: float | None
    pvalue_nivel: float | None
    adf_diff1: float | None
    pvalue_diff1: float | None
    d_sugerido: int
    D_sugerido: int
    comentario: str
    interpretacion: str


def _ensure_dirs() -> None:
    for path in (OUTPUT_DIR, TABLAS_DIR, SERIES_DIR, FIGURAS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    value = text.lower().strip()
    value = (
        value.replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ñ", "n")
    )
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def cargar_datos(path: Path = DATA_PATH, sheet_name: str = SHEET_NAME) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name)


def preparar_fecha(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fecha"] = pd.to_datetime(
        dict(year=out["Año"].astype(int), month=out["Mes cod"].astype(int), day=1)
    )
    return out.sort_values("fecha").reset_index(drop=True)


def filtrar_tipos_consistentes(
    df: pd.DataFrame, tipos_validos: list[str] = TIPOS_VALIDOS
) -> pd.DataFrame:
    return df[df["Tipo de Viajero"].isin(tipos_validos)].copy()


def resumir_calidad_datos(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    resumen = pd.DataFrame(
        {
            "metric": [
                "n_filas",
                "n_columnas",
                "fecha_min",
                "fecha_max",
                "anios_min",
                "anios_max",
                "meses_distintos",
                "viajero_min",
                "viajero_max",
                "viajero_negativos",
            ],
            "value": [
                len(df),
                len(df.columns),
                df["fecha"].min().date().isoformat(),
                df["fecha"].max().date().isoformat(),
                int(df["Año"].min()),
                int(df["Año"].max()),
                int(df["Mes cod"].nunique()),
                float(df["Viajero"].min()),
                float(df["Viajero"].max()),
                int((df["Viajero"] < 0).sum()),
            ],
        }
    )

    faltantes = (
        df.isna()
        .sum()
        .rename("faltantes")
        .to_frame()
        .assign(
            porcentaje=lambda x: (x["faltantes"] / len(df) * 100).round(4),
            dtype=[str(df[c].dtype) for c in df.columns],
        )
        .reset_index(names="columna")
    )
    return resumen, faltantes


def detectar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    dims = [
        "fecha",
        "Vía",
        "Frontera",
        "País",
        "Región",
        "Región dos",
        "Tipo de Viajero",
    ]
    completos = int(df.duplicated().sum())
    por_dims = (
        df.groupby(dims, dropna=False)["Viajero"]
        .agg(n_registros="size", suma_viajero="sum")
        .reset_index()
    )
    posibles = por_dims[por_dims["n_registros"] > 1].copy()
    resumen = pd.DataFrame(
        {
            "tipo": ["duplicados_completos", "posibles_duplicados_dimensionales"],
            "conteo": [completos, len(posibles)],
        }
    )
    if not posibles.empty:
        posibles["tipo"] = "detalle_posibles_duplicados_dimensionales"
        cols = ["tipo", "n_registros", "suma_viajero", *dims]
        detalle = posibles[cols]
        return pd.concat([resumen, detalle], ignore_index=True, sort=False)
    return resumen


def detectar_atipicos(
    df: pd.DataFrame, group_cols: list[str] | None = None, value_col: str = "Viajero"
) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ["Tipo de Viajero"]

    records: list[dict[str, Any]] = []

    q1 = df[value_col].quantile(0.25)
    q3 = df[value_col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    global_mask = (df[value_col] < low) | (df[value_col] > high)
    records.append(
        {
            "alcance": "global",
            "grupo": "dataset",
            "n_outliers": int(global_mask.sum()),
            "limite_inferior": float(low),
            "limite_superior": float(high),
        }
    )

    grouped = df.groupby(group_cols, dropna=False)
    for key, gdf in grouped:
        if len(gdf) < 4:
            continue
        gq1 = gdf[value_col].quantile(0.25)
        gq3 = gdf[value_col].quantile(0.75)
        giqr = gq3 - gq1
        glow = gq1 - 1.5 * giqr
        ghigh = gq3 + 1.5 * giqr
        mask = (gdf[value_col] < glow) | (gdf[value_col] > ghigh)
        label = key if isinstance(key, str) else " | ".join(map(str, key))
        records.append(
            {
                "alcance": "grupo",
                "grupo": label,
                "n_outliers": int(mask.sum()),
                "limite_inferior": float(glow),
                "limite_superior": float(ghigh),
            }
        )

    out = pd.DataFrame(records)
    out["nota"] = (
        "No eliminar automaticamente; revisar pandemia, estacionalidad y cambios reales."
    )
    return out.sort_values(["alcance", "n_outliers"], ascending=[True, False])


def dividir_train_test(
    serie: pd.Series, train_ratio: float = TRAIN_RATIO
) -> tuple[pd.Series, pd.Series]:
    n = len(serie)
    n_train = max(PERIODO_ESTACIONAL * 2, int(np.floor(n * train_ratio)))
    n_train = min(n_train, n - 1)
    return serie.iloc[:n_train].copy(), serie.iloc[n_train:].copy()


def obtener_top_n(
    df: pd.DataFrame, categoria: str, n: int = 3, value_col: str = "Viajero"
) -> pd.DataFrame:
    return (
        df.groupby(categoria, dropna=False)[value_col]
        .sum()
        .sort_values(ascending=False)
        .reset_index(name="viajeros")
        .head(n)
    )


def construir_serie_mensual(
    df: pd.DataFrame,
    nombre: str,
    filtros: dict[str, Any] | None = None,
    fecha_min: pd.Timestamp | None = None,
    fecha_max: pd.Timestamp | None = None,
) -> pd.Series:
    work = df.copy()
    if filtros:
        for col, value in filtros.items():
            if isinstance(value, (list, tuple, set)):
                work = work[work[col].isin(list(value))]
            else:
                work = work[work[col] == value]
    serie = work.groupby("fecha")["Viajero"].sum().sort_index()
    if fecha_min is None:
        fecha_min = serie.index.min()
    if fecha_max is None:
        fecha_max = serie.index.max()
    idx = pd.date_range(fecha_min, fecha_max, freq=FRECUENCIA, name="fecha")
    serie = serie.reindex(idx)
    serie.name = nombre
    return serie


def validar_serie_mensual(serie: pd.Series) -> dict[str, Any]:
    inferred = pd.infer_freq(serie.index)
    return {
        "nombre": serie.name,
        "inicio": serie.index.min().date().isoformat(),
        "fin": serie.index.max().date().isoformat(),
        "observaciones": int(len(serie)),
        "faltantes": int(serie.isna().sum()),
        "frecuencia_inferida": inferred or "no_inferida",
    }


def graficar_serie(
    serie: pd.Series,
    nombre: str,
    path: Path,
    train: pd.Series | None = None,
    test: pd.Series | None = None,
    zoom: tuple[str, str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    if train is not None and test is not None:
        train.plot(ax=ax, label="Train", color="#1f77b4", linewidth=1.8)
        test.plot(ax=ax, label="Test", color="#d62728", linewidth=1.8)
        ax.axvline(train.index.max(), color="black", linestyle="--", linewidth=1)
    else:
        serie.plot(ax=ax, color="#1f77b4", linewidth=1.8)

    pandemic_start = pd.Timestamp("2020-03-01")
    pandemic_end = pd.Timestamp("2021-12-01")
    ax.axvspan(pandemic_start, pandemic_end, color="#ffcc99", alpha=0.3, label="Pandemia")
    if zoom:
        ax.set_xlim(pd.Timestamp(zoom[0]), pd.Timestamp(zoom[1]))
    ax.set_title(nombre)
    ax.set_ylabel("Viajeros")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def descomponer_serie(
    serie: pd.Series, nombre: str, periodo: int, output_dir: Path
) -> tuple[str, str]:
    clean = serie.dropna()
    additive_path = output_dir / f"descomposicion_{slugify(nombre)}_aditiva.png"
    multiplicative_path = output_dir / f"descomposicion_{slugify(nombre)}_multiplicativa.png"
    notes: list[str] = []

    if len(clean) < periodo * 2:
        notes.append("No hay suficientes observaciones para descomposicion.")
        return "", " ".join(notes)

    decomp_add = seasonal_decompose(clean, model="additive", period=periodo)
    fig = decomp_add.plot()
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    fig.savefig(additive_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if (clean <= 0).any():
        notes.append("Multiplicativa no valida por ceros o valores no positivos.")
        return additive_path.name, " ".join(notes)

    decomp_mul = seasonal_decompose(clean, model="multiplicative", period=periodo)
    fig = decomp_mul.plot()
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    fig.savefig(multiplicative_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return f"{additive_path.name}; {multiplicative_path.name}", " ".join(notes) or "Aditiva y multiplicativa generadas."


def evaluar_transformacion(serie_train: pd.Series) -> tuple[str, str]:
    clean = serie_train.dropna()
    if clean.empty:
        return "sin_transformacion", "Serie sin datos disponibles en entrenamiento."

    rolling_mean = clean.rolling(PERIODO_ESTACIONAL, min_periods=max(3, PERIODO_ESTACIONAL // 2)).mean()
    rolling_std = clean.rolling(PERIODO_ESTACIONAL, min_periods=max(3, PERIODO_ESTACIONAL // 2)).std()
    corr = rolling_mean.corr(rolling_std)
    min_value = float(clean.min())

    if min_value <= 0:
        if corr is not None and not np.isnan(corr) and corr > 0.35:
            return "log1p", "Hay valores no positivos y relacion positiva entre nivel y varianza; se sugiere log1p."
        return "sin_transformacion", "No se recomienda log por ceros y no hay evidencia fuerte de heterocedasticidad."

    if corr is not None and not np.isnan(corr) and corr > 0.35:
        try:
            lmbda = boxcox_normmax(clean.values, method="mle")
            if abs(lmbda) < 0.25:
                return "log", f"Relacion positiva entre nivel y varianza; lambda Box-Cox cercana a 0 ({lmbda:.2f})."
            return "boxcox_aproximada", f"Relacion positiva entre nivel y varianza; Box-Cox sugeriria lambda {lmbda:.2f}."
        except Exception:
            return "log", "Relacion positiva entre nivel y varianza; se sugiere log."
    return "sin_transformacion", "La amplitud estacional luce relativamente estable en entrenamiento."


def _transform_series(serie: pd.Series, transformacion: str) -> pd.Series:
    if transformacion == "log":
        return np.log(serie)
    if transformacion == "log1p":
        return np.log1p(serie)
    return serie.copy()


def graficar_acf_pacf(
    serie: pd.Series, nombre: str, output_dir: Path, suffix: str = "nivel"
) -> tuple[str, str]:
    clean = serie.dropna()
    nlags = min(36, max(1, len(clean) // 2 - 1))
    acf_path = output_dir / f"acf_{slugify(nombre)}_{suffix}.png"
    pacf_path = output_dir / f"pacf_{slugify(nombre)}_{suffix}.png"

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(clean, ax=ax, lags=nlags, zero=False)
    ax.set_title(f"ACF - {nombre} ({suffix})")
    fig.tight_layout()
    fig.savefig(acf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pacf_lags = min(nlags, max(1, len(clean) // 2 - 2))
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_pacf(clean, ax=ax, lags=pacf_lags, zero=False, method="ywm")
    ax.set_title(f"PACF - {nombre} ({suffix})")
    fig.tight_layout()
    fig.savefig(pacf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return acf_path.name, pacf_path.name


def ejecutar_adf(serie: pd.Series) -> tuple[float | None, float | None]:
    clean = serie.dropna()
    if len(clean) < 10 or clean.nunique() <= 1:
        return None, None
    try:
        stat, pvalue, *_ = adfuller(clean, autolag="AIC")
        return float(stat), float(pvalue)
    except Exception:
        return None, None


def diferenciar_serie(
    serie: pd.Series, lag: int = 1, seasonal_lag: int | None = None
) -> pd.Series:
    out = serie.diff(lag)
    if seasonal_lag:
        out = out.diff(seasonal_lag)
    return out


def _infer_d_orders(serie_train: pd.Series, transformacion: str) -> tuple[int, int, str]:
    transformed = _transform_series(serie_train.dropna(), transformacion)
    _, p_level = ejecutar_adf(transformed)
    diff1 = diferenciar_serie(transformed, lag=1)
    _, p_diff1 = ejecutar_adf(diff1)
    seasonal = diferenciar_serie(transformed, lag=PERIODO_ESTACIONAL)
    _, p_seasonal = ejecutar_adf(seasonal)
    seasonal_diff = diferenciar_serie(diff1, lag=PERIODO_ESTACIONAL)
    _, p_both = ejecutar_adf(seasonal_diff)

    d = 0 if p_level is not None and p_level < 0.05 else 1
    D = 0
    if p_seasonal is not None and p_seasonal < 0.05 and d == 0:
        D = 1
    elif p_both is not None and p_both < 0.05 and p_diff1 is not None and p_diff1 >= 0.05:
        D = 1

    comment = (
        f"ADF nivel={p_level if p_level is not None else 'NA'}, "
        f"ADF diff1={p_diff1 if p_diff1 is not None else 'NA'}, "
        f"ADF diff estacional={p_seasonal if p_seasonal is not None else 'NA'}, "
        f"ADF diff1+est={p_both if p_both is not None else 'NA'}."
    )
    return d, D, comment


def analizar_serie(
    serie: pd.Series,
    nombre: str,
    periodo: int = PERIODO_ESTACIONAL,
    output_dir: Path = FIGURAS_DIR,
) -> SeriesAnalysisResult:
    serie = serie.asfreq(FRECUENCIA)
    train, test = dividir_train_test(serie)
    slug = slugify(nombre)

    train_df = train.rename("valor").rename_axis("fecha").reset_index()
    test_df = test.rename("valor").rename_axis("fecha").reset_index()
    train_file = SERIES_DIR / f"{slug}_train.csv"
    test_file = SERIES_DIR / f"{slug}_test.csv"
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    graficar_serie(serie, f"Serie completa - {nombre}", output_dir / f"serie_{slug}_completa.png")
    graficar_serie(
        serie,
        f"Serie train/test - {nombre}",
        output_dir / f"serie_{slug}_split.png",
        train=train,
        test=test,
    )
    graficar_serie(
        serie,
        f"Zoom 2019-2022 - {nombre}",
        output_dir / f"serie_{slug}_zoom_2019_2022.png",
        zoom=("2019-01-01", "2022-12-01"),
    )

    transformacion, justificacion = evaluar_transformacion(train)
    transformed_train = _transform_series(train.dropna(), transformacion)
    adf_nivel, pvalue_nivel = ejecutar_adf(transformed_train)
    diff1 = diferenciar_serie(transformed_train, lag=1)
    adf_diff1, pvalue_diff1 = ejecutar_adf(diff1)
    d_sugerido, D_sugerido, comentario_adf = _infer_d_orders(train, transformacion)

    decomp_files, decomp_note = descomponer_serie(train, nombre, periodo, output_dir)
    graficar_acf_pacf(transformed_train, nombre, output_dir, suffix="nivel")
    if not diff1.dropna().empty:
        graficar_acf_pacf(diff1.dropna(), nombre, output_dir, suffix="diff1")

    interpretacion = (
        f"{nombre}: {len(serie)} observaciones entre {serie.index.min().date()} y "
        f"{serie.index.max().date()}; transformacion sugerida={transformacion}, "
        f"d={d_sugerido}, D={D_sugerido}."
    )
    comentario = f"{comentario_adf} {decomp_note}".strip()

    return SeriesAnalysisResult(
        serie=nombre,
        archivo_train=train_file.name,
        archivo_test=test_file.name,
        inicio=serie.index.min().date().isoformat(),
        fin=serie.index.max().date().isoformat(),
        frecuencia=FRECUENCIA,
        n_observaciones=len(serie),
        fecha_final_train=train.index.max().date().isoformat(),
        fecha_inicial_test=test.index.min().date().isoformat(),
        transformacion=transformacion,
        justificacion_transformacion=justificacion,
        adf_nivel=adf_nivel,
        pvalue_nivel=pvalue_nivel,
        adf_diff1=adf_diff1,
        pvalue_diff1=pvalue_diff1,
        d_sugerido=d_sugerido,
        D_sugerido=D_sugerido,
        comentario=f"{comentario} Descomposiciones: {decomp_files or 'no_generadas'}.",
        interpretacion=interpretacion,
    )


def _save_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLAS_DIR / name
    df.to_csv(path, index=False)
    return path


def _plot_bar(series: pd.Series, title: str, filename: str, xlabel: str = "Viajeros") -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    series.sort_values().plot(kind="barh", ax=ax, color="#4c78a8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap(df: pd.DataFrame, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df, cmap="YlGnBu", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generar_eda(df: pd.DataFrame, df_filtrado: pd.DataFrame) -> dict[str, str]:
    fig_paths: dict[str, str] = {}

    total_mensual = construir_serie_mensual(
        df_filtrado,
        "Total mensual",
        fecha_min=df_filtrado["fecha"].min(),
        fecha_max=df_filtrado["fecha"].max(),
    )
    graficar_serie(total_mensual, "Volumen mensual filtrado", FIGURAS_DIR / "eda_total_mensual.png")
    fig_paths["eda_total_mensual"] = "eda_total_mensual.png"

    top_paises = obtener_top_n(df_filtrado, "País", 10).set_index("País")["viajeros"]
    _plot_bar(top_paises, "Top 10 paises por volumen", "eda_top_paises.png")
    fig_paths["eda_top_paises"] = "eda_top_paises.png"

    top_regiones = obtener_top_n(df_filtrado, "Región", 10).set_index("Región")["viajeros"]
    _plot_bar(top_regiones, "Top regiones por volumen", "eda_top_regiones.png")
    fig_paths["eda_top_regiones"] = "eda_top_regiones.png"

    vias = (
        df_filtrado.groupby("Vía")["Viajero"].sum().sort_values()
    )
    _plot_bar(vias, "Vias de ingreso", "eda_vias.png")
    fig_paths["eda_vias"] = "eda_vias.png"

    fronteras = obtener_top_n(df_filtrado, "Frontera", 10).set_index("Frontera")["viajeros"]
    _plot_bar(fronteras, "Top 10 fronteras", "eda_top_fronteras.png")
    fig_paths["eda_top_fronteras"] = "eda_top_fronteras.png"

    tipos = (
        df.groupby("Tipo de Viajero")["Viajero"].sum().sort_values()
    )
    _plot_bar(tipos, "Tipo de viajero", "eda_tipo_viajero.png")
    fig_paths["eda_tipo_viajero"] = "eda_tipo_viajero.png"

    via_frontera = (
        df_filtrado.pivot_table(
            index="Frontera", columns="Vía", values="Viajero", aggfunc="sum", fill_value=0
        )
        .sort_values(by=list(df_filtrado["Vía"].dropna().unique()), ascending=False)
        .head(12)
    )
    _plot_heatmap(via_frontera, "Cruce via por frontera", "eda_via_por_frontera.png")
    fig_paths["eda_via_por_frontera"] = "eda_via_por_frontera.png"

    region_via = df_filtrado.pivot_table(
        index="Región dos", columns="Vía", values="Viajero", aggfunc="sum", fill_value=0
    )
    _plot_heatmap(region_via, "Cruce region por via", "eda_region_por_via.png")
    fig_paths["eda_region_por_via"] = "eda_region_por_via.png"

    tipo_anio = df.pivot_table(
        index="Año", columns="Tipo de Viajero", values="Viajero", aggfunc="sum", fill_value=0
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    tipo_anio.plot(ax=ax, linewidth=2)
    ax.set_title("Tipo de viajero por año")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / "eda_tipo_por_anio.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig_paths["eda_tipo_por_anio"] = "eda_tipo_por_anio.png"

    return fig_paths


def _build_manifest(
    resultados: list[SeriesAnalysisResult],
    fig_paths: dict[str, str],
    extra_tables: list[str],
) -> Path:
    lines = [
        "# Manifest Parte 1",
        "",
        "## Series exactas",
    ]
    for result in resultados:
        lines.extend(
            [
                f"- Serie: {result.serie}",
                f"  - Train/Test: series/{result.archivo_train}, series/{result.archivo_test}",
                f"  - Transformacion sugerida: {result.transformacion}",
                f"  - d sugerido: {result.d_sugerido}",
                f"  - D sugerido: {result.D_sugerido}",
                f"  - Observaciones: {result.comentario}",
            ]
        )

    lines.extend(["", "## Tablas", *[f"- tablas/{name}" for name in extra_tables]])
    lines.extend(["", "## Figuras"])
    lines.extend([f"- figuras/{path}" for path in sorted(fig_paths.values())])
    lines.extend(["", f"- {EDA_WARNING}"])

    manifest = OUTPUT_DIR / "manifest.md"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def run() -> dict[str, Any]:
    _ensure_dirs()

    df = cargar_datos()
    df = preparar_fecha(df)
    df_filtrado = filtrar_tipos_consistentes(df)

    resumen_dataset, faltantes = resumir_calidad_datos(df)
    duplicados = detectar_duplicados(df)
    atipicos = detectar_atipicos(df, group_cols=["Tipo de Viajero", "Vía"])

    _save_table(resumen_dataset, "resumen_dataset.csv")
    _save_table(faltantes, "faltantes.csv")
    _save_table(duplicados, "duplicados.csv")
    _save_table(atipicos, "atipicos.csv")

    top_paises = obtener_top_n(df_filtrado, "País", 10)
    top_regiones = obtener_top_n(df_filtrado, "Región", 10)
    top_fronteras = obtener_top_n(df_filtrado, "Frontera", 10)
    top_3_fronteras = obtener_top_n(df_filtrado, "Frontera", 3)

    _save_table(top_paises, "top_paises.csv")
    _save_table(top_regiones, "top_regiones.csv")
    _save_table(top_fronteras, "top_fronteras.csv")
    _save_table(top_3_fronteras, "top_3_fronteras.csv")

    fig_paths = generar_eda(df, df_filtrado)

    fecha_min = df_filtrado["fecha"].min()
    fecha_max = df_filtrado["fecha"].max()

    series_map: dict[str, pd.Series] = {
        "total": construir_serie_mensual(
            df_filtrado,
            "Total mensual",
            fecha_min=fecha_min,
            fecha_max=fecha_max,
        )
    }

    for frontera in top_3_fronteras["Frontera"].tolist():
        label = f"frontera_{frontera}"
        pretty = f"Frontera {frontera}"
        series_map[label] = construir_serie_mensual(
            df_filtrado,
            pretty,
            filtros={"Frontera": frontera},
            fecha_min=fecha_min,
            fecha_max=fecha_max,
        )

    for via in ["Aérea", "Terrestre", "Marítima"]:
        series_map[f"via_{slugify(via)}"] = construir_serie_mensual(
            df_filtrado,
            f"Vía {via}",
            filtros={"Vía": via},
            fecha_min=fecha_min,
            fecha_max=fecha_max,
        )

    series_validation = pd.DataFrame([validar_serie_mensual(s) for s in series_map.values()])
    _save_table(series_validation, "validacion_series.csv")

    total = series_map["total"]
    suma_vias = (
        series_map["via_aerea"].fillna(0)
        + series_map["via_terrestre"].fillna(0)
        + series_map["via_maritima"].fillna(0)
    )
    diff_vias = (total.fillna(0) - suma_vias).abs()

    resultados: list[SeriesAnalysisResult] = []
    ordered_keys = [
        "total",
        *[f"frontera_{frontera}" for frontera in top_3_fronteras["Frontera"].tolist()],
        "via_aerea",
        "via_terrestre",
        "via_maritima",
    ]
    for key in ordered_keys:
        resultados.append(analizar_serie(series_map[key], series_map[key].name))

    resultados_df = pd.DataFrame(
        [
            {
                "serie": r.serie,
                "adf_nivel": r.adf_nivel,
                "pvalue_nivel": r.pvalue_nivel,
                "adf_diff1": r.adf_diff1,
                "pvalue_diff1": r.pvalue_diff1,
                "d_sugerido": r.d_sugerido,
                "D_sugerido": r.D_sugerido,
                "transformacion": r.transformacion,
                "comentario": r.comentario,
            }
            for r in resultados
        ]
    )
    _save_table(resultados_df, "resultados_estacionariedad.csv")

    extra_tables = [
        "resumen_dataset.csv",
        "faltantes.csv",
        "duplicados.csv",
        "atipicos.csv",
        "top_paises.csv",
        "top_regiones.csv",
        "top_fronteras.csv",
        "top_3_fronteras.csv",
        "validacion_series.csv",
        "resultados_estacionariedad.csv",
    ]
    manifest = _build_manifest(resultados, fig_paths, extra_tables)

    return {
        "n_series": len(resultados),
        "fecha_corte": resultados[0].fecha_final_train,
        "diff_vias_max": float(diff_vias.max()),
        "manifest": str(manifest),
        "resultados": resultados,
        "top_3_fronteras": top_3_fronteras,
        "warning_pais": EDA_WARNING,
        "resumen_dataset": resumen_dataset,
        "series_validation": series_validation,
    }
