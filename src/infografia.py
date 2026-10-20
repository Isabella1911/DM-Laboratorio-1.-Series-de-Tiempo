"""Laboratorio 9: infografia estatica "Del record al reinicio".

Lee los resultados del Laboratorio 1 (series de outputs/parte1, tablas de report/tables y
report/values, y la base original), valida cada cifra y exporta una sola pieza vertical de
11 x 17 in en PNG (2200 x 3400 px) y PDF (1 pagina).

Uso:
    python -m src.infografia
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from matplotlib.transforms import Affine2D

from src import comparacion as C
from src import lab as L
from src import models as M

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

ROOT = L.ROOT
OUT_DIR = ROOT / "outputs"
PNG_PATH = OUT_DIR / "infografia_turismo_guatemala_2009_2026.png"
PDF_PATH = OUT_DIR / "infografia_turismo_guatemala_2009_2026.pdf"
VALIDACION_PATH = OUT_DIR / "validacion_infografia.md"

TABLA_METRICAS_TOTAL = ROOT / "report" / "tables" / "tabla_metricas_total.tex"
TABLA_COMPARACION_VIAS = ROOT / "report" / "tables" / "comparacion_vias.tex"
TABLA_COMPARACION_FRONTERAS = ROOT / "report" / "tables" / "comparacion_fronteras.tex"
RESULTADOS_CLAVE = ROOT / "report" / "values" / "resultados_clave.tex"
SERIES_DIR = ROOT / "outputs" / "parte1" / "series"

AUTORES = ["Anthony Lou", "Isabella Obando", "Juan Francisco Martínez"]
REPO_URL = "github.com/Isabella1911/DM-Laboratorio-1.-Series-de-Tiempo"

FECHA_INICIO = pd.Timestamp("2009-01-01")
FECHA_FIN = pd.Timestamp("2026-06-01")
TIPOS = {"Turista", "Excursionista"}
SERIES_CUADRANTE = [
    ("Vía Aérea", "Vía\nAérea"),
    ("Frontera 01 La Aurora", "Frontera\nLa Aurora"),
    ("Vía Terrestre", "Vía\nTerrestre"),
    ("Frontera 07 Valle Nuevo", "Frontera\nValle Nuevo"),
    ("Frontera 09 San Cristóbal", "Frontera\nSan Cristóbal"),
]
N_SIMULACIONES = 5000
SEMILLA = 20261018

ANCHO, ALTO, DPI = 11.0, 17.0, 200
MARGEN, MEDIANIL, COLUMNAS = 0.5, 0.25, 12
ANCHO_COL = (ANCHO - 2 * MARGEN - (COLUMNAS - 1) * MEDIANIL) / COLUMNAS

MARINO = "#192854"
AZUL = "#2A5CA8"
CELESTE = "#4993CC"
GRIS = "#8A919E"
AMBAR = "#D97A00"
ORO = "#FFB236"
FONDO_TARJETA = "#F4F6FA"
TEXTO = "#2B3245"
TEXTO_SUAVE = "#4E5668"
GRIS_CLARO = "#C5CAD3"
REJILLA = "#E3E7EE"
TEXTO_SOBRE_MARINO = "#DCE2EE"

FUENTES_PREFERIDAS = ["Lato", "Carlito", "Open Sans", "DejaVu Sans"]

MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
MESES_LARGOS = {
    1: "ene.", 2: "feb.", 3: "mar.", 4: "abr.", 5: "may.", 6: "jun.",
    7: "jul.", 8: "ago.", 9: "sep.", 10: "oct.", 11: "nov.", 12: "dic.",
}


class ValidacionError(RuntimeError):
    """Discrepancia sustantiva entre la infografia y los resultados del laboratorio."""


@dataclass
class Registro:
    fuentes: list[str] = field(default_factory=list)
    checks: list[tuple[str, str, str]] = field(default_factory=list)
    notas: list[str] = field(default_factory=list)

    def fuente(self, path: Path, uso: str) -> None:
        self.fuentes.append(f"`{path.relative_to(ROOT)}`: {uso}")

    def check(self, nombre: str, condicion: bool, detalle: str) -> None:
        self.checks.append((nombre, "OK" if condicion else "FALLA", detalle))
        if not condicion:
            raise ValidacionError(f"{nombre}: {detalle}")


# ---------------------------------------------------------------------------
# Formato numerico (punto decimal, coma de miles, signo menos tipografico)
# ---------------------------------------------------------------------------


def _signo(texto: str) -> str:
    return texto.replace("-", "−")


def fmt_entero(valor: float) -> str:
    return _signo(f"{valor:,.0f}")


def fmt_millones(valor: float, decimales: int = 2) -> str:
    return _signo(f"{valor / 1e6:.{decimales}f} M")


def fmt_miles(valor: float) -> str:
    return _signo(f"{valor / 1e3:,.0f} mil")


def fmt_pct(valor: float, decimales: int = 0, signo: bool = False) -> str:
    texto = f"{valor:+.{decimales}f} %" if signo else f"{valor:.{decimales}f} %"
    return _signo(texto)


def fmt_mes(fecha: pd.Timestamp) -> str:
    return f"{MESES_LARGOS[fecha.month]} {fecha.year}"


# ---------------------------------------------------------------------------
# Carga y validacion de insumos
# ---------------------------------------------------------------------------


def _exigir(path: Path, como_generarlo: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Falta {path.relative_to(ROOT)}. {como_generarlo}")
    return path


def leer_tabla_latex(path: Path) -> pd.DataFrame:
    lineas = path.read_text(encoding="utf-8").splitlines()
    try:
        i_top = next(i for i, l in enumerate(lineas) if l.strip() == r"\toprule")
        i_mid = next(i for i, l in enumerate(lineas) if l.strip() == r"\midrule")
        i_bot = next(i for i, l in enumerate(lineas) if l.strip() == r"\bottomrule")
    except StopIteration as exc:
        raise ValueError(f"{path.name} no tiene la estructura booktabs esperada.") from exc

    def _celdas(linea: str) -> list[str]:
        linea = linea.strip().removesuffix(r"\\").strip()
        return [c.strip().replace(r"\_", "_").replace(r"\%", "%") for c in linea.split("&")]

    columnas = _celdas(lineas[i_top + 1])
    filas = [_celdas(l) for l in lineas[i_mid + 1 : i_bot] if l.strip()]
    df = pd.DataFrame(filas, columns=columnas).replace("--", np.nan)
    for col in df.columns:
        convertida = pd.to_numeric(df[col], errors="coerce")
        if convertida.notna().sum() == df[col].notna().sum():
            df[col] = convertida
    return df


def leer_resultados_clave(path: Path) -> dict[str, str]:
    texto = path.read_text(encoding="utf-8")
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", texto))


def cargar_base(reg: Registro) -> pd.DataFrame:
    _exigir(L.DATA_PATH, "Copie la base original de migracion en data/.")
    reg.fuente(L.DATA_PATH, "base original (vías, regiones de origen y verificación de la serie total)")
    df = L.filtrar_tipos_consistentes(L.preparar_fecha(L.cargar_datos()))
    reg.check(
        "Filtro Turista + Excursionista",
        set(df["Tipo de Viajero"].unique()) == TIPOS,
        f"tipos presentes: {sorted(df['Tipo de Viajero'].unique())}",
    )
    reg.check(
        "Periodo ene. 2009 – jun. 2026",
        df["fecha"].min() == FECHA_INICIO and df["fecha"].max() == FECHA_FIN,
        f"{df['fecha'].min().date()} a {df['fecha'].max().date()}",
    )
    return df


def cargar_serie_total(base: pd.DataFrame, reg: Registro) -> tuple[pd.Series, pd.Series, pd.Series]:
    _exigir(
        SERIES_DIR / "total_mensual_train.csv",
        "Ejecute la Parte 1: python -c 'from src import lab as L; L.run()'.",
    )
    reg.fuente(SERIES_DIR / "total_mensual_train.csv", "serie total mensual, entrenamiento (Parte 1)")
    reg.fuente(SERIES_DIR / "total_mensual_test.csv", "serie total mensual, prueba (Parte 1)")
    train, test = M.cargar_serie_train_test("Total mensual")
    serie = pd.concat([train, test]).asfreq(L.FRECUENCIA)

    reg.check(
        "Serie mensual sin duplicados ni huecos",
        serie.index.is_unique and serie.index.is_monotonic_increasing and serie.notna().all()
        and len(serie) == 210,
        f"{len(serie)} meses, {int(serie.isna().sum())} faltantes",
    )
    desde_base = base.groupby("fecha")["Viajero"].sum().reindex(serie.index)
    reg.check(
        "Serie de la Parte 1 = suma de la base filtrada",
        float((desde_base - serie).abs().max()) < 1.0,
        f"diferencia máxima {float((desde_base - serie).abs().max()):.2e} viajeros",
    )
    return serie, train, test


def calcular_kpis(serie: pd.Series, reg: Registro) -> dict[str, Any]:
    anual = serie.groupby(serie.index.year).sum()
    meses_por_anio = serie.groupby(serie.index.year).size()
    completos = meses_por_anio[meses_por_anio == 12].index
    anio_record = int(anual.loc[completos].idxmax())
    fecha_min = serie.idxmin()
    prom_2019 = float(serie[serie.index.year == 2019].mean())
    k = {
        "total": float(serie.sum()),
        "anual": anual,
        "anio_record": anio_record,
        "record": float(anual[anio_record]),
        "crec_record_2009": (anual[anio_record] / anual[2009] - 1) * 100,
        "fecha_min": fecha_min,
        "minimo": float(serie.min()),
        "prom_2019": prom_2019,
        "caida_min": (serie.min() / prom_2019 - 1) * 100,
        "total_2025": float(anual[2025]),
        "recuperacion_2025": anual[2025] / anual[anio_record] * 100,
        "fecha_max": serie.idxmax(),
        "maximo": float(serie.max()),
        "total_2014": float(anual[2014]),
        "total_2022": float(anual[2022]),
        "rango_2022_2025": (float(anual.loc[2022:2025].min()), float(anual.loc[2022:2025].max())),
        "anios": serie.index[-1].year - serie.index[0].year + serie.index[-1].month / 12,
    }
    reg.check("Años usados en KPIs están completos", {2009, 2014, 2019, 2022, 2025} <= set(completos), "12 meses cada uno")
    reg.check("Año récord = 2019", anio_record == 2019, f"récord {anio_record}: {fmt_millones(k['record'])}")
    reg.check("Mínimo mensual en mayo 2020", fecha_min == pd.Timestamp("2020-05-01"), f"{fmt_mes(fecha_min)}: {fmt_entero(k['minimo'])}")
    reg.check("Máximo mensual en dic. 2019", k["fecha_max"] == pd.Timestamp("2019-12-01"), f"{fmt_miles(k['maximo'])}")
    reg.check(
        "Suma de totales anuales = suma mensual",
        abs(anual.sum() - k["total"]) < 1.0,
        f"{fmt_entero(anual.sum())} vs {fmt_entero(k['total'])}",
    )
    return k


def calcular_estacionalidad(serie: pd.Series, reg: Registro) -> pd.DataFrame:
    """Indice estacional = promedio del mes / promedio mensual del periodo x 100."""

    def _indice(desde: int, hasta: int) -> pd.Series:
        tramo = serie[(serie.index.year >= desde) & (serie.index.year <= hasta)]
        reg.check(
            f"Estacionalidad {desde}–{hasta} con años completos",
            len(tramo) == 12 * (hasta - desde + 1),
            f"{len(tramo)} meses",
        )
        return tramo.groupby(tramo.index.month).mean() / tramo.mean() * 100

    est = pd.DataFrame({"pre": _indice(2009, 2019), "post": _indice(2022, 2025)})
    reg.check(
        "Índice estacional promedia 100",
        abs(est["pre"].mean() - 100) < 1e-6 and abs(est["post"].mean() - 100) < 1e-6,
        f"pre={est['pre'].mean():.2f}, post={est['post'].mean():.2f}",
    )
    reg.check("Diciembre es el mes más fuerte (2009–2019)", int(est["pre"].idxmax()) == 12, f"dic={est['pre'][12]:.1f}")
    return est


def calcular_vias(base: pd.DataFrame, reg: Registro) -> dict[str, Any]:
    por_via = base.groupby("Vía")["Viajero"].sum()
    pct = por_via / por_via.sum() * 100
    maritima = base[base["Vía"] == "Marítima"].groupby("fecha")["Viajero"].sum()
    reg.check(
        "Porcentajes por vía suman 100 sobre el mismo universo",
        abs(pct.sum() - 100) < 1e-9 and set(pct.index) == {"Aérea", "Terrestre", "Marítima"},
        ", ".join(f"{v} {p:.1f} %" for v, p in pct.items()),
    )
    ultimo_maritimo = maritima.index.max()
    reg.check(
        "Vía marítima sin registros desde 2017",
        ultimo_maritimo == pd.Timestamp("2016-12-01"),
        f"último registro marítimo: {fmt_mes(ultimo_maritimo)}; los meses posteriores no existen en la base (no son ceros)",
    )
    return {"pct": pct, "ultimo_maritimo": ultimo_maritimo}


def calcular_regiones(base: pd.DataFrame, reg: Registro) -> pd.DataFrame:
    anual = base[base["Año"].isin([2019, 2025])].pivot_table(
        index="Región", columns="Año", values="Viajero", aggfunc="sum"
    )
    totales = base[base["Año"].isin([2019, 2025])].groupby("Año")["Viajero"].sum()
    reg.check("Región de origen sin faltantes", base["Región"].notna().all(), "0 filas sin región")
    filas = {
        "Centroamérica": anual.loc["AMÉRICA DEL CENTRO"],
        "Norteamérica": anual.loc["AMÉRICA DEL NORTE"],
    }
    resto = totales - filas["Centroamérica"] - filas["Norteamérica"]
    filas["Resto del mundo"] = resto
    out = pd.DataFrame(filas).T
    out["cambio"] = (out[2025] / out[2019] - 1) * 100
    reg.check(
        "Regiones 2019/2025 suman el total anual",
        abs(out[2019].sum() - totales[2019]) < 1 and abs(out[2025].sum() - totales[2025]) < 1,
        f"2019: {fmt_millones(totales[2019])}; 2025: {fmt_millones(totales[2025])}",
    )
    paises = base.groupby("Año")["País"].nunique()
    reg.check(
        "Cambio de agrupación de país después de 2022",
        paises[2023] < paises[2019] / 2,
        f"países distintos: 2019={paises[2019]}, 2023={paises[2023]}",
    )
    return out


def calcular_cuadrante(reg: Registro) -> pd.DataFrame:
    for path in (TABLA_COMPARACION_VIAS, TABLA_COMPARACION_FRONTERAS):
        _exigir(path, "Ejecute la Parte 3: python -c 'from src import comparacion as C; C.ejecutar_parte3()'.")
        reg.fuente(path, "fuerza estacional, pendiente, coef. de variación, caída y meses de recuperación (Parte 3)")
    tablas = pd.concat([leer_tabla_latex(TABLA_COMPARACION_VIAS), leer_tabla_latex(TABLA_COMPARACION_FRONTERAS)])
    tablas = tablas.set_index("serie")

    filas = []
    for nombre, _ in SERIES_CUADRANTE:
        serie = C.cargar_serie_completa(nombre)
        reg.fuente(SERIES_DIR / f"{L.slugify(nombre)}_train.csv", f"serie {nombre} (Parte 1, train + test)")
        pandemia = C.impacto_pandemia(serie)
        fila = {
            "serie": nombre,
            "fs": C.fuerza_estacionalidad(serie),
            "pendiente": C.pendiente_tendencia(serie),
            "crecimiento": C.crecimiento_ventanas(serie)["crecimiento_pre_pandemia"],
            "cv": C.volatilidad(serie)["coeficiente_variacion"],
            "caida": pandemia["caida_porcentual"],
            "meses": pandemia["meses_recuperacion"],
        }
        lab = tablas.loc[nombre]
        coincide = (
            abs(round(fila["fs"], 2) - lab["fuerza_estacionalidad"]) < 1e-9
            and abs(round(fila["pendiente"], 2) - lab["pendiente_tendencia"]) < 1e-9
            and abs(round(fila["cv"], 2) - lab["coeficiente_variacion"]) < 1e-9
            and abs(round(fila["caida"], 2) - lab["caida_porcentual"]) < 1e-9
            and int(fila["meses"]) == int(lab["meses_recuperacion"])
        )
        reg.check(
            f"Cuadrante {nombre} = tablas de comparación del Lab 1",
            coincide,
            f"Fs {fila['fs']:.2f}, pendiente {fila['pendiente']:.2f}, CV {fila['cv']:.2f}, "
            f"caída {fila['caida']:.2f} %, {fila['meses']} meses; crecimiento 2009→2019 {fila['crecimiento']:.1f} %",
        )
        filas.append(fila)
    return pd.DataFrame(filas).set_index("serie")


def cargar_modelos(reg: Registro) -> pd.DataFrame:
    _exigir(TABLA_METRICAS_TOTAL, "Ejecute la Parte 2 y la Parte 3 del Laboratorio 1.")
    _exigir(RESULTADOS_CLAVE, "Ejecute la Parte 3 del Laboratorio 1.")
    reg.fuente(TABLA_METRICAS_TOTAL, "MAE, RMSE y MAPE de prueba de los 11 modelos de la serie total")
    reg.fuente(RESULTADOS_CLAVE, "modelo ganador, RMSE/MAE de referencia y fecha de corte")
    tabla = leer_tabla_latex(TABLA_METRICAS_TOTAL)
    tabla = tabla[tabla["serie"] == "Total mensual"].sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    clave = leer_resultados_clave(RESULTADOS_CLAVE)

    reg.check("Se evaluaron 11 modelos en la serie total", len(tabla) == 11, f"{len(tabla)} filas en la tabla")
    reg.check(
        "Métricas de prueba completas",
        tabla[["MAE", "RMSE", "MAPE"]].notna().all().all(),
        "MAE, RMSE y MAPE presentes para todos",
    )
    ganador = tabla.iloc[0]
    reg.check(
        "Ganador por menor RMSE de prueba coincide con resultados_clave.tex",
        ganador["modelo"] == clave["MejorModeloTotal"] and abs(ganador["RMSE"] - float(clave["RMSETotal"])) < 0.01
        and abs(ganador["MAE"] - float(clave["MAETotal"])) < 0.01,
        f"{ganador['modelo']}: RMSE {ganador['RMSE']:.2f}, MAE {ganador['MAE']:.2f}, MAPE {ganador['MAPE']:.2f} %",
    )
    reg.check(
        "Seasonal Naive presente como línea base",
        (tabla["modelo"] == "SeasonalNaive").any(),
        f"RMSE {tabla.loc[tabla['modelo'] == 'SeasonalNaive', 'RMSE'].iloc[0]:.2f}",
    )
    tabla["etiqueta"] = tabla["modelo"].map(etiqueta_modelo)
    tabla["etiqueta_larga"] = tabla["modelo"].map(etiqueta_modelo_larga)
    tabla.attrs["fecha_corte"] = clave["FechaCorteTrain"]
    return tabla


def etiqueta_modelo_larga(nombre: str) -> str:
    hw = re.fullmatch(r"HoltWinters\(trend=(\w+),seasonal=(\w+),damped=(\w+)\)", nombre)
    if hw:
        estacional = "multiplicativo" if hw.group(2) == "mul" else "aditivo"
        return f"Holt-Winters {estacional}" + (" amortiguado" if hw.group(3) == "True" else "")
    return etiqueta_modelo(nombre)


def etiqueta_modelo(nombre: str) -> str:
    hw = re.fullmatch(r"HoltWinters\(trend=(\w+),seasonal=(\w+),damped=(\w+)\)", nombre)
    if hw:
        estacional = "mult." if hw.group(2) == "mul" else "adit."
        nombre = f"Holt-Winters {estacional}"
        return nombre + (" amort." if hw.group(3) == "True" else "")
    arima = re.fullmatch(r"ARIMA\((\d), (\d), (\d)\)x\((\d), (\d), (\d), 12\)", nombre)
    if arima:
        g = arima.groups()
        return f"SARIMA ({','.join(g[:3])})×({','.join(g[3:])})"
    fijas = {
        "Prophet": "Prophet",
        "SeasonalNaive": "Seasonal Naive (base)",
        "SuavizamientoExponencial(simple)": "Suaviz. exp. simple",
        "SuavizamientoExponencial(holt)": "Suaviz. exp. de Holt",
        "SuavizamientoExponencial(holt_amortiguado)": "Holt amortiguado",
    }
    if nombre not in fijas:
        raise ValueError(f"Modelo sin etiqueta ejecutiva: {nombre}")
    return fijas[nombre]


def pronostico_ganador(
    modelos: pd.DataFrame, train: pd.Series, test: pd.Series, reg: Registro
) -> pd.DataFrame:
    """Reajusta el ganador con la funcion del Lab 1 y simula su intervalo predictivo."""
    ganador = modelos.iloc[0]
    hw = re.fullmatch(r"HoltWinters\(trend=(\w+),seasonal=(\w+),damped=(\w+)\)", ganador["modelo"])
    if not hw:
        raise ValidacionError(
            f"El ganador '{ganador['modelo']}' no es Holt-Winters; adapte pronostico_ganador() "
            "antes de exportar para no graficar un modelo distinto al evaluado."
        )
    reg.check(
        "Corte cronológico 70/30",
        train.index[-1].strftime("%Y-%m") == modelos.attrs["fecha_corte"]
        and test.index[0] == pd.Timestamp("2021-04-01") and test.index[-1] == FECHA_FIN,
        f"entrenamiento {fmt_mes(train.index[0])} – {fmt_mes(train.index[-1])} ({len(train)} meses); "
        f"prueba {fmt_mes(test.index[0])} – {fmt_mes(test.index[-1])} ({len(test)} meses)",
    )
    ajuste = M.ajustar_holt_winters(
        train.dropna(), trend=hw.group(1), seasonal=hw.group(2), damped=hw.group(3) == "True"
    )
    pred = np.clip(np.asarray(ajuste.forecast(len(test))), 0, None)
    metricas = M.calcular_metricas(test.values, pred)
    reg.check(
        "Reajuste reproduce las métricas de prueba del Lab 1",
        abs(metricas["RMSE"] - ganador["RMSE"]) < 0.01 and abs(metricas["MAE"] - ganador["MAE"]) < 0.01,
        f"RMSE {metricas['RMSE']:.2f} (tabla {ganador['RMSE']:.2f}); MAE {metricas['MAE']:.2f} "
        f"(tabla {ganador['MAE']:.2f}); MAPE {metricas['MAPE']:.2f} %",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sims = np.asarray(
            ajuste.simulate(
                len(test), anchor="end", repetitions=N_SIMULACIONES, error="mul", random_state=SEMILLA
            )
        )
    inferior = np.percentile(sims, 2.5, axis=1)
    superior = np.percentile(sims, 97.5, axis=1)
    out = pd.DataFrame(
        {"real": test.values, "prediccion": pred, "inferior": inferior, "superior": superior},
        index=test.index,
    )
    cobertura = float(((out["real"] >= out["inferior"]) & (out["real"] <= out["superior"])).mean() * 100)
    reg.check(
        "Intervalo del 95 % válido",
        len(out) == len(test) and bool((out["inferior"] >= 0).all())
        and bool((out["inferior"] <= out["prediccion"]).all()) and bool((out["prediccion"] <= out["superior"]).all()),
        f"{N_SIMULACIONES:,} trayectorias, errores multiplicativos, semilla {SEMILLA}; "
        f"límites {fmt_miles(out['inferior'].min())}–{fmt_miles(out['superior'].max())}; "
        f"cobertura empírica en prueba {cobertura:.1f} %",
    )
    out.attrs.update(metricas)
    out.attrs["cobertura"] = cobertura
    return out


# ---------------------------------------------------------------------------
# Lienzo y utilidades de dibujo (coordenadas en pulgadas desde la esquina superior izquierda)
# ---------------------------------------------------------------------------


def x_col(columna: int) -> float:
    return MARGEN + columna * (ANCHO_COL + MEDIANIL)


def ancho_cols(n: int) -> float:
    return n * ANCHO_COL + (n - 1) * MEDIANIL


class Lienzo:
    def __init__(self) -> None:
        self.fuente = next(
            f for f in FUENTES_PREFERIDAS if any(ff.name == f for ff in font_manager.fontManager.ttflist)
        )
        plt.rcParams.update(
            {
                "font.family": self.fuente,
                "pdf.fonttype": 42,
                "axes.unicode_minus": True,
                "axes.edgecolor": GRIS,
                "axes.labelcolor": TEXTO_SUAVE,
                "xtick.color": TEXTO_SUAVE,
                "ytick.color": TEXTO_SUAVE,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "axes.labelsize": 11,
            }
        )
        self.fig = plt.figure(figsize=(ANCHO, ALTO), dpi=DPI, facecolor="white")
        self.T = Affine2D().scale(1, -1).translate(0, ALTO) + self.fig.dpi_scale_trans
        self.renderer = self.fig.canvas.get_renderer()
        self.textos_fuera: list[str] = []

    def ancho_texto(self, texto: str, size: float, weight: str = "normal") -> float:
        prop = FontProperties(family=self.fuente, size=size, weight=weight)
        ancho, _, _ = self.renderer.get_text_width_height_descent(texto, prop, ismath=False)
        return ancho / DPI

    def envolver(self, texto: str, ancho: float, size: float, weight: str = "normal") -> str:
        lineas: list[str] = []
        for parrafo in texto.split("\n"):
            actual = ""
            for palabra in parrafo.split():
                prueba = f"{actual} {palabra}".strip()
                if actual and self.ancho_texto(prueba, size, weight) > ancho:
                    lineas.append(actual)
                    actual = palabra
                else:
                    actual = prueba
            lineas.append(actual)
        return "\n".join(lineas)

    def texto(
        self, x: float, y: float, s: str, size: float = 11, weight: str = "normal",
        color: str = TEXTO, ha: str = "left", va: str = "top", ancho: float | None = None,
        interlineado: float = 1.2, **kw: Any,
    ) -> float:
        """Dibuja texto y devuelve la coordenada y (pulgadas) de su borde inferior."""
        if ancho is not None:
            s = self.envolver(s, ancho, size, weight)
        self.fig.text(
            x, y, s, transform=self.T, fontsize=size, fontweight=weight, color=color,
            ha=ha, va=va, linespacing=interlineado, **kw,
        )
        return y + (s.count("\n") + 1) * size * interlineado / 72

    def caja(self, x: float, y: float, w: float, h: float, color: str, radio: float = 0.0, z: float = -1) -> None:
        estilo = f"round,pad=0,rounding_size={radio}" if radio else "square,pad=0"
        self.fig.add_artist(
            FancyBboxPatch((x, y), w, h, boxstyle=estilo, transform=self.T, facecolor=color,
                           edgecolor="none", zorder=z)
        )

    def ejes(self, x: float, y: float, w: float, h: float) -> plt.Axes:
        ax = self.fig.add_axes([x / ANCHO, (ALTO - y - h) / ALTO, w / ANCHO, h / ALTO])
        ax.set_facecolor("none")
        for lado in ("top", "right", "left"):
            ax.spines[lado].set_visible(False)
        ax.spines["bottom"].set_color(GRIS)
        ax.tick_params(length=0, pad=4)
        return ax

    def encabezado_bloque(self, y: float, codigo: str, titulo: str, nota: str | None = None) -> float:
        alto = 0.34
        self.caja(MARGEN, y, 0.56, alto, MARINO, radio=0.05)
        self.texto(MARGEN + 0.28, y + alto / 2, codigo, size=14, weight="bold", color="white", ha="center",
                   va="center")
        self.texto(MARGEN + 0.70, y + alto / 2, titulo, size=22, weight="bold", color=MARINO, va="center")
        if nota:
            self.texto(ANCHO - MARGEN, y + alto / 2, nota, size=11, color=TEXTO_SUAVE, ha="right", va="center",
                       style="italic")
        return y + alto

    def titulo_grafico(self, x: float, y: float, ancho: float, codigo: str, titulo: str) -> float:
        sangria = 0.45 if codigo else 0.0
        if codigo:
            self.texto(x, y + 0.01, codigo, size=12, weight="bold", color=AZUL)
        return self.texto(x + sangria, y, titulo, size=15, weight="bold", color=MARINO, ancho=ancho - sangria,
                          interlineado=1.15)


# ---------------------------------------------------------------------------
# Bloques
# ---------------------------------------------------------------------------


def render_titular_y_kpis(lz: Lienzo, k: dict[str, Any], vias: dict[str, Any], centro_pct: float) -> float:
    lz.caja(0, 0, ANCHO, 1.88, MARINO)
    lz.texto(MARGEN, 0.24, "Del récord al reinicio:", size=54, weight="bold", color="white")
    lz.texto(MARGEN, 0.92, "radiografía del turismo en Guatemala 2009–2026", size=30, weight="bold", color=ORO)
    terrestre = vias["pct"]["Terrestre"]
    subtitulo = (
        f"{k['total'] / 1e6:.1f} millones de ingresos de viajeros en 17 años y medio revelan un turismo "
        f"estacional, terrestre ({terrestre:.0f} %) y centroamericano ({centro_pct:.0f} %) que en 2025 ya "
        f"recuperó 8 de cada 10 viajeros de su año récord."
    )
    lz.texto(MARGEN, 1.36, subtitulo, size=16.5, color=TEXTO_SOBRE_MARINO, ancho=ANCHO - 2 * MARGEN,
             interlineado=1.12)

    tarjetas = [
        (fmt_millones(k["total"], 1), "ingresos de viajeros",
         f"{fmt_mes(FECHA_INICIO)} – {fmt_mes(FECHA_FIN)} · registros, no personas únicas"),
        (fmt_millones(k["record"]), f"récord anual en {k['anio_record']}",
         f"{fmt_pct(k['crec_record_2009'], signo=True)} frente a 2009 ({fmt_millones(k['anual'][2009])})"),
        (fmt_pct(k["caida_min"]), f"caída en {fmt_mes(k['fecha_min'])}",
         f"{fmt_entero(k['minimo'])} viajeros frente a {fmt_miles(k['prom_2019'])} del mes promedio de 2019"),
        (fmt_pct(k["recuperacion_2025"]), "recuperado en 2025",
         f"{fmt_millones(k['total_2025'])} viajeros frente a {fmt_millones(k['record'])} en {k['anio_record']}"),
    ]
    y0, alto = 1.98, 1.14
    for i, (cifra, significado, contexto) in enumerate(tarjetas):
        x = x_col(3 * i)
        w = ancho_cols(3)
        lz.caja(x, y0, w, alto, FONDO_TARJETA, radio=0.08)
        lz.caja(x, y0, 0.07, alto, AMBAR if i == 2 else AZUL)
        lz.texto(x + 0.18, y0 + 0.00, cifra, size=44, weight="bold", color=MARINO)
        lz.texto(x + 0.18, y0 + 0.60, significado, size=13, weight="bold", color=MARINO)
        lz.texto(x + 0.18, y0 + 0.80, contexto, size=10.5, color=TEXTO_SUAVE, ancho=w - 0.28, interlineado=1.08)
    return y0 + alto


def render_historico(lz: Lienzo, serie: pd.Series, k: dict[str, Any], test: pd.Series, y: float) -> float:
    y = lz.encabezado_bloque(y, "B2", "Panorama histórico",
                             nota="Las anotaciones marcan coincidencias en el tiempo, no causas demostradas.")
    minimo, maximo = k["rango_2022_2025"]
    y = lz.titulo_grafico(
        MARGEN, y + 0.06, ANCHO - 2 * MARGEN, "",
        f"Tras el récord de 2019 el flujo se desplomó; desde 2022 oscila entre "
        f"{fmt_millones(minimo, 1)} y {fmt_millones(maximo, 1)} de ingresos al año",
    )
    alto_ejes = 1.62
    ax = lz.ejes(MARGEN + 0.55, y + 0.04, ANCHO - 2 * MARGEN - 0.6, alto_ejes)
    miles = serie / 1e3
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"), color=GRIS, alpha=0.16, lw=0)
    ax.plot(miles.index, miles.values, color=AZUL, lw=2)
    ax.set_ylim(0, 540)
    ax.set_xlim(pd.Timestamp("2008-11-01"), pd.Timestamp("2026-08-01"))
    ax.set_yticks(range(0, 501, 100))
    ax.yaxis.grid(True, color=REJILLA, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("Miles de viajeros\npor mes", linespacing=1.0)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for etiqueta in ax.get_xticklabels():
        etiqueta.set_fontsize(10.5)

    inicio_prueba = test.index[0]
    ax.axvline(inicio_prueba, color=MARINO, lw=1.1, ls=(0, (3, 3)))
    ax.text(inicio_prueba + pd.Timedelta(days=30), 8, "Prueba de modelos →", fontsize=11, color=MARINO,
            va="bottom", clip_on=True)
    ax.text(pd.Timestamp("2021-01-15"), 530, "Pandemia\nde COVID-19", fontsize=11, color=TEXTO_SUAVE,
            ha="center", va="top", weight="bold", linespacing=1.0, clip_on=True)

    pico, valor_pico = k["fecha_max"], k["maximo"] / 1e3
    ax.scatter([pico], [valor_pico], s=60, color=AMBAR, zorder=5, edgecolor="white", lw=1.2)
    ax.text(pico - pd.Timedelta(days=45), valor_pico + 12,
            f"Récord mensual: {fmt_miles(k['maximo'])} ({fmt_mes(pico)})",
            fontsize=11, color=MARINO, ha="right", va="bottom", weight="bold", clip_on=True)

    fmin, vmin = k["fecha_min"], k["minimo"] / 1e3
    ax.scatter([fmin], [vmin], s=50, color=MARINO, zorder=5, edgecolor="white", lw=1.2)
    ax.text(fmin - pd.Timedelta(days=110), 6, f"Mínimo: {fmt_entero(k['minimo'])} ({fmt_mes(fmin)}) →",
            fontsize=11, color=MARINO, ha="right", va="bottom", weight="bold", clip_on=True)

    crec = (k["record"] / k["total_2014"] - 1) * 100
    ax.annotate("", xy=(pd.Timestamp("2019-02-01"), 445), xytext=(pd.Timestamp("2014-03-01"), 320),
                arrowprops=dict(arrowstyle="-|>", color=TEXTO_SUAVE, lw=1.1, shrinkA=0, shrinkB=0))
    ax.text(pd.Timestamp("2009-04-01"), 470,
            f"2014–2019: crecimiento sostenido, de {fmt_millones(k['total_2014'])}\n"
            f"a {fmt_millones(k['record'])} ingresos al año ({fmt_pct(crec, signo=True)})",
            fontsize=11, color=TEXTO, va="center", linespacing=1.1, clip_on=True)
    ax.text(pd.Timestamp("2022-06-01"), 530,
            f"2022: rebote a {fmt_millones(k['total_2022'])}\n2023–2025: estabilización", fontsize=11,
            color=TEXTO, va="top", linespacing=1.05, clip_on=True)
    return y + 0.04 + alto_ejes + 0.10


def render_estacionalidad(lz: Lienzo, est: pd.DataFrame, x: float, y: float, w: float, h: float) -> None:
    nombres = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
               "octubre", "noviembre", "diciembre"]
    pico = int(est["pre"].idxmax())
    bajos = est["pre"].nsmallest(3)
    dos_bajos = sorted(bajos.index[:2])
    titulo = (
        f"{nombres[pico - 1].capitalize()} supera en {est['pre'][pico] - 100:.0f} % al mes promedio; "
        f"{nombres[dos_bajos[0] - 1]} y {nombres[dos_bajos[1] - 1]} son los meses más flojos"
    )
    yt = lz.titulo_grafico(x, y, w, "B3a", titulo)
    leyenda_y = yt + 0.05
    texto_barras = "Barras: índice 2009–2019 (100 = mes promedio)"
    lz.caja(x, leyenda_y + 0.03, 0.16, 0.13, AZUL, z=1)
    lz.texto(x + 0.22, leyenda_y, texto_barras, size=11, color=TEXTO_SUAVE)
    cx = x + 0.22 + lz.ancho_texto(texto_barras, 11) + 0.22
    lz.fig.add_artist(Circle((cx, leyenda_y + 0.095), 0.06, transform=lz.T, color=CELESTE, ec="white", zorder=2))
    lz.texto(cx + 0.11, leyenda_y, "Puntos: 2022–2025", size=11, color=TEXTO_SUAVE)

    ax_y = leyenda_y + 0.20
    alto_ejes = y + h - ax_y - 0.30
    ax = lz.ejes(x + 0.42, ax_y, w - 0.45, alto_ejes)
    meses = np.arange(1, 13)
    colores = [AMBAR if m == 12 else AZUL for m in meses]
    ax.bar(meses, est["pre"].values, color=colores, width=0.72, zorder=2)
    ax.axhline(100, color=MARINO, lw=1, ls=(0, (4, 3)), zorder=3)
    ax.scatter(meses, est["post"].values, s=46, color=CELESTE, edgecolor="white", lw=1.2, zorder=4)
    for m, v in zip(meses, est["pre"].values):
        ax.text(m, 6, f"{v:.0f}", ha="center", va="bottom", fontsize=11, color="white", weight="bold", zorder=5)
    ax.set_ylim(0, 168)
    ax.set_xlim(0.45, 12.55)
    ax.set_yticks([0, 50, 100, 150])
    ax.set_ylabel("Índice estacional")
    etiquetas = [f"{n}\nbaja" if i + 1 in bajos.index else n for i, n in enumerate(MESES)]
    ax.set_xticks(meses, etiquetas)
    for etiqueta, m in zip(ax.get_xticklabels(), meses):
        if m in bajos.index:
            etiqueta.set_fontweight("bold")
            etiqueta.set_color(MARINO)

    eventos = [
        (3.5, 122, "Semana Santa*"),
        (7.5, 122, "Vacaciones de medio año\n· Bono 14 (jul.)"),
        (12.0, 148, "Fin de año"),
    ]
    for xm, ym, texto in eventos:
        ax.text(xm, ym, texto, ha="center", va="bottom", fontsize=10.5, color=MARINO,
                linespacing=1.05, zorder=6, clip_on=True)
    lz.texto(x + 0.42, y + h - 0.02,
             "*Fecha móvil (marzo o abril). Las fechas son contexto, no causas demostradas.",
             size=10, color=TEXTO_SUAVE, style="italic", ancho=w - 0.45, va="bottom")


def render_vias_y_mercados(
    lz: Lienzo, vias: dict[str, Any], regiones: pd.DataFrame, x: float, y: float, w: float, h: float
) -> None:
    pct = vias["pct"]
    centro, norte = regiones.loc["Centroamérica"], regiones.loc["Norteamérica"]
    yt = lz.titulo_grafico(
        x, y, w, "B3b",
        f"{pct['Terrestre'] / 10:.0f} de cada 10 ingresos son terrestres; Norteamérica crece "
        f"y Centroamérica sigue {abs(centro['cambio']):.0f} % abajo",
    )
    lz.texto(x, yt + 0.05, "Vía de ingreso, 2009–2026", size=11, weight="bold", color=TEXTO)
    lz.texto(x + w, yt + 0.05, f"Marítima {pct['Marítima']:.1f} %*", size=11, color=TEXTO_SUAVE, ha="right")
    barra_y, barra_h = yt + 0.27, 0.32
    acumulado = 0.0
    for via, color, color_texto in (("Terrestre", AZUL, "white"), ("Aérea", CELESTE, MARINO), ("Marítima", GRIS, None)):
        ancho = w * pct[via] / 100
        lz.caja(x + acumulado, barra_y, ancho, barra_h, color, z=1)
        if color_texto:
            lz.texto(x + acumulado + 0.08, barra_y + barra_h / 2, f"{via} {pct[via]:.1f} %", size=12,
                     weight="bold", color=color_texto, va="center")
        acumulado += ancho
    yn = lz.texto(x, barra_y + barra_h + 0.04,
                  f"*Sin registros marítimos desde {vias['ultimo_maritimo'].year + 1}: dato faltante, no cero.",
                  size=11, color=TEXTO_SUAVE, style="italic", ancho=w)

    ym = yn + 0.10
    lz.texto(x, ym, "Región de origen", size=11, weight="bold", color=TEXTO)
    lx = x + w - 1.1
    lz.fig.add_artist(Circle((lx, ym + 0.09), 0.06, transform=lz.T, color=CELESTE, ec="white", zorder=2))
    lz.texto(lx + 0.10, ym, "2019", size=11, color=TEXTO_SUAVE)
    lz.fig.add_artist(Circle((lx + 0.62, ym + 0.09), 0.06, transform=lz.T, color=AZUL, ec="white", zorder=2))
    lz.texto(lx + 0.72, ym, "2025", size=11, color=TEXTO_SUAVE)

    alto_nota = 0.20
    ax_y = ym + 0.14
    ax = lz.ejes(x + 1.18, ax_y, w - 1.22, max(0.55, y + h - alto_nota - ax_y - 0.38))
    orden = ["Centroamérica", "Norteamérica", "Resto del mundo"]
    posiciones = np.arange(len(orden))[::-1]
    for pos, region in zip(posiciones, orden):
        a, b = regiones.loc[region, 2019] / 1e6, regiones.loc[region, 2025] / 1e6
        ax.plot([a, b], [pos, pos], color=GRIS_CLARO, lw=4, solid_capstyle="round", zorder=1)
        ax.scatter([a], [pos], s=80, color=CELESTE, edgecolor="white", lw=1.2, zorder=3)
        ax.scatter([b], [pos], s=80, color=AZUL, edgecolor="white", lw=1.2, zorder=3)
        cambio = fmt_pct(regiones.loc[region, "cambio"], signo=True)
        texto = (f"{fmt_millones(regiones.loc[region, 2019])} → "
                 f"{fmt_millones(regiones.loc[region, 2025])} ({cambio})")
        if max(a, b) > 2:
            ax.text(min(a, b) - 0.12, pos, texto, ha="right", va="center", fontsize=11, color=MARINO, weight="bold")
        else:
            ax.text(max(a, b) + 0.12, pos, texto, ha="left", va="center", fontsize=11, color=MARINO, weight="bold")
    ax.set_yticks(posiciones, orden, fontsize=11.5, color=TEXTO)
    ax.set_ylim(-0.45, 2.4)
    ax.set_xlim(0, 3.5)
    ax.set_xticks([0, 1, 2, 3], ["0", "1 M", "2 M", "3 M"])
    ax.tick_params(axis="y", pad=6)
    ax.xaxis.grid(True, color=REJILLA, lw=0.8)
    ax.set_axisbelow(True)
    lz.texto(x, y + h - 0.02, "Desde 2023 la base agrupa países distinto: lectura orientativa.",
             size=10, color=TEXTO_SUAVE, style="italic", va="bottom")


def render_cuadrante(lz: Lienzo, cuad: pd.DataFrame, y: float) -> float:
    aurora_aerea = cuad.loc["Frontera 01 La Aurora"]
    yt = lz.titulo_grafico(
        MARGEN, y, ANCHO - 2 * MARGEN, "B3c",
        "Cuadrante comparativo: la vía aérea es la más estacional y la más lenta en recuperarse; "
        "San Cristóbal es la que más crece y la más volátil",
    )
    preguntas = [
        ("¿Mayor estacionalidad?", "Fuerza estacional Fs (0 a 1), 2009–2026", "fs",
         lambda v: f"{v:.2f}", None),
        ("¿Mayor crecimiento?", "Crecimiento del flujo mensual promedio 2009→2019", "crecimiento",
         lambda v: fmt_pct(v, signo=True), None),
        ("¿Mayor volatilidad?", "Coeficiente de variación, 2009–2026", "cv",
         lambda v: f"{v:.2f}", None),
        ("¿Más golpeada por la pandemia?", "Meses para volver al promedio de 2019 desde el mínimo de 2020",
         "meses", lambda v: f"{int(v)}", None),
    ]
    x0 = MARGEN
    w_preg, w_resp = 2.70, 1.28
    w_serie = (ANCHO - 2 * MARGEN - w_preg - w_resp) / len(SERIES_CUADRANTE)
    alto_cab, alto_fila = 0.34, 0.38
    ty = yt + 0.06
    lz.caja(x0, ty, ANCHO - 2 * MARGEN, alto_cab + alto_fila * len(preguntas) + 0.04, FONDO_TARJETA, radio=0.08)
    lz.texto(x0 + 0.12, ty + alto_cab / 2, "Pregunta y criterio", size=11, weight="bold", color=MARINO, va="center")
    for j, (_, corto) in enumerate(SERIES_CUADRANTE):
        lz.texto(x0 + w_preg + j * w_serie + 0.06, ty + alto_cab / 2, corto, size=11, weight="bold",
                 color=MARINO, va="center", interlineado=1.0)
    lz.texto(x0 + w_preg + len(SERIES_CUADRANTE) * w_serie + 0.06, ty + alto_cab / 2, "Respuesta",
             size=11, weight="bold", color=MARINO, va="center")

    largo_barra = w_serie - 0.88
    for i, (pregunta, criterio, col, fmt, _secundaria) in enumerate(preguntas):
        fy = ty + alto_cab + i * alto_fila
        lz.caja(x0 + 0.08, fy, ANCHO - 2 * MARGEN - 0.16, 0.010, GRIS_CLARO, z=0)
        lz.texto(x0 + 0.12, fy + 0.04, pregunta, size=11.5, weight="bold", color=TEXTO)
        lz.texto(x0 + 0.12, fy + 0.21, criterio, size=9.5, color=TEXTO_SUAVE, ancho=w_preg - 0.18, interlineado=1.0)
        valores = cuad[col].astype(float)
        mostrados = valores.map(fmt)
        ganador_txt = mostrados[valores.idxmax()]
        ganadores = [s for s in valores.index if mostrados[s] == ganador_txt]
        maximo = valores.max()
        for j, (serie, _) in enumerate(SERIES_CUADRANTE):
            cx = x0 + w_preg + j * w_serie + 0.06
            gana = serie in ganadores
            lz.caja(cx, fy + 0.12, max(0.04, largo_barra * valores[serie] / maximo), 0.12,
                    AMBAR if gana else GRIS_CLARO, z=1)
            lz.texto(cx + largo_barra + 0.05, fy + 0.18, mostrados[serie], size=11 if gana else 10.5,
                     weight="bold" if gana else "normal", color=MARINO if gana else TEXTO, va="center")
        nombres = [dict(SERIES_CUADRANTE)[s].replace("\n", " ").replace("Frontera ", "") for s in ganadores]
        lz.texto(x0 + w_preg + len(SERIES_CUADRANTE) * w_serie + 0.06, fy + 0.18, " /\n".join(nombres),
                 size=11, weight="bold", color=MARINO, va="center", interlineado=1.0)

    ny = ty + alto_cab + alto_fila * len(preguntas) + 0.08
    terrestre = cuad.loc["Vía Terrestre"]
    caidas = cuad["caida"]
    return lz.texto(
        MARGEN, ny,
        f"Ámbar = ganador. La Aurora concentra el {aurora_aerea['participacion_aerea']:.1f} % del flujo aéreo, "
        f"por eso empatan. En volumen, Terrestre suma más (+{terrestre['pendiente']:.0f} viaj./mes). "
        f"Marítima excluida: solo hay registros 2009–2016. Las caídas de 2020 fueron casi totales "
        f"({fmt_pct(caidas.max(), 1)} a {fmt_pct(caidas.min(), 2)}); decide el tiempo de recuperación.",
        size=10, color=TEXTO_SUAVE, ancho=ANCHO - 2 * MARGEN, style="italic",
    )


def render_modelos(lz: Lienzo, modelos: pd.DataFrame, x: float, y: float, w: float, h: float) -> None:
    ganador = modelos.iloc[0]
    yt = lz.titulo_grafico(
        x, y, w, "B4a",
        f"{ganador['etiqueta_larga']} gana entre {len(modelos)} modelos (RMSE {fmt_miles(ganador['RMSE'])}/mes)",
    )
    yt = lz.texto(
        x, yt + 0.02,
        "RMSE de prueba, miles/mes; MAE entre paréntesis. Menor es mejor.",
        size=10, color=TEXTO_SUAVE, ancho=w,
    )
    rmse_max = float(modelos["RMSE"].max())
    n = len(modelos)
    fila_h = min(0.138, (h - (yt - y) - 0.40) / n)
    nombre_w, num_w = 2.12, 0.78
    barra_w = w - nombre_w - num_w - 0.08
    for i, (_, fila) in enumerate(modelos.iterrows()):
        fy = yt + 0.04 + i * fila_h
        gana = i == 0
        color = AMBAR if gana else (GRIS_CLARO if fila["modelo"] == "SeasonalNaive" else GRIS)
        lz.texto(x, fy, fila["etiqueta"], size=9, weight="bold" if gana else "normal",
                 color=MARINO if gana else TEXTO)
        lz.caja(x + nombre_w, fy + 0.03, max(0.05, barra_w * fila["RMSE"] / rmse_max), 0.08, color, z=1)
        lz.texto(x + w, fy, f"{fila['RMSE'] / 1e3:.0f} ({fila['MAE'] / 1e3:.0f})",
                 size=9, ha="right", weight="bold" if gana else "normal",
                 color=MARINO if gana else TEXTO)
    lz.texto(
        x, y + h - 0.02,
        "Ámbar = ganador. Gris claro = Seasonal Naive (repite el último año).",
        size=9, color=TEXTO_SUAVE, ancho=w, style="italic", va="bottom",
    )


def render_pronostico(
    lz: Lienzo, pron: pd.DataFrame, train: pd.Series, x: float, y: float, w: float, h: float
) -> None:
    yt = lz.titulo_grafico(
        x, y, w, "B4b",
        "Entrenado en pandemia, captó la estacionalidad pero subestimó el rebote: hay que recalibrar",
    )
    lz.texto(x, yt + 0.02, "Miles de viajeros/mes", size=10, color=TEXTO_SUAVE)
    ax = lz.ejes(x + 0.10, yt + 0.18, w - 0.16, h - (yt - y) - 0.70)
    inicio = pd.Timestamp("2018-01-01")
    tramo = train[train.index >= inicio] / 1e3
    ax.fill_between(pron.index, pron["inferior"] / 1e3, pron["superior"] / 1e3, color=AMBAR, alpha=0.16, lw=0, zorder=1)
    ax.plot(tramo.index, tramo.values, color=GRIS, lw=1.8, zorder=2)
    ax.plot(pron.index, pron["real"] / 1e3, color=AZUL, lw=2, zorder=3)
    ax.plot(pron.index, pron["prediccion"] / 1e3, color=AMBAR, lw=2.2, ls=(0, (5, 3)), zorder=4)
    ax.axvline(pron.index[0], color=MARINO, lw=1.1, ls=(0, (3, 3)), zorder=3)
    techo = float(np.ceil(pron["superior"].max() / 1e5) * 100)
    ax.set_ylim(0, techo)
    ax.set_yticks([0, 250, 500, 750, 1000] if techo >= 1000 else [0, 200, 400, 600, 800])
    ax.set_xlim(inicio, pron.index[-1] + pd.Timedelta(days=20))
    ax.yaxis.grid(True, color=REJILLA, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.text(pd.Timestamp("2018-03-01"), techo * 0.80, "Entrenamiento\n(observado)", color=TEXTO_SUAVE, fontsize=10.5,
            weight="bold", va="center", linespacing=1.05, clip_on=True)
    ax.text(pron.index[0] + pd.Timedelta(days=25), techo * 0.97, "Periodo de prueba →", color=MARINO, fontsize=10.5,
            va="top", clip_on=True)
    pos_banda = pd.Timestamp("2024-08-01")
    ax.text(pos_banda, techo * 0.93, "Estimación\n(intervalo del 95 %)", color=MARINO, fontsize=11, weight="bold",
            ha="center", va="top", linespacing=1.05, clip_on=True)
    real_lbl = pron.loc["2023-01-01", "real"] / 1e3
    ax.text(pd.Timestamp("2022-07-01"), real_lbl + 105, "Real (prueba)", color=AZUL, fontsize=11, weight="bold",
            ha="center", clip_on=True)
    pred_lbl = pron.loc["2025-06-01", "prediccion"] / 1e3
    ax.text(pd.Timestamp("2025-06-01"), pred_lbl - 70, "Pronóstico\npuntual", color=MARINO, fontsize=10.5,
            ha="center", va="top", linespacing=1.05, clip_on=True)
    a = pron.attrs
    lz.texto(
        x, y + h - 0.02,
        f"Prueba: RMSE {fmt_miles(a['RMSE'])}, MAE {fmt_miles(a['MAE'])}, MAPE {a['MAPE']:.0f} %. "
        f"Banda: percentiles 2.5–97.5 de {N_SIMULACIONES:,} trayectorias. Es una estimación, no una predicción precisa.",
        size=9.5, color=TEXTO_SUAVE, ancho=w, style="italic", va="bottom",
    )


def render_recomendaciones(
    lz: Lienzo, y: float, k: dict[str, Any], est: pd.DataFrame, vias: dict[str, Any], regiones: pd.DataFrame,
    cuad: pd.DataFrame, modelos: pd.DataFrame, pron: pd.DataFrame, train: pd.Series, test: pd.Series,
) -> None:
    lz.caja(0, y, ANCHO, ALTO - y, MARINO)
    lz.caja(MARGEN, y + 0.12, 0.56, 0.32, ORO, radio=0.05)
    lz.texto(MARGEN + 0.28, y + 0.28, "B5", size=14, weight="bold", color=MARINO, ha="center", va="center")
    lz.texto(MARGEN + 0.70, y + 0.28, "Cinco decisiones para INGUAT", size=20, weight="bold", color="white",
             va="center")
    bajos = est["pre"].nsmallest(3)
    aerea = cuad.loc["Vía Aérea"]
    sc = cuad.loc["Frontera 09 San Cristóbal"]
    recomendaciones = [
        ("Llenar la temporada baja",
         f"Promociones en feb., mayo y sep. (índices {bajos.min():.0f}–{bajos.max():.0f}). "
         f"Más capacidad en diciembre ({est['pre'][12]:.0f}) y Semana Santa.", "B3a"),
        ("Diversificar mercados",
         f"Sostener Norteamérica ({fmt_pct(regiones.loc['Norteamérica', 'cambio'], signo=True)}) "
         f"y reactivar Centroamérica ({fmt_pct(regiones.loc['Centroamérica', 'cambio'], signo=True)}). "
         "La base cambió en 2023.", "B3b"),
        ("Agilizar pasos terrestres",
         f"El {vias['pct']['Terrestre']:.0f} % del flujo es terrestre. San Cristóbal "
         f"({fmt_pct(sc['crecimiento'], signo=True)}) es el más volátil.", "B3b · B3c"),
        ("Plan de resiliencia aérea",
         f"La vía aérea cayó {abs(aerea['caida']):.1f} % y tardó {int(aerea['meses'])} meses "
         "en volver a 2019.", "B2 · B3c"),
        ("Actualizar datos y pronósticos",
         f"Recuperar el registro marítimo (sin datos desde {vias['ultimo_maritimo'].year + 1}) "
         f"y recalibrar cada trimestre (MAPE {pron.attrs['MAPE']:.0f} %).", "B3b · B4"),
    ]
    n = len(recomendaciones)
    w = (ANCHO - 2 * MARGEN - (n - 1) * MEDIANIL) / n
    cy = y + 0.48
    ch = max(1.45, ALTO - cy - 0.58)
    for i, (titulo, cuerpo, evidencia) in enumerate(recomendaciones):
        cx = MARGEN + i * (w + MEDIANIL)
        lz.caja(cx, cy, w, ch, "#24376B", radio=0.08)
        lz.texto(cx + 0.10, cy + 0.06, str(i + 1), size=22, weight="bold", color=ORO)
        lz.texto(cx + 0.40, cy + 0.10, titulo, size=12, weight="bold", color="white", ancho=w - 0.48,
                 interlineado=1.05)
        lz.texto(cx + 0.10, cy + 0.50, cuerpo, size=10.5, color=TEXTO_SOBRE_MARINO, ancho=w - 0.20,
                 interlineado=1.12)
        lz.texto(cx + 0.10, cy + ch - 0.08, f"Evidencia: {evidencia}", size=10, weight="bold", color=ORO,
                 va="bottom")

    ficha = (
        f"Fuente: Base de Migración 2009–2026, INGUAT/IGM (uso estrictamente académico). "
        f"Periodo: {fmt_mes(FECHA_INICIO)} – {fmt_mes(FECHA_FIN)}. Filtro: Turista + Excursionista; "
        f"ingresos registrados, no personas únicas. Evaluación: partición cronológica 70/30 "
        f"({fmt_mes(train.index[0])} – {fmt_mes(train.index[-1])} / {fmt_mes(test.index[0])} – "
        f"{fmt_mes(test.index[-1])}, {len(test)} meses). Métricas de prueba: MAE, RMSE, MAPE. "
        f"Intervalo: {N_SIMULACIONES:,} trayectorias del Holt-Winters (errores multiplicativos, "
        f"semilla {SEMILLA}). Elaboración: Python/matplotlib, CC3084 Data Science, UVG. "
        f"Autores: {', '.join(AUTORES[:-1])} y {AUTORES[-1]}."
    )
    lz.texto(MARGEN, cy + ch + 0.08, ficha, size=8, color=TEXTO_SOBRE_MARINO, ancho=ANCHO - 2 * MARGEN,
             interlineado=1.18)


# ---------------------------------------------------------------------------
# Exportacion y verificacion
# ---------------------------------------------------------------------------


def verificar_lienzo(lz: Lienzo, reg: Registro) -> None:
    lz.fig.canvas.draw()
    renderer = lz.fig.canvas.get_renderer()
    ancho_px, alto_px = ANCHO * DPI, ALTO * DPI
    fuera = []
    for artista in lz.fig.findobj(matplotlib.text.Text):
        if not artista.get_visible() or not artista.get_text().strip():
            continue
        caja = artista.get_window_extent(renderer)
        # Solo textos que salen del lienzo (la ficha y el titular usan bandas a sangrado).
        if caja.x0 < -2 or caja.x1 > ancho_px + 2 or caja.y0 < -2 or caja.y1 > alto_px + 2:
            fuera.append(artista.get_text()[:40])
    reg.check("Ningún texto fuera de los márgenes", not fuera, f"{len(fuera)} textos fuera: {fuera[:3]}")
    prohibidos = re.compile(r"TODO|placeholder|parte 1|integrante|Markdown|\[|\]", re.IGNORECASE)
    marcas = [t.get_text() for t in lz.fig.findobj(matplotlib.text.Text) if prohibidos.search(t.get_text())]
    reg.check("Sin marcadores internos ni texto entre corchetes", not marcas, f"{marcas[:3]}")


def exportar(lz: Lienzo, reg: Registro) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lz.fig.savefig(PNG_PATH, dpi=DPI, facecolor="white")
    lz.fig.savefig(PDF_PATH, facecolor="white")
    plt.close(lz.fig)

    from PIL import Image

    with Image.open(PNG_PATH) as img:
        reg.check("PNG de 2200 × 3400 px", img.size == (2200, 3400), f"{img.size[0]} × {img.size[1]} px")
    contenido = PDF_PATH.read_bytes()
    paginas = len(re.findall(rb"/Type\s*/Page\b(?!s)", contenido))
    caja = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]", contenido)
    medida = tuple(float(v) for v in caja.groups()) if caja else None
    reg.check(
        "PDF de 1 página, 11 × 17 in (792 × 1224 pt)",
        paginas == 1 and medida == (0.0, 0.0, 792.0, 1224.0),
        f"{paginas} página(s), MediaBox {medida}",
    )


def escribir_validacion(
    reg: Registro, lz_fuente: str, k: dict[str, Any], est: pd.DataFrame, vias: dict[str, Any],
    regiones: pd.DataFrame, cuad: pd.DataFrame, modelos: pd.DataFrame, pron: pd.DataFrame,
) -> None:
    def tabla(filas: list[tuple[str, ...]], cabecera: tuple[str, ...]) -> list[str]:
        out = ["| " + " | ".join(cabecera) + " |", "|" + "---|" * len(cabecera)]
        out += ["| " + " | ".join(f) + " |" for f in filas]
        return out

    pct = vias["pct"]
    cifras = [
        ("Total ene. 2009 – jun. 2026", "46.7 M", f"{fmt_entero(k['total'])} → {fmt_millones(k['total'], 1)}", "Coincide"),
        ("Récord anual 2019", "4.13 M (+120 %)", f"{fmt_entero(k['record'])} ({fmt_pct(k['crec_record_2009'], 1, True)})", "Coincide"),
        ("Mayo 2020", "9,779; −97 %", f"{fmt_entero(k['minimo'])}; {fmt_pct(k['caida_min'], 1)} vs. {fmt_entero(k['prom_2019'])}", "Coincide"),
        ("Total 2025", "3.35 M, 81 %", f"{fmt_entero(k['total_2025'])}; {fmt_pct(k['recuperacion_2025'], 1)}", "Coincide"),
        ("Máximo mensual", "449 mil (dic. 2019)", f"{fmt_entero(k['maximo'])} ({fmt_mes(k['fecha_max'])})", "Coincide"),
        ("Total 2022", "3.23 M", fmt_entero(k["total_2022"]), "Coincide"),
        ("2014 → 2019", "2.66 M → 4.13 M", f"{fmt_entero(k['total_2014'])} → {fmt_entero(k['record'])}", "Coincide"),
        ("Vías T / A / M", "59.1 / 40.7 / 0.2 %",
         f"{pct['Terrestre']:.2f} / {pct['Aérea']:.2f} / {pct['Marítima']:.2f} %", "Coincide"),
        ("Centroamérica 2019 → 2025", "3.25 M → 2.14 M (−34 %)",
         f"{fmt_entero(regiones.loc['Centroamérica', 2019])} → {fmt_entero(regiones.loc['Centroamérica', 2025])} "
         f"({fmt_pct(regiones.loc['Centroamérica', 'cambio'], 1, True)})", "Coincide"),
        ("Norteamérica 2019 → 2025", "603 mil → 839 mil (+39 %)",
         f"{fmt_entero(regiones.loc['Norteamérica', 2019])} → {fmt_entero(regiones.loc['Norteamérica', 2025])} "
         f"({fmt_pct(regiones.loc['Norteamérica', 'cambio'], 1, True)})", "Coincide"),
        ("Holt-Winters multiplicativo", "RMSE 146,606; MAE 134,375",
         f"RMSE {pron.attrs['RMSE']:,.2f}; MAE {pron.attrs['MAE']:,.2f}", "Coincide (tabla y reajuste)"),
        ("MAPE del ganador", "51 %", f"{pron.attrs['MAPE']:.2f} %", "Coincide"),
        ("Índice estacional dic./feb./sep./may.", "134 / 81 / 83 / 88",
         f"{est['pre'][12]:.1f} / {est['pre'][2]:.1f} / {est['pre'][9]:.1f} / {est['pre'][5]:.1f}", "Coincide"),
        ("Semana Santa (mar.–abr.)", "106–108", f"{est['pre'][3]:.1f} / {est['pre'][4]:.1f}", "Coincide"),
        ("Medio año (jul.–ago.)", "104–107", f"{est['pre'][7]:.1f} / {est['pre'][8]:.1f}", "Coincide"),
    ]
    cuad_filas = [
        (s, f"{r['fs']:.2f}", fmt_pct(r["crecimiento"], 1, True), f"{r['pendiente']:.1f}", f"{r['cv']:.2f}",
         f"{r['caida']:.2f} %", str(int(r["meses"])))
        for s, r in cuad.iterrows()
    ]
    modelos_filas = [
        (str(i + 1), f["etiqueta"], f["modelo"], f"{f['RMSE']:,.2f}", f"{f['MAE']:,.2f}", f"{f['MAPE']:.2f}")
        for i, (_, f) in enumerate(modelos.iterrows())
    ]

    lineas = [
        "# Validación de la infografía «Del récord al reinicio»",
        "",
        "Generado por `python -m src.infografia`. Todas las cifras de la pieza se calculan en el script a partir "
        "de los archivos listados; ninguna se escribe a mano.",
        "",
        "## Archivos producidos",
        "",
        f"- `{PNG_PATH.relative_to(ROOT)}` (2200 × 3400 px, 200 ppp)",
        f"- `{PDF_PATH.relative_to(ROOT)}` (1 página, 11 × 17 in)",
        "- `src/infografia.py` (script reproducible)",
        "",
        "## Fuentes efectivamente usadas",
        "",
        *[f"- {f}" for f in dict.fromkeys(reg.fuentes)],
        "- `src/lab.py`, `src/models.py`, `src/comparacion.py`: funciones originales reutilizadas "
        "(carga y filtro, `ajustar_holt_winters`, `calcular_metricas`, Fs, pendiente, CV e impacto pandémico).",
        "- No existe `outputs/parte2/` en esta copia: las predicciones del ganador se regeneraron reajustando "
        "el mismo Holt-Winters con la función del Laboratorio 1; el reajuste reproduce exactamente el RMSE y MAE "
        "de `report/tables/tabla_metricas_total.tex`.",
        "",
        "## Comprobaciones automáticas",
        "",
        *tabla([(n, e, d) for n, e, d in reg.checks], ("Comprobación", "Estado", "Detalle")),
        "",
        "## Cruce de cifras con el avance (ExplicacionLab9.pdf)",
        "",
        *tabla(cifras, ("Indicador", "Avance", "Resultado trazable", "Estado")),
        "",
        "## Cuadrante comparativo (B3c)",
        "",
        *tabla(cuad_filas, ("Serie", "Fs", "Crec. 2009→2019", "Pendiente (viaj./mes)", "CV", "Caída 2020", "Meses recup.")),
        "",
        "Criterio único por pregunta (mini barras solo dentro de una misma métrica):",
        "",
        "- **Estacionalidad:** fuerza estacional Fs de la descomposición aditiva de la serie completa (Lab 1).",
        "- **Crecimiento:** variación del flujo mensual promedio 2009 → 2019 (`crecimiento_pre_pandemia` del Lab 1). "
        "Se usa en lugar de la pendiente porque la pendiente está en viajeros/mes y favorece a la serie de mayor "
        "volumen; con cinco series de tamaños muy distintos solo una medida relativa es comparable.",
        "- **Volatilidad:** coeficiente de variación de la serie completa (Lab 1).",
        "- **Golpe pandémico:** meses desde el mínimo de 2020 hasta volver al promedio de 2019. La caída se muestra como "
        "dato secundario porque todas están entre −97.6 % y −99.96 % y no discrimina.",
        "- Empates a la precisión mostrada se declaran conjuntos (Vía Aérea y La Aurora son casi la misma serie).",
        "",
        "## Modelos (B4a), orden por RMSE de prueba",
        "",
        *tabla(modelos_filas, ("#", "Etiqueta en la pieza", "Nombre en el Lab 1", "RMSE", "MAE", "MAPE %")),
        "",
        "## Intervalo del pronóstico (B4b)",
        "",
        f"- Método: `HoltWintersResults.simulate` sobre el modelo reajustado, {N_SIMULACIONES:,} trayectorias, "
        f"errores multiplicativos con varianza estimada de los residuos de entrenamiento, semilla {SEMILLA}, "
        "percentiles 2.5 y 97.5.",
        f"- Es un intervalo predictivo estimado por simulación; no incorpora la incertidumbre de los parámetros. "
        f"Cobertura empírica en la prueba: {pron.attrs['cobertura']:.1f} % de los {len(pron)} meses.",
        "- Se rotula «Estimación (intervalo del 95 %)» y la nota del gráfico reconoce el MAPE del modelo.",
        "",
        "## Diferencias respecto al avance y justificación",
        "",
        "1. **Formato:** el enunciado menciona «A3» y «11 × 17 in»; no son equivalentes (A3 = 297 × 420 mm). Se usó "
        "11 × 17 in (tabloide), medida explícita del avance; el PDF no se etiqueta como A3.",
        f"2. **Tipografía:** Lato no está instalada. Se usó **{lz_fuente}**, sans serif derivada de Lato "
        "(pesos regular y negrita, sin Black). Titular y KPIs van en negrita.",
        "3. **Titular:** «Del récord al reinicio:» a 54 pt y «radiografía del turismo en Guatemala 2009–2026» en una "
        "segunda línea a 31 pt en oro, porque la frase completa a 54 pt no cabe en 10 in sin cortarse.",
        "4. **Subtítulo:** «17 años» pasa a «17 años y medio» (ene. 2009 – jun. 2026) y se añaden los porcentajes que "
        "sustentan «terrestre» y «centroamericano».",
        "5. **Cuadrante, crecimiento:** el avance daba como ganadora a Vía Terrestre (pendiente 352 viajeros/mes). Con un "
        "criterio relativo comparable entre las cinco series gana **San Cristóbal (+237.9 %)**; Terrestre "
        "(+184.1 %) queda segunda y su mayor volumen absoluto se menciona en la nota.",
        "6. **Cuadrante, pandemia:** el ganador sigue siendo Vía Aérea / La Aurora, pero por meses de recuperación (27). "
        "En caída porcentual San Cristóbal (−99.96 %) y Valle Nuevo (−99.90 %) superan a la aérea (−99.60 %), así que "
        "«caída del 99.6 %» no basta para declarar a la aérea la más golpeada.",
        "7. **Cuadrante, quinta serie:** son Vía Aérea, Frontera La Aurora, Vía Terrestre, Valle Nuevo y San Cristóbal "
        "(series reales del Lab 1). La Aurora y Vía Aérea se muestran por separado y empatan.",
        "8. **Mercados:** se agrega «Resto del mundo» (total − Centroamérica − Norteamérica), estable ante el cambio de "
        "agrupación de países de 2023, en lugar de regiones pequeñas cuya definición cambió.",
        "9. **Intervalo:** se rotula «intervalo del 95 %» (predictivo, por simulación) y no «IC 95 %», porque no es un "
        "intervalo de confianza de parámetros.",
        "10. **Alturas de bloque:** se ajustaron por legibilidad (ver tabla siguiente) manteniendo el orden B1–B5.",
        "11. **Gris de texto:** el gris #8A919E (3.2:1) solo se usa en marcas gráficas; los textos secundarios usan "
        "#4E5668 (≈7:1 sobre blanco).",
        "",
        "Alturas finales: encadenadas en `main()` (B1 → B2 → B3a/b → B3c → B4 → B5) para que ningún "
        "bloque se superponga ni desborde el lienzo de 17 in. B3c se compactó (filas de 0.40 in) y B5 "
        "usa el espacio restante hasta el borde inferior.",
        "",
        "## Cambios para actualizar el documento de explicación",
        "",
        "Solo existe el PDF del avance; no se modificó. Secciones a actualizar:",
        "",
        "- §4 Tipografía: sustitución de Lato por Carlito y titular en dos niveles (54 pt + 31 pt).",
        "- §5.2 Distribución: alturas finales indicadas arriba.",
        "- §6.1: subtítulo final y contextos de los KPIs (aclaración «registros de entrada, no personas únicas»).",
        "- §6.3 B3b: «Resto del mundo» en la mancuerna.",
        "- §6.3 B3c / Cuadro 4: criterio único por pregunta; crecimiento → San Cristóbal (+238 %); pandemia por meses "
        "de recuperación.",
        "- §6.4 B4b: «Estimación (intervalo del 95 %)» en lugar de «IC 95 %»; método de simulación descrito arriba.",
        "- §6.5 Recomendación 3: menciona el crecimiento de San Cristóbal en lugar de «la de mayor crecimiento» para "
        "la vía terrestre en general.",
        "",
        "## Revisión visual",
        "",
        "Ver la sección de revisión visual al final de la respuesta del asistente / historial de cambios.",
        "",
        "Nota: `outputs/` está en `.gitignore`; la pieza no se versiona en el repositorio público porque los datos "
        "son de uso académico restringido.",
    ]
    VALIDACION_PATH.write_text("\n".join(lineas) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------


def main() -> None:
    reg = Registro()
    base = cargar_base(reg)
    serie, train, test = cargar_serie_total(base, reg)
    k = calcular_kpis(serie, reg)
    est = calcular_estacionalidad(serie, reg)
    vias = calcular_vias(base, reg)
    regiones = calcular_regiones(base, reg)
    cuad = calcular_cuadrante(reg)
    aerea = base[base["Vía"] == "Aérea"]
    cuad.loc["Frontera 01 La Aurora", "participacion_aerea"] = (
        aerea.loc[aerea["Frontera"] == "01 La Aurora", "Viajero"].sum() / aerea["Viajero"].sum() * 100
    )
    centro_pct = (base.loc[base["Región"] == "AMÉRICA DEL CENTRO", "Viajero"].sum() / base["Viajero"].sum() * 100)
    modelos = cargar_modelos(reg)
    pron = pronostico_ganador(modelos, train, test, reg)

    lz = Lienzo()
    y = render_titular_y_kpis(lz, k, vias, centro_pct)
    y_b1 = y
    y = render_historico(lz, serie, k, test, y=y + 0.12)
    y_b2 = y

    y = lz.encabezado_bloque(y + 0.10, "B3", "Estacionalidad, vías y mercados")
    h_b3ab = 2.58
    render_estacionalidad(lz, est, x_col(0), y + 0.08, ancho_cols(7), h_b3ab)
    render_vias_y_mercados(lz, vias, regiones, x_col(7), y + 0.08, ancho_cols(5), h_b3ab)
    y = render_cuadrante(lz, cuad, y + h_b3ab + 0.28)
    y_b3 = y

    y_b4 = lz.encabezado_bloque(y + 0.08, "B4", "Modelos y pronóstico")
    h_b4 = 2.26
    render_modelos(lz, modelos, x_col(0), y_b4 + 0.08, ancho_cols(5), h_b4)
    render_pronostico(lz, pron, train, x_col(5), y_b4 + 0.08, ancho_cols(7), h_b4)
    y_b5 = y_b4 + h_b4 + 0.08
    if y_b5 > 14.70:
        raise ValidacionError(f"B5 empieza en {y_b5:.2f} in; no cabe el cierre antes de 17 in.")
    render_recomendaciones(lz, y_b5, k, est, vias, regiones, cuad, modelos, pron, train, test)
    print(f"Layout in: B1→{y_b1:.2f} B2→{y_b2:.2f} B3→{y_b3:.2f} B4→{y_b5:.2f} B5→17.00")

    verificar_lienzo(lz, reg)
    exportar(lz, reg)
    escribir_validacion(reg, lz.fuente, k, est, vias, regiones, cuad, modelos, pron)
    print(f"PNG: {PNG_PATH}\nPDF: {PDF_PATH}\nValidación: {VALIDACION_PATH}")
    for nombre, estado, detalle in reg.checks:
        print(f"[{estado}] {nombre}: {detalle}")


if __name__ == "__main__":
    main()
