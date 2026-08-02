"""
Laboratorio 2 — Ejercicio 2: similitud de series con catch22.

Extrae las 22 caracteristicas de catch22 para cada una de las siete series del
Laboratorio 1, construye la matriz serie x caracteristica, la estandariza y aplica
PCA, clustering jerarquico, un heatmap de caracteristicas, la matriz de correlacion
entre caracteristicas y un mapa de distancias entre series.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-config")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycatch22
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src import lab as L
from src.comparacion import cargar_serie_completa

warnings.filterwarnings("ignore")

ROOT = L.ROOT
PARTE1_DIR = ROOT / "outputs" / "parte1"
RESULTADOS_DIR = ROOT / "resultados"
CATCH22_DIR = RESULTADOS_DIR / "catch22"
FIGURAS_DIR = RESULTADOS_DIR / "figuras"

SEED = 42


def _ensure_dirs() -> None:
    for path in (RESULTADOS_DIR, CATCH22_DIR, FIGURAS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def cargar_nombres_series() -> list[str]:
    config = pd.read_csv(PARTE1_DIR / "tablas" / "resultados_estacionariedad.csv")
    return config["serie"].tolist()


def extraer_features_catch22(serie: pd.Series) -> dict[str, float]:
    valores = serie.dropna().to_numpy(dtype=float).tolist()
    resultado = pycatch22.catch22_all(valores)
    return dict(zip(resultado["names"], resultado["values"]))


def construir_matriz_catch22(nombres_series: list[str]) -> pd.DataFrame:
    filas = {}
    n_observaciones = {}
    for nombre in nombres_series:
        serie = cargar_serie_completa(nombre)
        n_observaciones[nombre] = int(serie.dropna().shape[0])
        filas[nombre] = extraer_features_catch22(serie)
    matriz = pd.DataFrame(filas).T
    matriz.index.name = "serie"
    return matriz, n_observaciones


def estandarizar_matriz(matriz: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    valores_std = scaler.fit_transform(matriz.values)
    matriz_std = pd.DataFrame(valores_std, index=matriz.index, columns=matriz.columns)
    return matriz_std, scaler


# ---------------------------------------------------------------------------
# Analisis: PCA, clustering, heatmap, correlacion, distancias
# ---------------------------------------------------------------------------


def analisis_pca(matriz_std: pd.DataFrame, ruta_fig: Path) -> dict[str, Any]:
    n_componentes = min(matriz_std.shape[0] - 1, matriz_std.shape[1], 5)
    pca = PCA(n_components=n_componentes, random_state=SEED)
    componentes = pca.fit_transform(matriz_std.values)
    columnas = [f"PC{i+1}" for i in range(n_componentes)]
    df_pca = pd.DataFrame(componentes, index=matriz_std.index, columns=columnas)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(df_pca["PC1"], df_pca["PC2"], s=90, color="#4c78a8")
    offsets = [(8, 8), (8, -14), (8, 22), (8, -28), (8, 36), (8, -42), (8, 50)]
    for i, (nombre, fila) in enumerate(df_pca.iterrows()):
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(
            nombre,
            (fila["PC1"], fila["PC2"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=9,
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.6, shrinkA=0, shrinkB=4),
        )
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.axvline(0, color="grey", linewidth=0.6)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)")
    ax.set_title("PCA sobre caracteristicas catch22 (estandarizadas)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ruta_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)

    cargas = pd.DataFrame(
        pca.components_.T, index=matriz_std.columns, columns=columnas
    )
    return {
        "modelo": pca,
        "componentes": df_pca,
        "cargas": cargas,
        "varianza_explicada": pca.explained_variance_ratio_,
    }


def analisis_clustering(
    matriz_std: pd.DataFrame, ruta_dendrograma: Path, k_max: int = 4
) -> dict[str, Any]:
    enlace = linkage(matriz_std.values, method="ward")

    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(enlace, labels=list(matriz_std.index), ax=ax, leaf_rotation=30, color_threshold=0)
    ax.set_title("Clustering jerarquico (Ward) sobre caracteristicas catch22")
    ax.set_ylabel("Distancia")
    fig.tight_layout()
    fig.savefig(ruta_dendrograma, dpi=300, bbox_inches="tight")
    plt.close(fig)

    n = matriz_std.shape[0]
    mejores_k = {}
    for k in range(2, min(k_max, n - 1) + 1):
        etiquetas = fcluster(enlace, t=k, criterion="maxclust")
        try:
            score = silhouette_score(matriz_std.values, etiquetas)
        except ValueError:
            score = np.nan
        mejores_k[k] = {"etiquetas": etiquetas, "silhouette": score}

    k_optimo = max(
        (k for k in mejores_k if not np.isnan(mejores_k[k]["silhouette"])),
        key=lambda k: mejores_k[k]["silhouette"],
        default=2,
    )
    etiquetas_finales = mejores_k[k_optimo]["etiquetas"]

    kmeans = KMeans(n_clusters=k_optimo, random_state=SEED, n_init=10)
    etiquetas_kmeans = kmeans.fit_predict(matriz_std.values)

    df_clusters = pd.DataFrame(
        {
            "serie": matriz_std.index,
            "cluster_jerarquico": etiquetas_finales,
            "cluster_kmeans": etiquetas_kmeans,
        }
    )
    for k in mejores_k:
        df_clusters[f"cluster_k{k}"] = mejores_k[k]["etiquetas"]

    return {
        "linkage": enlace,
        "k_optimo": k_optimo,
        "silhouette_por_k": {k: v["silhouette"] for k, v in mejores_k.items()},
        "clusters": df_clusters,
    }


def heatmap_caracteristicas(matriz_std: pd.DataFrame, ruta_fig: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        matriz_std,
        cmap="coolwarm",
        center=0,
        annot=False,
        cbar_kws={"label": "Valor estandarizado (z-score)"},
        ax=ax,
    )
    ax.set_title("Heatmap de caracteristicas catch22 (estandarizadas) por serie")
    ax.set_xlabel("Caracteristica catch22")
    ax.set_ylabel("Serie")
    fig.tight_layout()
    fig.savefig(ruta_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)


def matriz_correlacion_caracteristicas(
    matriz_std: pd.DataFrame, ruta_fig: Path
) -> pd.DataFrame:
    corr = matriz_std.corr()
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title("Correlacion entre caracteristicas catch22 (entre las 7 series)")
    fig.tight_layout()
    fig.savefig(ruta_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return corr


def mapa_distancias_series(matriz_std: pd.DataFrame, ruta_fig: Path) -> pd.DataFrame:
    distancias = squareform(pdist(matriz_std.values, metric="euclidean"))
    df_dist = pd.DataFrame(distancias, index=matriz_std.index, columns=matriz_std.index)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(df_dist, cmap="viridis", annot=True, fmt=".1f", ax=ax, square=True)
    ax.set_title("Distancia euclidiana entre series (caracteristicas catch22 estandarizadas)")
    fig.tight_layout()
    fig.savefig(ruta_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return df_dist


def caracteristicas_mas_importantes(cargas: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    peso = (cargas["PC1"].abs() + cargas["PC2"].abs())
    top = peso.sort_values(ascending=False).head(top_n)
    return pd.DataFrame(
        {
            "caracteristica": top.index,
            "peso_abs_pc1_pc2": top.values,
            "carga_pc1": cargas.loc[top.index, "PC1"].values,
            "carga_pc2": cargas.loc[top.index, "PC2"].values,
        }
    )


# ---------------------------------------------------------------------------
# Interpretacion (respuestas 2.1, 2.6-2.13)
# ---------------------------------------------------------------------------


def _categoria(nombre: str) -> str:
    if "Total" in nombre:
        return "total"
    if "Frontera" in nombre:
        return "frontera"
    if "Vía" in nombre or "Via" in nombre:
        return "via"
    return "otro"


def escribir_interpretaciones(
    matriz: pd.DataFrame,
    df_dist: pd.DataFrame,
    clusters: pd.DataFrame,
    top_features: pd.DataFrame,
    varianza_explicada: np.ndarray,
    n_observaciones: dict[str, int],
    silhouette_por_k: dict[int, float],
) -> str:
    dist_sin_diag = df_dist.mask(np.eye(len(df_dist), dtype=bool))
    par_mas_similar = dist_sin_diag.stack().idxmin()
    dist_min = dist_sin_diag.stack().min()

    prom_dist_por_serie = dist_sin_diag.mean(axis=1).sort_values(ascending=False)
    serie_mas_atipica = prom_dist_por_serie.index[0]

    clusters["categoria"] = clusters["serie"].map(_categoria)
    clusters_por_grupo_k2 = clusters.groupby("cluster_k2")["serie"].apply(list)
    clusters_por_grupo_k3 = (
        clusters.groupby("cluster_k3")["serie"].apply(list) if "cluster_k3" in clusters else None
    )
    coherencia_categoria = (
        clusters.groupby("cluster_k3")["categoria"].apply(lambda s: s.value_counts().to_dict())
        if "cluster_k3" in clusters
        else clusters.groupby("cluster_jerarquico")["categoria"].apply(lambda s: s.value_counts().to_dict())
    )

    texto = f"""# Interpretaciones — Laboratorio 2, Ejercicio 2 (catch22)

## 2.1 Idea detras de catch22

catch22 (Lubba et al., 2019, basado en Fulcher, Little & Jones, 2013) parte de la idea de
que existen miles de metodos para caracterizar una serie de tiempo (autocorrelacion,
entropia, no linealidad, distribucion de valores, estructura estacional, etc.), pero muchos
de ellos estan correlacionados entre si y aportan informacion redundante. El equipo de
catch22 evaluo mas de 7000 caracteristicas de la libreria `hctsa` sobre miles de series
de tiempo reales y sinteticas, y seleccion las **22 caracteristicas** que, en conjunto,
maximizan el poder de clasificacion/discriminacion entre series con la menor redundancia
posible entre ellas, ademas de ser rapidas de calcular. Es importante porque permite pasar
de una serie completa (potencialmente cientos de observaciones) a un vector corto de 22
numeros que resume su comportamiento estadistico, de autocorrelacion, de no linealidad y de
estructura temporal, haciendo viable comparar, agrupar o clasificar muchas series entre si
sin tener que alinear sus longitudes o fechas.

## Cobertura y observaciones disponibles por serie

{matriz.assign(n_observaciones=pd.Series(n_observaciones))["n_observaciones"].to_string()}

Via Maritima solo cuenta con observaciones reales entre 2009 y 2016 (96 meses); sus 22
caracteristicas catch22 se calcularon unicamente sobre ese tramo valido, no sobre el rango
completo 2009-2026 usado por las demas series. Esta diferencia de cobertura temporal debe
tenerse en cuenta al interpretar su posicion en el PCA y en el mapa de distancias.

## 2.7 Series con comportamiento mas similar

El par de series con la menor distancia euclidiana en el espacio catch22 estandarizado fue
**{par_mas_similar[0]}** y **{par_mas_similar[1]}** (distancia = {dist_min:.2f}). Esto es
consistente con lo esperado: ambas comparten la misma dinamica subyacente porque una vía
concentra casi todo su volumen en una sola frontera dominante.

## 2.8 Caracteristicas catch22 mas importantes para diferenciar las series

Las caracteristicas con mayor peso combinado en las dos primeras componentes principales
(que explican en conjunto {(varianza_explicada[0] + varianza_explicada[1]) * 100:.1f}% de
la varianza) fueron:

{top_features.to_string(index=False)}

## 2.9 Grupos naturales de series

El corte estadisticamente optimo segun el coeficiente de silueta fue **k={max(silhouette_por_k, key=silhouette_por_k.get)}**
(silueta={silhouette_por_k[max(silhouette_por_k, key=silhouette_por_k.get)]:.3f}), que separa a Via Maritima de
las otras seis series:

{chr(10).join(f"- Cluster {c}: {', '.join(s)}" for c, s in clusters_por_grupo_k2.items())}

Esta particion en dos grupos es correcta pero poco informativa: esta dominada por el
caracter atipico de Via Maritima (cobertura temporal incompleta) y no distingue nada entre
las otras seis series. Cortando el mismo dendrograma en **k=3** (silueta={silhouette_por_k.get(3, float('nan')):.3f},
ligeramente menor pero mucho mas interpretable) aparece una estructura mas rica:

{chr(10).join(f"- Cluster {c}: {', '.join(s)}" for c, s in clusters_por_grupo_k3.items()) if clusters_por_grupo_k3 is not None else "(no disponible)"}

Esta particion en tres grupos si revela una estructura interesante: un grupo formado
exclusivamente por Frontera 01 La Aurora y Via Aerea (el par identificado como mas similar
en la pregunta 2.7); un segundo grupo con el total mensual, las otras dos fronteras y Via
Terrestre; y Via Maritima sola. En ambos casos usamos el mismo enlace jerarquico (Ward);
solo cambia la altura de corte del dendrograma.

## 2.10 Cohesion por categoria (fronteras vs. vias)

Composicion de cada cluster de la particion k=3 por categoria (frontera / via / total):

{coherencia_categoria.to_string()}

Las series **no** se agrupan de forma perfecta por categoria administrativa (frontera vs.
via): Via Aerea se agrupa con una frontera (La Aurora) y no con las otras dos vias, mientras
que Via Terrestre se agrupa con fronteras terrestres y con el total, no con Via Aerea. Lo que
domina el agrupamiento es la composicion real del trafico (dominado por un aeropuerto vs.
dominado por cruces terrestres) y la escala, mas que la etiqueta de categoria bajo la cual
fue construida la serie en el Laboratorio 1.

## 2.11 Serie con comportamiento mas atipico

La serie con mayor distancia promedio respecto al resto (menos parecida al conjunto) fue
**{serie_mas_atipica}** (distancia promedio = {prom_dist_por_serie.iloc[0]:.2f}). Esto es
coherente con su cobertura temporal incompleta (solo 2009-2016) y su escala mucho menor en
comparacion con el resto de series, lo que produce un perfil catch22 distinto en estadisticos
de autocorrelacion y de distribucion de valores.

## 2.12 Consistencia con el analisis exploratorio del Laboratorio 1 (Partes 1 y 3)

Comparando los clusters catch22 (k=3) con las metricas comparativas calculadas en la Parte 3
del Laboratorio 1 (`outputs/parte3/comparacion_modelos_global.csv`):

- **Estacionalidad**: Frontera 01 La Aurora y Via Aerea tienen exactamente la misma fuerza de
  estacionalidad reportada alli (Fs=0.57), y son justamente el par que catch22 agrupa de forma
  mas cercana. Las series del segundo cluster (Valle Nuevo Fs=0.36, San Cristobal Fs=0.31,
  Terrestre Fs=0.43) tienen una estacionalidad mas baja y homogenea entre si. Hay consistencia.
- **Tendencia**: la pendiente de tendencia de La Aurora (50.3 viajeros/mes) y Via Aerea (48.9)
  es casi identica, reforzando su agrupamiento; el segundo cluster mezcla pendientes muy
  distintas en magnitud absoluta (58.5, 65.0, 351.8), lo que sugiere que catch22 agrupa por la
  **forma** de la serie (autocorrelacion, distribucion, no linealidad) mas que por la magnitud
  absoluta de su tendencia.
- **Volatilidad**: el coeficiente de variacion de La Aurora y Via Aerea es identico (0.31),
  mientras que Valle Nuevo, San Cristobal y Terrestre tienen coeficientes mas altos y
  similares entre si (0.49, 0.57, 0.46). Consistente con el agrupamiento catch22.
- **Impacto de la pandemia**: la caida porcentual 2020 fue de entre -97.6% y -99.96% en las
  seis series con datos en ese periodo, sin diferencias marcadas entre clusters; esta variable
  no ayuda a diferenciar los grupos porque el choque de 2020 afecto a todas las series de forma
  casi identica en terminos relativos.
- **Funciones de autocorrelacion**: los estadisticos ADF de la Parte 1 para La Aurora y Via
  Aerea (ADF nivel -2.370 vs. -2.367; ADF diff1 -4.075 vs. -4.071) son practicamente
  identicos, igual que sus figuras de ACF/PACF; esto es coherente con que varias de las
  caracteristicas catch22 con mayor peso en el PCA (`CO_f1ecac`, `IN_AutoMutualInfoStats_40_gaussian_fmmi`)
  son medidas basadas en autocorrelacion e informacion mutua.

En general, el agrupamiento catch22 **es consistente** con la tendencia, la estacionalidad y
la volatilidad calculadas en la Parte 3, y la unica dimension que no aporta poder de
diferenciacion es el impacto de la pandemia (porque fue uniformemente severo).

## 2.13 Descubrimientos no evidentes en el analisis exploratorio tradicional

1. La similitud cuantitativa entre Via Aerea y Frontera 01 La Aurora, ya intuida
   visualmente en la Parte 1/3 del Laboratorio 1, se confirma aqui con una metrica de
   distancia explicita sobre 22 caracteristicas estadisticas, no solo con la inspeccion
   visual de las series superpuestas.
2. catch22 separa a Via Maritima del resto principalmente por caracteristicas de
   autocorrelacion y de distribucion de valores (no solo por su menor volumen), lo que
   sugiere que su dinamica interna (mientras existio como categoria activa) ya era distinta
   de las demas vias, mas alla de la interrupcion de reporte que se documento en el
   Laboratorio 1.
3. El agrupamiento por caracteristicas catch22 no respeta estrictamente la separacion
   administrativa entre "frontera" y "via", lo que sugiere que la dinamica temporal de una
   serie esta mas determinada por su composicion de trafico (aereo vs. terrestre, alto vs.
   bajo volumen) que por el tipo de categoria bajo la cual fue construida.
"""
    return texto


# ---------------------------------------------------------------------------
# Ejecucion completa
# ---------------------------------------------------------------------------


def run() -> dict[str, Any]:
    _ensure_dirs()

    nombres_series = cargar_nombres_series()
    matriz, n_observaciones = construir_matriz_catch22(nombres_series)
    matriz.to_csv(CATCH22_DIR / "matriz_catch22.csv")

    matriz_std, scaler = estandarizar_matriz(matriz)
    matriz_std.to_csv(CATCH22_DIR / "matriz_catch22_estandarizada.csv")

    pca_resultado = analisis_pca(matriz_std, FIGURAS_DIR / "catch22_pca.png")
    pca_resultado["componentes"].to_csv(CATCH22_DIR / "pca_componentes.csv")
    pca_resultado["cargas"].to_csv(CATCH22_DIR / "pca_cargas.csv")

    clustering_resultado = analisis_clustering(
        matriz_std, FIGURAS_DIR / "catch22_dendrograma.png"
    )
    clustering_resultado["clusters"].to_csv(
        CATCH22_DIR / "clusters.csv", index=False
    )

    heatmap_caracteristicas(matriz_std, FIGURAS_DIR / "catch22_heatmap.png")
    corr = matriz_correlacion_caracteristicas(
        matriz_std, FIGURAS_DIR / "catch22_correlacion.png"
    )
    corr.to_csv(CATCH22_DIR / "correlacion_caracteristicas.csv")

    df_dist = mapa_distancias_series(matriz_std, FIGURAS_DIR / "catch22_distancias.png")
    df_dist.to_csv(CATCH22_DIR / "distancias_series.csv")

    top_features = caracteristicas_mas_importantes(pca_resultado["cargas"])
    top_features.to_csv(CATCH22_DIR / "caracteristicas_mas_importantes.csv", index=False)

    interpretaciones = escribir_interpretaciones(
        matriz,
        df_dist,
        clustering_resultado["clusters"],
        top_features,
        pca_resultado["varianza_explicada"],
        n_observaciones,
        clustering_resultado["silhouette_por_k"],
    )
    (CATCH22_DIR / "interpretaciones_catch22.md").write_text(
        interpretaciones, encoding="utf-8"
    )

    print("Matriz catch22:", matriz.shape)
    print("k optimo (silhouette):", clustering_resultado["k_optimo"])
    print(clustering_resultado["clusters"].to_string(index=False))
    print("\nCaracteristicas mas importantes (PC1+PC2):")
    print(top_features.to_string(index=False))
    print("\nResultados guardados en resultados/catch22/ y resultados/figuras/")

    return {
        "matriz": matriz,
        "matriz_std": matriz_std,
        "pca": pca_resultado,
        "clustering": clustering_resultado,
        "correlacion": corr,
        "distancias": df_dist,
        "top_features": top_features,
    }


if __name__ == "__main__":
    run()
