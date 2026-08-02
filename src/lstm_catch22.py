"""
Laboratorio 2 — Ejercicio 2.14.

Construye un modelo LSTM hibrido que recibe, ademas de la ventana de rezagos, un vector
estatico con las 22 caracteristicas catch22 de la serie (calculadas solo con el conjunto de
entrenamiento, para no filtrar informacion del conjunto de prueba). Se compara contra el
mejor modelo LSTM puro del Ejercicio 1 para la misma serie (Total mensual).
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-config")

import matplotlib

matplotlib.use("Agg")

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers

from src import lstm_lab as L
from src.catch22_lab import extraer_features_catch22

warnings.filterwarnings("ignore")

ROOT = L.ROOT
RESULTADOS_DIR = L.RESULTADOS_DIR
FIGURAS_DIR = L.FIGURAS_DIR

SEED = L.SEED
SERIE_SLUG = L.SERIE_1_SLUG
NOMBRE_SERIE = L.NOMBRE_SERIE_1


def _ensure_dirs() -> None:
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTADOS_DIR / "catch22").mkdir(parents=True, exist_ok=True)
    L.MODELOS_DIR.mkdir(parents=True, exist_ok=True)


def cargar_mejor_config_lstm(nombre_archivo: str = "configuracion_serie_1.json") -> dict[str, Any]:
    with open(RESULTADOS_DIR / nombre_archivo, encoding="utf-8") as f:
        return json.load(f)


def vector_catch22_train(train: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Caracteristicas catch22 calculadas solo con el conjunto de entrenamiento,
    normalizadas dentro de si mismas (media 0, desviacion 1 sobre sus 22 valores)
    para no filtrar informacion del conjunto de prueba ni depender de otras series."""
    features = extraer_features_catch22(train)
    nombres = list(features.keys())
    valores = np.array(list(features.values()), dtype=float)
    normalizado = (valores - valores.mean()) / (valores.std() + 1e-9)
    return normalizado.astype(np.float32), nombres


def preparar_datos_catch22(
    train: pd.Series, test: pd.Series, look_back: int, vector_catch22: np.ndarray
) -> dict[str, Any]:
    datos = L.preparar_datos(train, test, look_back, porcentaje_validacion=0.20)
    for clave, x_clave in (("train", "X_train"), ("val", "X_val"), ("test", "X_test"), ("completo", "X_completo")):
        n = datos[x_clave].shape[0]
        datos[f"catch22_{clave}"] = np.tile(vector_catch22, (n, 1)).astype(np.float32)
    return datos


def construir_modelo_lstm_catch22(
    look_back: int,
    n_features_catch22: int,
    unidades_lstm: int,
    unidades_dense_catch22: int,
    dropout: float,
    learning_rate: float,
) -> keras.Model:
    entrada_secuencia = keras.Input(shape=(look_back, 1), name="secuencia")
    entrada_catch22 = keras.Input(shape=(n_features_catch22,), name="catch22")

    x_seq = layers.LSTM(unidades_lstm)(entrada_secuencia)
    x_seq = layers.Dropout(dropout)(x_seq)

    x_cat = layers.Dense(unidades_dense_catch22, activation="relu")(entrada_catch22)

    concatenado = layers.Concatenate()([x_seq, x_cat])
    salida = layers.Dense(1)(concatenado)

    modelo = keras.Model(
        inputs=[entrada_secuencia, entrada_catch22], outputs=salida, name="lstm_catch22"
    )
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return modelo


def entrenar_lstm_catch22(
    train: pd.Series, test: pd.Series, config: dict[str, Any], vector_catch22: np.ndarray
) -> tuple[keras.Model, dict, dict, pd.DataFrame]:
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)

    look_back = int(config["look_back"])
    datos = preparar_datos_catch22(train, test, look_back, vector_catch22)

    modelo = construir_modelo_lstm_catch22(
        look_back=look_back,
        n_features_catch22=len(vector_catch22),
        unidades_lstm=int(config["unidades_1"]),
        unidades_dense_catch22=8,
        dropout=float(config["dropout"]),
        learning_rate=float(config["learning_rate"]),
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, min_delta=1e-5, restore_best_weights=True
    )
    history = modelo.fit(
        [datos["X_train"], datos["catch22_train"]],
        datos["y_train"],
        validation_data=([datos["X_val"], datos["catch22_val"]], datos["y_val"]),
        epochs=300,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        callbacks=[early_stopping],
        verbose=0,
    )
    mejor_epoca = int(np.argmin(history.history["val_loss"]) + 1)

    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)
    modelo_final = construir_modelo_lstm_catch22(
        look_back=look_back,
        n_features_catch22=len(vector_catch22),
        unidades_lstm=int(config["unidades_1"]),
        unidades_dense_catch22=8,
        dropout=float(config["dropout"]),
        learning_rate=float(config["learning_rate"]),
    )
    historial_final = modelo_final.fit(
        [datos["X_completo"], datos["catch22_completo"]],
        datos["y_completo"],
        epochs=max(1, mejor_epoca),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        verbose=0,
    )

    pred_test_scaled = modelo_final.predict([datos["X_test"], datos["catch22_test"]], verbose=0)
    pred_test = datos["scaler"].inverse_transform(pred_test_scaled).reshape(-1)
    y_test_real = test.to_numpy(dtype=float).reshape(-1)
    metricas_test = L.calcular_metricas(y_test_real, pred_test)

    predicciones = pd.DataFrame(
        {"Real": y_test_real, "Prediccion_LSTM_catch22": pred_test}, index=test.index
    )
    return modelo_final, historial_final.history, metricas_test, predicciones


def graficar_comparacion(
    predicciones_plain: pd.DataFrame,
    predicciones_catch22: pd.DataFrame,
    nombre_serie: str,
    ruta: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(predicciones_plain.index, predicciones_plain["Real"], marker="o", color="black", label="Valor real")
    ax.plot(
        predicciones_plain.index,
        predicciones_plain["Prediccion_LSTM"],
        marker="o",
        linestyle="--",
        label="LSTM (Ejercicio 1)",
    )
    ax.plot(
        predicciones_catch22.index,
        predicciones_catch22["Prediccion_LSTM_catch22"],
        marker="o",
        linestyle="--",
        label="LSTM + catch22 (Ejercicio 2.14)",
    )
    ax.set_title(f"LSTM vs. LSTM + catch22 — {nombre_serie}")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Valor")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def escribir_discusion(
    config: dict[str, Any], metricas_plain: dict, metricas_catch22: dict, nombre_serie: str
) -> str:
    mejor = "LSTM + catch22" if metricas_catch22["RMSE"] < metricas_plain["RMSE"] else "LSTM (Ejercicio 1)"
    diferencia_pct = (
        (metricas_catch22["RMSE"] - metricas_plain["RMSE"]) / metricas_plain["RMSE"] * 100
    )
    texto = f"""# Discusion — LSTM + catch22 (Ejercicio 2.14)

## Diseno del experimento

Para poder aislar el efecto de agregar las caracteristicas catch22 (y no confundirlo con un
cambio de arquitectura), el modelo `LSTM + catch22` usa exactamente el mismo backbone LSTM que
el mejor modelo del Ejercicio 1 para **{nombre_serie}** (`{config['modelo']}`: look_back=
{int(config['look_back'])}, {int(config['unidades_1'])} unidades LSTM, dropout=
{config['dropout']}, learning_rate={config['learning_rate']}, batch_size=
{int(config['batch_size'])}). La unica diferencia es una segunda entrada: el vector de 22
caracteristicas catch22, calculado **unicamente con el conjunto de entrenamiento** (para no
filtrar informacion del conjunto de prueba) y normalizado internamente (media 0, desviacion 1
sobre sus propios 22 valores). Ese vector se pasa por una capa densa de 8 unidades y se
concatena con la salida de la LSTM antes de la capa de salida.

## Resultados

| Modelo | MAE prueba | RMSE prueba | MAPE prueba | R2 prueba |
|---|---|---|---|---|
| LSTM (Ejercicio 1) | {metricas_plain['MAE']:.2f} | {metricas_plain['RMSE']:.2f} | {metricas_plain['MAPE']:.2f}% | {metricas_plain['R2']:.3f} |
| LSTM + catch22 (2.14) | {metricas_catch22['MAE']:.2f} | {metricas_catch22['RMSE']:.2f} | {metricas_catch22['MAPE']:.2f}% | {metricas_catch22['R2']:.3f} |

El RMSE de prueba cambio en {diferencia_pct:+.2f}% al agregar catch22. El mejor modelo para
**{nombre_serie}** fue **{mejor}**.

## Discusion

Para una unica serie, el vector catch22 es una constante (no cambia entre observaciones de
la misma serie), por lo que la rama densa que lo procesa solo puede aportar un termino de
sesgo adicional a la red — no aporta informacion temporal nueva dentro de esta serie. Por eso
la diferencia de desempeno frente al LSTM puro se explica principalmente por la capacidad
extra del modelo (una capa densa y una concatenacion adicionales), no por un uso genuino de
las caracteristicas catch22 como señal predictiva.

El valor real de catch22 en un modelo de deep learning no esta en enriquecer una LSTM
entrenada sobre una sola serie, sino en escenarios donde el mismo modelo debe pronosticar
**varias series a la vez** (un modelo global): alli, el vector catch22 si varia de una serie a
otra y puede actuar como un "identificador" o embedding de contexto que le indica al modelo
que tipo de dinamica esta viendo (alta estacionalidad, alta volatilidad, dominada por
aeropuerto, etc.), lo que en principio le permitiria compartir capacidad entre series
similares (por ejemplo, entre Frontera 01 La Aurora y Via Aerea, identificadas como las mas
parecidas en el Ejercicio 2) sin tener que entrenar un modelo independiente para cada una.
Esta aplicacion mas ambiciosa (un unico LSTM global condicionado por catch22 para las siete
series) queda fuera del alcance de este ejercicio, que compara un LSTM contra su version
aumentada sobre una sola serie, pero es la direccion natural para aprovechar mejor estas
caracteristicas.
"""
    return texto


def run() -> dict[str, Any]:
    _ensure_dirs()

    print("Keras:", keras.__version__, "| backend:", keras.backend.backend())

    train, test = L.cargar_serie(SERIE_SLUG)
    train = L.convertir_a_serie(train, "train")
    test = L.convertir_a_serie(test, "test")

    config = cargar_mejor_config_lstm("configuracion_serie_1.json")
    print("Configuracion reutilizada del Ejercicio 1:", config["modelo"])

    predicciones_plain = pd.read_csv(
        RESULTADOS_DIR / "predicciones_serie_1.csv", index_col=0, parse_dates=True
    )
    comparacion_previa = pd.read_csv(RESULTADOS_DIR / "comparacion_mejores_lstm.csv")
    fila_previa = comparacion_previa[comparacion_previa["Serie"] == NOMBRE_SERIE].iloc[0]
    metricas_plain = {
        "MAE": float(fila_previa["MAE prueba"]),
        "RMSE": float(fila_previa["RMSE prueba"]),
        "MAPE": float(fila_previa["MAPE prueba"]),
        "R2": float(fila_previa["R2 prueba"]),
    }

    vector_catch22, nombres_features = vector_catch22_train(train)
    pd.Series(vector_catch22, index=nombres_features).to_csv(
        RESULTADOS_DIR / "catch22" / "vector_catch22_train_serie_1.csv"
    )

    modelo, historial, metricas_catch22, predicciones_catch22 = entrenar_lstm_catch22(
        train, test, config, vector_catch22
    )

    predicciones_catch22.to_csv(RESULTADOS_DIR / "predicciones_lstm_catch22_serie_1.csv")

    comparacion = pd.DataFrame(
        [
            {"Modelo": "LSTM (Ejercicio 1)", **metricas_plain},
            {"Modelo": "LSTM + catch22 (2.14)", **metricas_catch22},
        ]
    )
    comparacion.insert(0, "Serie", NOMBRE_SERIE)
    comparacion.to_csv(RESULTADOS_DIR / "comparacion_lstm_catch22.csv", index=False)

    graficar_comparacion(
        predicciones_plain, predicciones_catch22, NOMBRE_SERIE, FIGURAS_DIR / "lstm_vs_lstm_catch22.png"
    )

    modelo.save(L.MODELOS_DIR / "lstm_catch22_serie_1.keras")

    discusion = escribir_discusion(config, metricas_plain, metricas_catch22, NOMBRE_SERIE)
    (RESULTADOS_DIR / "catch22" / "discusion_lstm_catch22.md").write_text(
        discusion, encoding="utf-8"
    )

    print("\nComparacion LSTM vs LSTM+catch22:")
    print(comparacion.to_string(index=False))
    print("\nGuardado en resultados/ y resultados/catch22/")

    return {
        "config": config,
        "metricas_plain": metricas_plain,
        "metricas_catch22": metricas_catch22,
        "comparacion": comparacion,
        "predicciones_catch22": predicciones_catch22,
    }


if __name__ == "__main__":
    run()
