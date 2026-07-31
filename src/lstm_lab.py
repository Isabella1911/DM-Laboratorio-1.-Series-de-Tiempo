"""
Laboratorio 2 — LSTM para series de tiempo.

Usa Keras 3 con backend Torch (TensorFlow no tiene ruedas para Python 3.14).
La API (Sequential, LSTM, EarlyStopping) es la misma que en el enunciado.
"""

from __future__ import annotations

import json
import os
import random
import warnings
from pathlib import Path
from typing import Any

# Backend antes de importar keras
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-config")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SERIES_DIR = ROOT / "outputs" / "parte1" / "series"
RESULTADOS_DIR = ROOT / "resultados"
MODELOS_DIR = ROOT / "modelos"
FIGURAS_DIR = RESULTADOS_DIR / "figuras"

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

# Series del laboratorio anterior (mismos train/test)
SERIE_1_SLUG = "total_mensual"
SERIE_2_SLUG = "frontera_01_la_aurora"
NOMBRE_SERIE_1 = "Total mensual"
NOMBRE_SERIE_2 = "Frontera 01 La Aurora"

# Mejores modelos del laboratorio 1 (tablas del informe)
LAB1_MEJORES = {
    NOMBRE_SERIE_1: {
        "Modelo": "HoltWinters(trend=add,seasonal=mul,damped=False)",
        "MAE": 134374.79,
        "RMSE": 146605.58,
        "MAPE": 51.47,
    },
    NOMBRE_SERIE_2: {
        "Modelo": "ARIMA(1, 1, 1)x(0, 0, 1, 12)",
        "MAE": 22967.05,
        "RMSE": 27986.19,
        "MAPE": 22.89,
    },
}

CONFIGURACIONES = [
    {
        "modelo": "LSTM_1",
        "look_back": 6,
        "unidades_1": 32,
        "unidades_2": None,
        "dropout": 0.10,
        "learning_rate": 0.001,
        "batch_size": 16,
    },
    {
        "modelo": "LSTM_2",
        "look_back": 12,
        "unidades_1": 64,
        "unidades_2": None,
        "dropout": 0.20,
        "learning_rate": 0.001,
        "batch_size": 16,
    },
    {
        "modelo": "LSTM_3",
        "look_back": 12,
        "unidades_1": 64,
        "unidades_2": 32,
        "dropout": 0.20,
        "learning_rate": 0.0005,
        "batch_size": 32,
    },
    {
        "modelo": "LSTM_4",
        "look_back": 24,
        "unidades_1": 64,
        "unidades_2": 32,
        "dropout": 0.30,
        "learning_rate": 0.0005,
        "batch_size": 32,
    },
]


def cargar_serie(slug: str) -> tuple[pd.Series, pd.Series]:
    train = pd.read_csv(SERIES_DIR / f"{slug}_train.csv", parse_dates=["fecha"])
    test = pd.read_csv(SERIES_DIR / f"{slug}_test.csv", parse_dates=["fecha"])
    train = train.set_index("fecha")["valor"].astype(float).sort_index()
    test = test.set_index("fecha")["valor"].astype(float).sort_index()
    train.name = "valor"
    test.name = "valor"
    return train, test


def convertir_a_serie(datos, nombre: str) -> pd.Series:
    if isinstance(datos, pd.DataFrame):
        if datos.shape[1] != 1:
            raise ValueError(f"{nombre} debe contener una sola columna.")
        datos = datos.iloc[:, 0]
    if not isinstance(datos, pd.Series):
        datos = pd.Series(np.asarray(datos).reshape(-1))
    datos = pd.to_numeric(datos, errors="coerce").dropna().astype(float)
    if len(datos) == 0:
        raise ValueError(f"{nombre} no contiene observaciones válidas.")
    return datos


def resumen_conjuntos(
    train_1: pd.Series,
    test_1: pd.Series,
    train_2: pd.Series,
    test_2: pd.Series,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Serie": [NOMBRE_SERIE_1, NOMBRE_SERIE_2],
            "Observaciones de entrenamiento": [len(train_1), len(train_2)],
            "Observaciones de prueba": [len(test_1), len(test_2)],
            "Inicio de entrenamiento": [train_1.index.min(), train_2.index.min()],
            "Fin de entrenamiento": [train_1.index.max(), train_2.index.max()],
            "Inicio de prueba": [test_1.index.min(), test_2.index.min()],
            "Fin de prueba": [test_1.index.max(), test_2.index.max()],
        }
    )


def graficar_train_test(train, test, nombre, ruta: Path | None = None):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(train.index, train.values, label="Entrenamiento")
    ax.plot(test.index, test.values, label="Prueba")
    ax.set_title(f"División temporal — {nombre}")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Valor de la serie")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if ruta is not None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def crear_secuencias(valores_escalados, look_back):
    X, y = [], []
    for i in range(look_back, len(valores_escalados)):
        X.append(valores_escalados[i - look_back : i, 0])
        y.append(valores_escalados[i, 0])
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


def preparar_datos(train, test, look_back, porcentaje_validacion=0.20):
    train_values = train.to_numpy(dtype=float).reshape(-1, 1)
    test_values = test.to_numpy(dtype=float).reshape(-1, 1)

    if len(train_values) <= look_back + 5:
        raise ValueError(
            f"El entrenamiento tiene {len(train_values)} valores y look_back={look_back}. "
            "Reduzca look_back o utilice una serie más larga."
        )

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_values)
    test_scaled = scaler.transform(test_values)

    X_completo, y_completo = crear_secuencias(train_scaled, look_back)
    cantidad_validacion = max(1, int(len(X_completo) * porcentaje_validacion))
    punto_validacion = len(X_completo) - cantidad_validacion

    X_train = X_completo[:punto_validacion]
    y_train = y_completo[:punto_validacion]
    X_val = X_completo[punto_validacion:]
    y_val = y_completo[punto_validacion:]

    contexto_test = np.vstack((train_scaled[-look_back:], test_scaled))
    X_test, y_test = crear_secuencias(contexto_test, look_back)

    return {
        "scaler": scaler,
        "train_scaled": train_scaled,
        "test_scaled": test_scaled,
        "X_completo": X_completo,
        "y_completo": y_completo,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def calcular_mape_seguro(y_real, y_pred):
    y_real = np.asarray(y_real, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mascara = y_real != 0
    if not np.any(mascara):
        return np.nan
    return np.mean(np.abs((y_real[mascara] - y_pred[mascara]) / y_real[mascara])) * 100


def calcular_metricas(y_real, y_pred):
    y_real = np.asarray(y_real, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return {
        "MAE": float(mean_absolute_error(y_real, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "MAPE": float(calcular_mape_seguro(y_real, y_pred)),
        "R2": float(r2_score(y_real, y_pred)),
    }


def construir_modelo_lstm(look_back, unidades_1, unidades_2, dropout, learning_rate):
    modelo = keras.Sequential(name="modelo_lstm")
    modelo.add(keras.Input(shape=(look_back, 1)))

    if unidades_2 is None:
        modelo.add(layers.LSTM(unidades_1))
        modelo.add(layers.Dropout(dropout))
    else:
        modelo.add(layers.LSTM(unidades_1, return_sequences=True))
        modelo.add(layers.Dropout(dropout))
        modelo.add(layers.LSTM(unidades_2))
        modelo.add(layers.Dropout(dropout))

    modelo.add(layers.Dense(1))
    optimizador = keras.optimizers.Adam(learning_rate=learning_rate)
    modelo.compile(optimizer=optimizador, loss="mse")
    return modelo


def entrenar_configuracion(train, test, config, verbose=0):
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)

    datos = preparar_datos(
        train=train,
        test=test,
        look_back=config["look_back"],
        porcentaje_validacion=0.20,
    )

    modelo = construir_modelo_lstm(
        look_back=config["look_back"],
        unidades_1=config["unidades_1"],
        unidades_2=config["unidades_2"],
        dropout=config["dropout"],
        learning_rate=config["learning_rate"],
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        min_delta=1e-5,
        restore_best_weights=True,
    )

    history = modelo.fit(
        datos["X_train"],
        datos["y_train"],
        validation_data=(datos["X_val"], datos["y_val"]),
        epochs=300,
        batch_size=config["batch_size"],
        shuffle=False,
        callbacks=[early_stopping],
        verbose=verbose,
    )

    pred_val_scaled = modelo.predict(datos["X_val"], verbose=0)
    y_val_real = datos["scaler"].inverse_transform(datos["y_val"].reshape(-1, 1)).reshape(-1)
    pred_val_real = datos["scaler"].inverse_transform(pred_val_scaled).reshape(-1)
    metricas = calcular_metricas(y_val_real, pred_val_real)
    mejor_epoca = int(np.argmin(history.history["val_loss"]) + 1)

    resultado = {
        **config,
        "epocas_ejecutadas": len(history.history["loss"]),
        "mejor_epoca": mejor_epoca,
        "val_MAE": metricas["MAE"],
        "val_RMSE": metricas["RMSE"],
        "val_MAPE": metricas["MAPE"],
        "val_R2": metricas["R2"],
    }
    return resultado, history.history


def tunear_serie(train, test, nombre_serie, configuraciones):
    resultados = []
    historiales = {}
    print(f"Tuneo de: {nombre_serie}")
    print("-" * 60)

    for config in configuraciones:
        print(f"Entrenando {config['modelo']}...")
        try:
            resultado, historial = entrenar_configuracion(
                train=train, test=test, config=config, verbose=0
            )
            resultados.append(resultado)
            historiales[config["modelo"]] = historial
            print(
                f"  RMSE validación: {resultado['val_RMSE']:.4f} | "
                f"MAE: {resultado['val_MAE']:.4f} | "
                f"Mejor época: {resultado['mejor_epoca']}"
            )
        except ValueError as error:
            print(f"  Configuración omitida: {error}")

    if len(resultados) < 2:
        raise ValueError(
            "Se necesitan por lo menos dos configuraciones válidas. "
            "Reduzca los valores de look_back."
        )

    tabla = pd.DataFrame(resultados).sort_values("val_RMSE").reset_index(drop=True)
    return tabla, historiales


def graficar_historiales(historiales, nombre_serie, ruta: Path | None = None):
    fig, ax = plt.subplots(figsize=(12, 6))
    for nombre_modelo, historial in historiales.items():
        ax.plot(historial["val_loss"], label=f"{nombre_modelo} - validación")
    ax.set_title(f"Pérdida de validación — {nombre_serie}")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE escalado")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if ruta is not None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def entrenar_mejor_y_predecir(train, test, fila_mejor_configuracion):
    config = fila_mejor_configuracion.to_dict()
    look_back = int(config["look_back"])
    mejor_epoca = max(1, int(config["mejor_epoca"]))

    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)

    datos = preparar_datos(train, test, look_back, porcentaje_validacion=0.20)
    unidades_2 = config["unidades_2"]
    if unidades_2 is None or (isinstance(unidades_2, float) and np.isnan(unidades_2)):
        unidades_2 = None
    else:
        unidades_2 = int(unidades_2)

    modelo = construir_modelo_lstm(
        look_back=look_back,
        unidades_1=int(config["unidades_1"]),
        unidades_2=unidades_2,
        dropout=float(config["dropout"]),
        learning_rate=float(config["learning_rate"]),
    )

    historial_final = modelo.fit(
        datos["X_completo"],
        datos["y_completo"],
        epochs=mejor_epoca,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        verbose=0,
    )

    pred_test_scaled = modelo.predict(datos["X_test"], verbose=0)
    pred_test = datos["scaler"].inverse_transform(pred_test_scaled).reshape(-1)
    y_test_real = test.to_numpy(dtype=float).reshape(-1)
    metricas_test = calcular_metricas(y_test_real, pred_test)

    predicciones = pd.DataFrame(
        {"Real": y_test_real, "Prediccion_LSTM": pred_test},
        index=test.index,
    )
    return modelo, historial_final.history, metricas_test, predicciones


def graficar_predicciones(predicciones, nombre_serie, ruta: Path | None = None):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(predicciones.index, predicciones["Real"], marker="o", label="Valor real")
    ax.plot(
        predicciones.index,
        predicciones["Prediccion_LSTM"],
        marker="o",
        label="Predicción LSTM",
    )
    ax.set_title(f"Valores reales y predicciones — {nombre_serie}")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Valor")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if ruta is not None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def escribir_interpretaciones(
    resultados_1: pd.DataFrame,
    resultados_2: pd.DataFrame,
    mejor_1: pd.Series,
    mejor_2: pd.Series,
    metricas_1: dict,
    metricas_2: dict,
) -> str:
    mejor_lab1_1 = LAB1_MEJORES[NOMBRE_SERIE_1]
    mejor_lab1_2 = LAB1_MEJORES[NOMBRE_SERIE_2]

    gana_1 = "LSTM" if metricas_1["RMSE"] < mejor_lab1_1["RMSE"] else mejor_lab1_1["Modelo"]
    gana_2 = "LSTM" if metricas_2["RMSE"] < mejor_lab1_2["RMSE"] else mejor_lab1_2["Modelo"]
    # Comparar entre series con MAPE (escalas distintas)
    mejor_serie = (
        NOMBRE_SERIE_1
        if metricas_1["MAPE"] <= metricas_2["MAPE"]
        else NOMBRE_SERIE_2
    )

    texto = f"""# Interpretaciones — Laboratorio 2 (LSTM)

## Resultados del tuneo — {NOMBRE_SERIE_1}

Se evaluaron {len(resultados_1)} configuraciones distintas. La configuración con el menor RMSE de validación fue **{mejor_1['modelo']}**, con un RMSE de **{mejor_1['val_RMSE']:.2f}** y un MAE de **{mejor_1['val_MAE']:.2f}**. Esta configuración utilizó un `look_back` de **{int(mejor_1['look_back'])}**, **{int(mejor_1['unidades_1'])}** unidades en la primera capa LSTM, un dropout de **{mejor_1['dropout']}** y una tasa de aprendizaje de **{mejor_1['learning_rate']}**.

## Resultados del tuneo — {NOMBRE_SERIE_2}

Se evaluaron {len(resultados_2)} configuraciones distintas. La configuración con el menor RMSE de validación fue **{mejor_2['modelo']}**, con un RMSE de **{mejor_2['val_RMSE']:.2f}** y un MAE de **{mejor_2['val_MAE']:.2f}**. El resultado muestra que **{mejor_2['modelo']}** (look_back={int(mejor_2['look_back'])}, capas={'1' if pd.isna(mejor_2['unidades_2']) or mejor_2['unidades_2'] is None else '2'}) fue la más estable en validación.

## Comparación e interpretación

Para la serie **{NOMBRE_SERIE_1}**, el mejor modelo fue **{mejor_1['modelo']}**. Obtuvo un RMSE de prueba de **{metricas_1['RMSE']:.2f}**, un MAE de **{metricas_1['MAE']:.2f}** y un MAPE de **{metricas_1['MAPE']:.2f}%**.

Para la serie **{NOMBRE_SERIE_2}**, el mejor modelo fue **{mejor_2['modelo']}**, con un RMSE de **{metricas_2['RMSE']:.2f}**.

Al comparar ambas series (escalas distintas), el modelo predijo mejor **{mejor_serie}** según MAPE de prueba ({min(metricas_1['MAPE'], metricas_2['MAPE']):.2f}%). No se utilizó el conjunto de prueba para seleccionar los hiperparámetros; la selección se realizó exclusivamente con el error de validación.

## Comparación con el laboratorio anterior

El mejor modelo del laboratorio anterior para **{NOMBRE_SERIE_1}** fue **{mejor_lab1_1['Modelo']}**, con un RMSE de **{mejor_lab1_1['RMSE']:.2f}**. El modelo LSTM obtuvo un RMSE de **{metricas_1['RMSE']:.2f}**. Por lo tanto, **{gana_1}** produjo el mejor resultado sobre el mismo conjunto de prueba.

En **{NOMBRE_SERIE_2}**, el mejor modelo previo fue **{mejor_lab1_2['Modelo']}** (RMSE **{mejor_lab1_2['RMSE']:.2f}**) frente a LSTM (RMSE **{metricas_2['RMSE']:.2f}**). El ganador es **{gana_2}**.

## Conclusiones

1. Se construyeron cuatro configuraciones LSTM por serie y se modificaron `look_back`, unidades, capas, dropout, tasa de aprendizaje y batch size.
2. El mejor modelo para **{NOMBRE_SERIE_1}** fue **{mejor_1['modelo']}** (RMSE prueba **{metricas_1['RMSE']:.2f}**).
3. El mejor modelo para **{NOMBRE_SERIE_2}** fue **{mejor_2['modelo']}** (RMSE prueba **{metricas_2['RMSE']:.2f}**).
4. La serie **{mejor_serie}** fue predicha con mayor precisión según las métricas de prueba.
5. Respecto al laboratorio anterior, LSTM **{'superó' if gana_1 == 'LSTM' or gana_2 == 'LSTM' else 'no superó de forma consistente'}** a los modelos clásicos; la comparación usa el mismo conjunto de prueba.
"""
    return texto


def run() -> dict[str, Any]:
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

    print("Keras:", keras.__version__, "| backend:", keras.backend.backend())

    train_1, test_1 = cargar_serie(SERIE_1_SLUG)
    train_2, test_2 = cargar_serie(SERIE_2_SLUG)
    train_1 = convertir_a_serie(train_1, "train_serie_1")
    test_1 = convertir_a_serie(test_1, "test_serie_1")
    train_2 = convertir_a_serie(train_2, "train_serie_2")
    test_2 = convertir_a_serie(test_2, "test_serie_2")

    resumen = resumen_conjuntos(train_1, test_1, train_2, test_2)
    resumen.to_csv(RESULTADOS_DIR / "resumen_conjuntos.csv", index=False)
    print(resumen.to_string(index=False))

    graficar_train_test(
        train_1, test_1, NOMBRE_SERIE_1, FIGURAS_DIR / "split_serie_1.png"
    )
    graficar_train_test(
        train_2, test_2, NOMBRE_SERIE_2, FIGURAS_DIR / "split_serie_2.png"
    )

    pd.DataFrame(CONFIGURACIONES).to_csv(
        RESULTADOS_DIR / "configuraciones.csv", index=False
    )

    resultados_1, hist_1 = tunear_serie(
        train_1, test_1, NOMBRE_SERIE_1, CONFIGURACIONES
    )
    resultados_2, hist_2 = tunear_serie(
        train_2, test_2, NOMBRE_SERIE_2, CONFIGURACIONES
    )

    resultados_1.to_csv(RESULTADOS_DIR / "tuneo_serie_1.csv", index=False)
    resultados_2.to_csv(RESULTADOS_DIR / "tuneo_serie_2.csv", index=False)

    graficar_historiales(hist_1, NOMBRE_SERIE_1, FIGURAS_DIR / "val_loss_serie_1.png")
    graficar_historiales(hist_2, NOMBRE_SERIE_2, FIGURAS_DIR / "val_loss_serie_2.png")

    mejor_1 = resultados_1.iloc[0]
    mejor_2 = resultados_2.iloc[0]

    modelo_1, _, metricas_1, pred_1 = entrenar_mejor_y_predecir(train_1, test_1, mejor_1)
    modelo_2, _, metricas_2, pred_2 = entrenar_mejor_y_predecir(train_2, test_2, mejor_2)

    graficar_predicciones(pred_1, NOMBRE_SERIE_1, FIGURAS_DIR / "pred_serie_1.png")
    graficar_predicciones(pred_2, NOMBRE_SERIE_2, FIGURAS_DIR / "pred_serie_2.png")

    comparacion_mejores = pd.DataFrame(
        [
            {
                "Serie": NOMBRE_SERIE_1,
                "Mejor modelo": mejor_1["modelo"],
                "Look back": int(mejor_1["look_back"]),
                "MAE prueba": metricas_1["MAE"],
                "RMSE prueba": metricas_1["RMSE"],
                "MAPE prueba": metricas_1["MAPE"],
                "R2 prueba": metricas_1["R2"],
            },
            {
                "Serie": NOMBRE_SERIE_2,
                "Mejor modelo": mejor_2["modelo"],
                "Look back": int(mejor_2["look_back"]),
                "MAE prueba": metricas_2["MAE"],
                "RMSE prueba": metricas_2["RMSE"],
                "MAPE prueba": metricas_2["MAPE"],
                "R2 prueba": metricas_2["R2"],
            },
        ]
    )
    comparacion_mejores.to_csv(
        RESULTADOS_DIR / "comparacion_mejores_lstm.csv", index=False
    )

    comparacion_labs = pd.DataFrame(
        [
            {
                "Serie": NOMBRE_SERIE_1,
                "Modelo": LAB1_MEJORES[NOMBRE_SERIE_1]["Modelo"],
                "MAE": LAB1_MEJORES[NOMBRE_SERIE_1]["MAE"],
                "RMSE": LAB1_MEJORES[NOMBRE_SERIE_1]["RMSE"],
                "MAPE": LAB1_MEJORES[NOMBRE_SERIE_1]["MAPE"],
            },
            {
                "Serie": NOMBRE_SERIE_1,
                "Modelo": f"LSTM ({mejor_1['modelo']})",
                "MAE": metricas_1["MAE"],
                "RMSE": metricas_1["RMSE"],
                "MAPE": metricas_1["MAPE"],
            },
            {
                "Serie": NOMBRE_SERIE_2,
                "Modelo": LAB1_MEJORES[NOMBRE_SERIE_2]["Modelo"],
                "MAE": LAB1_MEJORES[NOMBRE_SERIE_2]["MAE"],
                "RMSE": LAB1_MEJORES[NOMBRE_SERIE_2]["RMSE"],
                "MAPE": LAB1_MEJORES[NOMBRE_SERIE_2]["MAPE"],
            },
            {
                "Serie": NOMBRE_SERIE_2,
                "Modelo": f"LSTM ({mejor_2['modelo']})",
                "MAE": metricas_2["MAE"],
                "RMSE": metricas_2["RMSE"],
                "MAPE": metricas_2["MAPE"],
            },
        ]
    )
    comparacion_labs.to_csv(
        RESULTADOS_DIR / "comparacion_laboratorio_anterior.csv", index=False
    )

    pred_1.to_csv(RESULTADOS_DIR / "predicciones_serie_1.csv")
    pred_2.to_csv(RESULTADOS_DIR / "predicciones_serie_2.csv")

    modelo_1.save(MODELOS_DIR / "mejor_lstm_serie_1.keras")
    modelo_2.save(MODELOS_DIR / "mejor_lstm_serie_2.keras")

    with open(RESULTADOS_DIR / "configuracion_serie_1.json", "w", encoding="utf-8") as f:
        json.dump(mejor_1.to_dict(), f, indent=4, default=str)
    with open(RESULTADOS_DIR / "configuracion_serie_2.json", "w", encoding="utf-8") as f:
        json.dump(mejor_2.to_dict(), f, indent=4, default=str)

    interpretaciones = escribir_interpretaciones(
        resultados_1, resultados_2, mejor_1, mejor_2, metricas_1, metricas_2
    )
    (RESULTADOS_DIR / "interpretaciones.md").write_text(interpretaciones, encoding="utf-8")

    print("\nComparación mejores LSTM:")
    print(comparacion_mejores.to_string(index=False))
    print("\nComparación vs laboratorio anterior:")
    print(comparacion_labs.to_string(index=False))
    print("\nResultados y modelos guardados en resultados/ y modelos/")

    return {
        "resumen": resumen,
        "resultados_serie_1": resultados_1,
        "resultados_serie_2": resultados_2,
        "metricas_test_serie_1": metricas_1,
        "metricas_test_serie_2": metricas_2,
        "comparacion_mejores": comparacion_mejores,
        "comparacion_labs": comparacion_labs,
    }


if __name__ == "__main__":
    run()
