# Discusion — LSTM + catch22 (Ejercicio 2.14)

## Diseno del experimento

Para poder aislar el efecto de agregar las caracteristicas catch22 (y no confundirlo con un
cambio de arquitectura), el modelo `LSTM + catch22` usa exactamente el mismo backbone LSTM que
el mejor modelo del Ejercicio 1 para **Total mensual** (`LSTM_1`: look_back=
6, 32 unidades LSTM, dropout=
0.1, learning_rate=0.001, batch_size=
16). La unica diferencia es una segunda entrada: el vector de 22
caracteristicas catch22, calculado **unicamente con el conjunto de entrenamiento** (para no
filtrar informacion del conjunto de prueba) y normalizado internamente (media 0, desviacion 1
sobre sus propios 22 valores). Ese vector se pasa por una capa densa de 8 unidades y se
concatena con la salida de la LSTM antes de la capa de salida.

## Resultados

| Modelo | MAE prueba | RMSE prueba | MAPE prueba | R2 prueba |
|---|---|---|---|---|
| LSTM (Ejercicio 1) | 40554.45 | 50794.23 | 18.05% | 0.543 |
| LSTM + catch22 (2.14) | 60182.80 | 73537.01 | 31.85% | 0.043 |

El RMSE de prueba cambio en +44.77% al agregar catch22. El mejor modelo para
**Total mensual** fue **LSTM (Ejercicio 1)**.

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
