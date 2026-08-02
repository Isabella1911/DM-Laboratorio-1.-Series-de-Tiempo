# Interpretaciones — Laboratorio 2, Ejercicio 2 (catch22)

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

serie
Total mensual                210
Frontera 01 La Aurora        210
Frontera 07 Valle Nuevo      210
Frontera 09 San Cristóbal    210
Vía Aérea                    210
Vía Terrestre                210
Vía Marítima                  96

Via Maritima solo cuenta con observaciones reales entre 2009 y 2016 (96 meses); sus 22
caracteristicas catch22 se calcularon unicamente sobre ese tramo valido, no sobre el rango
completo 2009-2026 usado por las demas series. Esta diferencia de cobertura temporal debe
tenerse en cuenta al interpretar su posicion en el PCA y en el mapa de distancias.

## 2.7 Series con comportamiento mas similar

El par de series con la menor distancia euclidiana en el espacio catch22 estandarizado fue
**Frontera 01 La Aurora** y **Vía Aérea** (distancia = 0.27). Esto es
consistente con lo esperado: ambas comparten la misma dinamica subyacente porque una vía
concentra casi todo su volumen en una sola frontera dominante.

## 2.8 Caracteristicas catch22 mas importantes para diferenciar las series

Las caracteristicas con mayor peso combinado en las dos primeras componentes principales
(que explican en conjunto 80.6% de
la varianza) fueron:

                             caracteristica  peso_abs_pc1_pc2  carga_pc1  carga_pc2
                         DN_HistogramMode_5          0.529162   0.191009  -0.338154
                  PD_PeriodicityWang_th0_01          0.488291   0.103285   0.385006
                  SB_MotifThree_quantile_hh          0.472822  -0.232320   0.240501
SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1          0.433971   0.083991   0.349980
    IN_AutoMutualInfoStats_40_gaussian_fmmi          0.421466   0.033907   0.387559

## 2.9 Grupos naturales de series

El corte estadisticamente optimo segun el coeficiente de silueta fue **k=2**
(silueta=0.466), que separa a Via Maritima de
las otras seis series:

- Cluster 1: Total mensual, Frontera 01 La Aurora, Frontera 07 Valle Nuevo, Frontera 09 San Cristóbal, Vía Aérea, Vía Terrestre
- Cluster 2: Vía Marítima

Esta particion en dos grupos es correcta pero poco informativa: esta dominada por el
caracter atipico de Via Maritima (cobertura temporal incompleta) y no distingue nada entre
las otras seis series. Cortando el mismo dendrograma en **k=3** (silueta=0.388,
ligeramente menor pero mucho mas interpretable) aparece una estructura mas rica:

- Cluster 1: Frontera 01 La Aurora, Vía Aérea
- Cluster 2: Total mensual, Frontera 07 Valle Nuevo, Frontera 09 San Cristóbal, Vía Terrestre
- Cluster 3: Vía Marítima

Esta particion en tres grupos si revela una estructura interesante: un grupo formado
exclusivamente por Frontera 01 La Aurora y Via Aerea (el par identificado como mas similar
en la pregunta 2.7); un segundo grupo con el total mensual, las otras dos fronteras y Via
Terrestre; y Via Maritima sola. En ambos casos usamos el mismo enlace jerarquico (Ward);
solo cambia la altura de corte del dendrograma.

## 2.10 Cohesion por categoria (fronteras vs. vias)

Composicion de cada cluster de la particion k=3 por categoria (frontera / via / total):

cluster_k3          
1           frontera    1.0
            via         1.0
            total       NaN
2           frontera    2.0
            via         1.0
            total       1.0
3           frontera    NaN
            via         1.0
            total       NaN

Las series **no** se agrupan de forma perfecta por categoria administrativa (frontera vs.
via): Via Aerea se agrupa con una frontera (La Aurora) y no con las otras dos vias, mientras
que Via Terrestre se agrupa con fronteras terrestres y con el total, no con Via Aerea. Lo que
domina el agrupamiento es la composicion real del trafico (dominado por un aeropuerto vs.
dominado por cruces terrestres) y la escala, mas que la etiqueta de categoria bajo la cual
fue construida la serie en el Laboratorio 1.

## 2.11 Serie con comportamiento mas atipico

La serie con mayor distancia promedio respecto al resto (menos parecida al conjunto) fue
**Vía Marítima** (distancia promedio = 10.66). Esto es
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
