# Previsión de Ventas en Supermercado

Análisis de la demanda de productos en un supermercado y construcción de modelos de predicción de ventas para mejorar la gestión de inventario.

## Contexto

Este proyecto utiliza datos reales de ventas de una cadena de supermercados ecuatoriana (dataset de Kaggle: Store Sales - Time Series Forecasting). El análisis se centra en tres familias de productos — BEVERAGES, DAIRY y GROCERY — de una sola tienda, con datos desde enero de 2016 hasta agosto de 2017.

## Objetivo

Predecir las ventas futuras de cada familia de productos para ayudar a una empresa a:
- Planificar mejor el stock
- Reducir el exceso de inventario
- Anticipar picos de demanda
- Tomar mejores decisiones de compra

## Estructura del proyecto
```
prevision-ventas/
├── datos/
│   └── dataset_limpio.csv    # Dataset filtrado y limpio
├── graficas/                 # Gráficas generadas por el análisis
├── dataset.py                # Carga, limpieza y filtrado de datos
├── exploratorio.py           # Análisis exploratorio de ventas
└── modelo.py                 # Modelos de predicción y evaluación
```

## Scripts

### dataset.py
Carga el dataset original de Kaggle, filtra las familias y tienda de interés, y genera el dataset limpio listo para el análisis.

### exploratorio.py
Análisis exploratorio con 4 gráficas:
- Evolución de ventas en el tiempo por familia
- Ventas medias por familia
- Efecto de las promociones en las ventas
- Estacionalidad mensual y semanal

### modelo.py
Construcción y evaluación de 4 modelos de predicción:
- **Base 1** — ventas de la semana anterior
- **Base 2** — media de las últimas 4 semanas
- **Regresión lineal** — con variables lag, mes, día de la semana y promoción
- **SARIMA** — modelo de series temporales con estacionalidad semanal

## Resultados

| Modelo | MAE | RMSE | MAPE |
|---|---|---|---|
| Base 1 (semana anterior) | 180.78 | 287.91 | 12.11% |
| Base 2 (media 4 semanas) | 172.22 | 276.08 | 11.97% |
| Regresión lineal | 206.90 | 269.17 | 18.33% |
| SARIMA BEVERAGES | 205.74 | 306.05 | 11.65% |
| SARIMA DAIRY | 84.28 | 120.29 | 15.31% |
| SARIMA GROCERY | 227.06 | 356.07 | 11.63% |

**Conclusión:** Base 2 es el modelo más consistente en general. SARIMA destaca en DAIRY pero no mejora en las demás familias. La regresión lineal muestra el peor MAPE y sesgo de sobreestimación en DAIRY.

## Requisitos
```
pandas
matplotlib
seaborn
scikit-learn
statsmodels
plotly
```

## Datos

Los datos originales provienen del dataset público de Kaggle:
[Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

Los CSVs originales no están incluidos en el repositorio por su tamaño. Solo se incluye el dataset limpio generado por `dataset.py`.