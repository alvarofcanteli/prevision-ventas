import pandas as pd
import numpy as np
from pathlib import Path

# --- Carga ---
DATA_DIR = Path(r"C:\Users\afcan\proyectos\prevision_ventas\datos")
df = pd.read_csv(DATA_DIR / "dataset_limpio.csv", parse_dates=["date"])

# Comprobamos el rango de fechas disponible
print(f"Fecha inicio: {df['date'].min().date()}")
print(f"Fecha fin:    {df['date'].max().date()}")

# --- División train / test ---
# Los datos van de enero 2016 a agosto 2017 (no tenemos datos completos de 2017)
# Por eso no podemos hacer train=2016 y test=2017 completo
# Solución: usamos julio-agosto 2017 como test (138 filas, ~mes y medio)
# y todo lo anterior como train (1638 filas, 18 meses)
FECHA_CORTE = "2017-07-01"

train = df[df["date"] < FECHA_CORTE]
test  = df[df["date"] >= FECHA_CORTE]

print(f"\nTrain: {train['date'].min().date()} → {train['date'].max().date()} ({len(train)} filas)")
print(f"Test:  {test['date'].min().date()}  → {test['date'].max().date()} ({len(test)} filas)")


# --- Modelo Base 1: Semana anterior ---
# Para predecir las ventas de hoy usamos las ventas de hace 7 días
# Esto se llama lag de 7 días

# Calculamos el lag de 7 días para cada familia por separado
# sort_values asegura que los datos estén ordenados por fecha antes de desplazar
df_sorted = df.sort_values(["family", "date"])
df_sorted["pred_base1"] = df_sorted.groupby("family")["sales"].shift(7)

# Nos quedamos solo con el periodo de test para evaluar
test_b1 = df_sorted[df_sorted["date"] >= FECHA_CORTE].dropna(subset=["pred_base1"])


# --- Evaluación del Modelo Base 1 ---
# MAE: error medio absoluto
# RMSE: raíz del error cuadrático medio (penaliza errores grandes)
# MAPE: error porcentual medio (fácil de comunicar a una empresa)

def evaluar_modelo(nombre, real, predicho):
    mae  = np.mean(np.abs(real - predicho))
    rmse = np.sqrt(np.mean((real - predicho) ** 2))
    mape = np.mean(np.abs((real - predicho) / real)) * 100
    print(f"\n{nombre}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    return {"modelo": nombre, "MAE": mae, "RMSE": rmse, "MAPE": mape}

resultado_b1 = evaluar_modelo("Modelo Base 1 (semana anterior)", 
                               test_b1["sales"], 
                               test_b1["pred_base1"])



# --- Modelo Base 2: Media de las últimas 4 semanas ---
# Para predecir las ventas de hoy usamos la media de las ventas
# de hace 7, 14, 21 y 28 días — es decir, las últimas 4 semanas

# Calculamos los 4 lags necesarios para cada familia por separado
df_sorted["lag_7"]  = df_sorted.groupby("family")["sales"].shift(7)
df_sorted["lag_14"] = df_sorted.groupby("family")["sales"].shift(14)
df_sorted["lag_21"] = df_sorted.groupby("family")["sales"].shift(21)
df_sorted["lag_28"] = df_sorted.groupby("family")["sales"].shift(28)

# La predicción es la media de los 4 lags
df_sorted["pred_base2"] = df_sorted[["lag_7", "lag_14", "lag_21", "lag_28"]].mean(axis=1)

# Nos quedamos solo con el periodo de test y eliminamos filas sin predicción
test_b2 = df_sorted[df_sorted["date"] >= FECHA_CORTE].dropna(subset=["pred_base2"])

# Evaluamos el modelo con las mismas métricas que el anterior
resultado_b2 = evaluar_modelo("Modelo Base 2 (media 4 semanas)",
                               test_b2["sales"],
                               test_b2["pred_base2"])



from sklearn.linear_model import LinearRegression


# --- Modelo 3: Regresión lineal con lags y promoción ---
# Preparamos las variables que el modelo va a usar para predecir

# Extraemos mes y día de la semana como variables numéricas
df_sorted["mes"]        = df_sorted["date"].dt.month
df_sorted["dia_semana"] = df_sorted["date"].dt.dayofweek

# Definimos las variables que usará el modelo (features) y lo que queremos predecir (target)
FEATURES = ["lag_7", "lag_14", "mes", "dia_semana", "onpromotion"]
TARGET   = "sales"

# Eliminamos filas con valores nulos en las features (las primeras 14 filas sin lags completos)
df_modelo = df_sorted.dropna(subset=FEATURES)

# Dividimos en train y test usando la misma fecha de corte que antes
train_ml = df_modelo[df_modelo["date"] < FECHA_CORTE]
test_ml  = df_modelo[df_modelo["date"] >= FECHA_CORTE]

# Separamos features y target en train y test
X_train = train_ml[FEATURES]
y_train = train_ml[TARGET]
X_test  = test_ml[FEATURES]
y_test  = test_ml[TARGET]

# Entrenamos el modelo; aquí es donde "aprende" los pesos de cada variable
modelo_lr = LinearRegression()
modelo_lr.fit(X_train, y_train)

# Hacemos las predicciones sobre el test
pred_lr = modelo_lr.predict(X_test)

# Evaluamos el modelo
resultado_lr = evaluar_modelo("Modelo 3 (regresión lineal)",
                               y_test,
                               pred_lr)

# Mostramos los pesos que el modelo ha aprendido para cada variable
print("\nPesos del modelo:")
for feature, coef in zip(FEATURES, modelo_lr.coef_):
    print(f"  {feature}: {coef:.4f}")



from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")  # SARIMA genera muchos avisos técnicos, los ocultamos

# --- Modelo 4: SARIMA ---
# Aplicamos SARIMA a cada familia por separado
# porque trabaja mejor con una sola serie temporal

resultados_sarima = []

for familia in df_sorted["family"].unique():
    
    # Filtramos los datos de esta familia y los ordenamos por fecha
    serie = df_sorted[df_sorted["family"] == familia].set_index("date")["sales"]
    
    # Dividimos en train y test
    train_s = serie[serie.index < FECHA_CORTE]
    test_s  = serie[serie.index >= FECHA_CORTE]
    
    # Aseguramos que el índice tiene frecuencia diaria definida
    train_s = train_s.asfreq("D").fillna(0)
    test_s  = test_s.asfreq("D").fillna(0)

    # Entrenamos el modelo SARIMA
    # (1,0,1) — parámetros de tendencia: un lag, sin diferenciación, una media móvil
    # (1,0,1,7) — parámetros estacionales con ciclo de 7 días (semanal)
    modelo_sarima = SARIMAX(train_s, order=(1,0,1), seasonal_order=(1,0,1,7))
    resultado_fit = modelo_sarima.fit(disp=False)
    
    # Hacemos predicciones para el periodo de test
    pred_s = resultado_fit.forecast(steps=len(test_s))
    
    # Evaluamos el modelo para esta familia
    print(f"\n  Familia: {familia}")
    res = evaluar_modelo(f"Modelo 4 SARIMA ({familia})", test_s, pred_s)
    resultados_sarima.append(res)


import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Gráfica interactiva comparativa de modelos ---
# Puedes activar y desactivar cada línea haciendo clic en la leyenda
test_ml_copy = test_ml.copy()
test_ml_copy["pred_lr"] = pred_lr

fig = make_subplots(rows=3, cols=1, 
                    subplot_titles=["BEVERAGES", "DAIRY", "GROCERY"],
                    shared_xaxes=True,
                    vertical_spacing=0.08)

colores = {
    "Real":      "black",
    "Base 1":    "blue",
    "Base 2":    "orange",
    "Regresión": "green",
    "SARIMA":    "red"
}

for i, familia in enumerate(df_sorted["family"].unique(), start=1):

    # Datos reales y modelos base
    mask_base = (df_sorted["family"] == familia) & (df_sorted["date"] >= FECHA_CORTE)
    datos_base = df_sorted[mask_base].dropna(subset=["pred_base1", "pred_base2"]).set_index("date")

    # Datos regresión lineal
    datos_lr = test_ml_copy[test_ml_copy["family"] == familia].set_index("date")

    # Datos SARIMA
    serie = df_sorted[df_sorted["family"] == familia].set_index("date")["sales"]
    train_s = serie[serie.index < FECHA_CORTE].asfreq("D").fillna(0)
    test_s  = serie[serie.index >= FECHA_CORTE].asfreq("D").fillna(0)
    modelo_s = SARIMAX(train_s, order=(1,0,1), seasonal_order=(1,0,1,7))
    fit_s = modelo_s.fit(disp=False)
    pred_s = fit_s.forecast(steps=len(test_s))

    # Añadimos cada línea — showlegend=True solo en la primera familia para no repetir
    mostrar_leyenda = True
    fig.add_trace(go.Scatter(x=datos_base.index, y=datos_base["sales"],
                             name="Real", line=dict(color="black", width=2),
                             showlegend=mostrar_leyenda), row=i, col=1)

    fig.add_trace(go.Scatter(x=datos_base.index, y=datos_base["pred_base1"],
                             name="Base 1", line=dict(color="blue", dash="dash"),
                             showlegend=mostrar_leyenda), row=i, col=1)

    fig.add_trace(go.Scatter(x=datos_base.index, y=datos_base["pred_base2"],
                             name="Base 2", line=dict(color="orange", dash="dash"),
                             showlegend=mostrar_leyenda), row=i, col=1)

    fig.add_trace(go.Scatter(x=datos_lr.index, y=datos_lr["pred_lr"],
                             name="Regresión", line=dict(color="green", dash="dash"),
                             showlegend=mostrar_leyenda), row=i, col=1)

    fig.add_trace(go.Scatter(x=test_s.index, y=pred_s,
                             name="SARIMA", line=dict(color="red", dash="dash"),
                             showlegend=mostrar_leyenda), row=i, col=1)

fig.update_layout(title="Predicciones vs Real por familia (julio-agosto 2017)",
                  height=900,
                  hovermode="x unified")

# Guardamos como HTML para poder abrirlo en el navegador
fig.write_html(r"C:\Users\afcan\proyectos\prevision_ventas\graficas\Demand_forecasting_Grafico5_Comparativa_interactiva.html")

fig.show()