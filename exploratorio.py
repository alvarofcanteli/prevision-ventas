# --- Librerías ---
import pandas as pd          
import matplotlib.pyplot as plt  
from pathlib import Path      

# --- Carga del dataset limpio ---
DATA_DIR = Path(r"C:\Users\afcan\proyectos\prevision_ventas\datos")

# Cargamos el CSV limpio que generamos con dataset.py
# parse_dates convierte la columna date de texto a fecha real ya que aunque lo hayamos hecho, la fecha no se guarda
df = pd.read_csv(DATA_DIR / "dataset_limpio.csv", parse_dates=["date"])


# --- Gráfica 1: Evolución de ventas en el tiempo ---
# Creamos el lienzo: fig es la ventana, ax es el área de dibujo
# figsize define el tamaño en pulgadas (ancho, alto)
fig, ax = plt.subplots(figsize=(12, 5))

# Recorremos cada familia (BEVERAGES, DAIRY, GROCERY)
for familia in df["family"].unique():
    # Filtramos solo las filas de esta familia
    # set_index("date") pone la fecha como índice para que el eje X sea el tiempo
    # ["sales"] se queda solo con la columna de ventas
    datos = df[df["family"] == familia].set_index("date")["sales"]
    
    # Dibujamos las ventas diarias con línea fina y transparente (alpha=0.3)
    # Así se ve el detalle pero sin saturar la gráfica
    ax.plot(datos, alpha=0.3, linewidth=0.8, label=f"{familia} (diario)")
    
    # Media móvil de 7 días: suaviza el ruido diario para ver la tendencia
    # rolling(7) calcula la media de los últimos 7 días en cada punto
    ax.plot(datos.rolling(7).mean(), linewidth=2, label=f"{familia} (media 7d)")

# Títulos y etiquetas
ax.set_title("Evolución de ventas por familia (Tienda 1, 2016-2017)")
ax.set_xlabel("Fecha")
ax.set_ylabel("Ventas")

# Leyenda para identificar cada línea
ax.legend()

# Ajusta los márgenes para que no se corte nada
plt.tight_layout()

# Muestra la ventana con la gráfica y guardamos la gráfica en la carpeta gráficas

plt.savefig(r"C:\Users\afcan\proyectos\prevision_ventas\graficas\Demand_forecasting_Grafico1_Evolución_de_ventas_por_familia.png", dpi=150)

plt.show()

# --- Gráfica 2: Ventas medias por familia ---
# Calculamos la media de ventas agrupando por familia
# Esto resume en un solo número cuánto vende de media cada familia
medias = df.groupby("family")["sales"].mean()

# Creamos el lienzo
fig, ax = plt.subplots(figsize=(8, 5))

# Gráfico de barras — una barra por familia
# color define los colores de cada barra
medias.plot(kind="bar", ax=ax, color=["steelblue", "coral", "mediumseagreen"])

# Añadimos el valor exacto encima de cada barra para que sea más fácil de leer
for i, valor in enumerate(medias):
    ax.text(i, valor + 10, f"{valor:.0f}", ha="center", fontweight="bold")

ax.set_title("Ventas medias por familia (Tienda 1, 2016-2017)")
ax.set_xlabel("Familia")
ax.set_ylabel("Ventas medias diarias")

# Rotamos las etiquetas del eje X para que se lean bien
ax.tick_params(axis="x", rotation=0)

plt.tight_layout()

# Guardamos la gráfica en la carpeta graficas
plt.savefig(r"C:\Users\afcan\proyectos\prevision_ventas\graficas\Demand_forecasting_Grafico2_Ventas_medias_por_familia.png", dpi=150)

plt.show()


# --- Gráfica 3: Ventas con y sin promoción ---
# Creamos una columna nueva que indica si hay promoción o no
# Si onpromotion > 0 significa que hay productos en promoción ese día
df["hay_promocion"] = df["onpromotion"] > 0

# Calculamos la media de ventas agrupando por familia y si hay promoción o no
# Esto nos da 6 valores: 3 familias x 2 estados (con/sin promoción)
medias_promo = df.groupby(["family", "hay_promocion"])["sales"].mean().unstack()

# Renombramos las columnas para que la leyenda sea más clara
medias_promo.columns = ["Sin promoción", "Con promoción"]

# Creamos el lienzo
fig, ax = plt.subplots(figsize=(9, 5))

# Gráfico de barras agrupadas — dos barras por familia
medias_promo.plot(kind="bar", ax=ax, color=["steelblue", "coral"])

# Añadimos el valor exacto encima de cada barra
for container in ax.containers:
    ax.bar_label(container, fmt="%.0f", padding=3, fontweight="bold")

ax.set_title("Ventas medias con y sin promoción por familia (Tienda 1)")
ax.set_xlabel("Familia")
ax.set_ylabel("Ventas medias diarias")
ax.tick_params(axis="x", rotation=0)
ax.legend()

plt.tight_layout()

# Guardamos la gráfica
plt.savefig(r"C:\Users\afcan\proyectos\prevision_ventas\graficas\Demand_forecasting_Grafico3_Ventas_con_sin_promocion.png", dpi=150)

plt.show()


# --- Gráfica 4: Ventas por mes y por día de la semana ---
# Extraemos el mes y el día de la semana de la columna date
# Esto nos permite agrupar las ventas por estos periodos
df["mes"] = df["date"].dt.month
df["dia_semana"] = df["date"].dt.dayofweek  # 0=lunes, 6=domingo

# Calculamos la media de ventas por mes y por día de la semana
medias_mes = df.groupby("mes")["sales"].mean()
medias_dia = df.groupby("dia_semana")["sales"].mean()

# Creamos dos subgráficas una encima de la otra
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# --- Subgráfica 1: ventas por mes ---
medias_mes.plot(kind="bar", ax=ax1, color="steelblue")
ax1.set_title("Ventas medias por mes")
ax1.set_xlabel("Mes")
ax1.set_ylabel("Ventas medias")
# Ponemos los nombres de los meses en el eje X
ax1.set_xticklabels(["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                      "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"], rotation=0)
for container in ax1.containers:
    ax1.bar_label(container, fmt="%.0f", padding=3, fontweight="bold")

# --- Subgráfica 2: ventas por día de la semana ---
medias_dia.plot(kind="bar", ax=ax2, color="coral")
ax2.set_title("Ventas medias por día de la semana")
ax2.set_xlabel("Día")
ax2.set_ylabel("Ventas medias")
# Ponemos los nombres de los días en el eje X
ax2.set_xticklabels(["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"], rotation=0)
for container in ax2.containers:
    ax2.bar_label(container, fmt="%.0f", padding=3, fontweight="bold")

plt.tight_layout()

# Guardamos la gráfica
plt.savefig(r"C:\Users\afcan\proyectos\prevision_ventas\graficas\Demand_forecasting_Grafico4_Ventas_por_mes_y_dia.png", dpi=150)

plt.show()