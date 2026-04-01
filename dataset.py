# --- Librerías ---
import pandas as pd      
from pathlib import Path  

# --- Configuracion ---
# Carpeta donde están los CSVs originales
DATA_DIR = Path(r"C:\Users\afcan\proyectos\prevision_ventas\datos")
# Archivo de salida donde guardaremos el dataset ya limpio
OUTPUT_FILE = DATA_DIR / "dataset_limpio.csv"

# Familias de productos que nos interesan
FAMILIAS = ["DAIRY", "BEVERAGES", "GROCERY I"]
# Solo analizamos una tienda para simplificar
TIENDA = 1
# Descartamos datos anteriores a 2016 por ser menos relevantes
FECHA_INICIO = "2016-01-01"

# --- Carga ---
print("Cargando datos...")
# Leemos el CSV principal con todas las ventas (3 millones de filas)
train = pd.read_csv(DATA_DIR / "train.csv")
print(f"  train: {len(train):,} filas")

# --- Filtros ---
# Convertimos la columna date de texto a formato fecha real
# Sin esto Python no sabría ordenar ni comparar fechas correctamente
train["date"] = pd.to_datetime(train["date"])

# Aplicamos tres filtros a la vez con & (significa "y"):
# 1. Solo las familias que nos interesan
# 2. Solo la tienda 1
# 3. Solo fechas desde 2016
df = train[
    (train["family"].isin(FAMILIAS)) &
    (train["store_nbr"] == TIENDA) &
    (train["date"] >= FECHA_INICIO)
].copy()

# Eliminamos la columna id porque no aporta nada al análisis
df = df.drop(columns=["id"])

# Renombramos "GROCERY I" a "GROCERY" para simplificar
df["family"] = df["family"].replace("GROCERY I", "GROCERY")


print(f"  Tras filtros: {len(df):,} filas")

# --- Guardado ---
# Guardamos el dataset limpio como CSV
# index=False evita que pandas añada una columna extra con números de fila
df.to_csv(OUTPUT_FILE, index=False)
print(f"Dataset guardado: {OUTPUT_FILE}")

# Mostramos las primeras 5 filas para verificar que todo está bien
print(df.head())

#########################################################################################
#Lo que tenemos ahora es un dataset de una sola tienda, de 3 elementos (Dairy, Groceries y Beverages), y desde 2016 para adelante.
#El dataset incluye el número de ventas de cada elemento por cada día, Para el 01/01/2016 --> X Beverages, X Groceries y X Dairy. 
#########################################################################################