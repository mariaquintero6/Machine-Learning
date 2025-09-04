import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df_raw = pd.read_csv("Frenos/falla_frenos.csv", header=None)
df_split = df_raw[0].str.split(",", expand=True)

# La primera fila son los nombres de las columnas
df_split.columns = df_split.iloc[0]  

# Quitar esa primera fila de los datos
df_split = df_split.drop(0).reset_index(drop=True)

# Convertir a numérico donde aplique
df_split = df_split.apply(pd.to_numeric, errors="ignore")

print(df_split.head())
print(df_split.info())
print(df_split.isnull().sum())

df_new = df_split.drop_duplicates()
df_new.info()
print(df_new.head())

df_new.to_csv("Frenos/falla_frenos_limpio.csv", index=False)
