import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("Frenos/falla_frenos_limpio.csv")

X = df[[
    "kms_recorridos",
    "años_uso",
    "ultima_revision",
    "temperatura_frenos",
    "cambios_pastillas",
    "estilo_conduccion",
    "carga_promedio",
    "luz_alarma_freno"
]]

Y = df["falla_frenos"]

modelo = LogisticRegression()
modelo.fit(X, Y)

prediccion = modelo.predict(X)

for i, fila in df.iterrows():
    esperado = "no falla" if fila["falla_frenos"] == 1 else "falla"
    predicho = "no falla" if prediccion[i] == 1 else "falla"
    print(f"Carro {i+1}: Esperado: {esperado} | Predicho: {predicho}")
    
precision = modelo.score(X, Y)
print(f"Precisión del modelo: {precision*100:.1f}%")

nuevo_carro = [[50000, 3, 6, 75, 2, 3, 500, 0]]
prediccion_nuevo = modelo.predict(nuevo_carro)
resultado_nuevo = "no falla" if prediccion_nuevo[0] == 0 else "falla"
print(f"Predicción para el nuevo carro: {resultado_nuevo}")

joblib.dump(modelo, "Frenos/modelo_falla_frenos.pkl")
joblib.dump(X.columns.to_list(), "Frenos/columnas_modelo_falla_frenos.pkl")
