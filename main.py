from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

modelo = joblib.load("modelo_falla_frenos.pkl")
columnas = joblib.load("columnas_modelo_falla_frenos.pkl")

app = FastAPI(title="Bienvenido a la API de predicción de Fallas en Frenos")

class Carro(BaseModel):
    kms_recorridos: int
    años_uso: int
    ultima_revision: int
    temperatura_frenos: int
    cambios_pastillas: int
    estilo_conduccion: int
    carga_promedio: int
    luz_alarma_freno: int
    
@app.post("/predict/")
def predict(data: Carro):
    x_new = pd.DataFrame([[data.kms_recorridos, data.años_uso, data.ultima_revision,
                           data.temperatura_frenos, data.cambios_pastillas, data.estilo_conduccion,
                           data.carga_promedio, data.luz_alarma_freno]], columns=columnas)
    prediccion = modelo.predict(x_new)
    if prediccion[0] == 0:
        mensaje = "NO falla"
    else:
        mensaje = "SI falla"
    return {"mensaje_prediccion": mensaje}
    
@app.get("/")
def home():
    return {"message": "Bienvenido a la API de predicción de Fallas en Frenos"}