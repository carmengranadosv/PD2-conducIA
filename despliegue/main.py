from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

app = FastAPI()

# 1. Cargamos los modelos de la carpeta modelos_finales
modelo_p1 = joblib.load("modelos_finales/modelo_p1_rf.joblib")
modelo_p5 = joblib.load("modelos_finales/modelo_p5_xgboost.joblib")

# Estructura de datos que esperamos de la web
class DatosTaxi(BaseModel):
    zona_id: int

class DatosVTC(BaseModel):
    origen_id: int
    destino_id: int
    precio_base: float

# Endpoint para TAXI
@app.post("/api/taxi")
async def predict_taxi(datos: DatosTaxi):


# Endpoint para VTC
@app.post("/api/vtc")
async def predict_vtc(datos: DatosVTC):


# Servir los archivos HTML (index.html, taxi.html, etc)
app.mount("/", StaticFiles(directory=".", html=True), name="static")