from fastapi import FastAPI, Request
from fastapi.concurrency import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware


# Ruta base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Cargar modelo al iniciar el servidor
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Cargamos los modelos de la carpeta modelos_finales
    app.modelo_p1 = joblib.load("modelos_finales/modelo_p1_rf.joblib")
    app.modelo_p5 = joblib.load("modelos_finales/modelo_p5_xgboost.joblib")
    yield

# Instancia de la app
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

app.add_middleware(SessionMiddleware, secret_key="una_clave_secreta")



# 2. Carga de Datos complementarios 
# Necesitamos los lags y medias históricas para que los modelos de los problemas 1 y 2 funcionen
df_historico = pd.read_parquet("data/processed/tlc_clean/datos_final.parquet")

# 3. Estructura de datos que esperamos de la web
class DatosTaxi(BaseModel):
    zona_id: int

class DatosVTC(BaseModel):
    origen_id: int
    destino_id: int
    precio_base: float
    tipo_vehiculo: int # 1 para Taxi, 2 para VTC 

# 4. FUNCIONES DE APOYO (Lógica temporal)
def obtener_caracteristicas_tiempo():
    ahora = datetime.now()
    hora = ahora.hour
    # Convertir a seno/coseno como pide vuestro modelo
    hora_sen = np.sin(2 * np.pi * hora / 24)
    hora_cos = np.cos(2 * np.pi * hora / 24)
    dia_semana = ahora.weekday()
    es_fin_de_semana = 1 if dia_semana >= 5 else 0
    return hora_sen, hora_cos, es_fin_de_semana

# 5. ENDPOINTS
# Endpoint para TAXI
@app.post("/api/taxi")
async def predict_taxi(datos: DatosTaxi):
    # a. Obtener datos temporales
    h_sen, h_cos, es_finge = obtener_caracteristicas_tiempo()
    
    # b. Buscar datos históricos de la zona (Lags)
    # Buscamos en el dataframe la fila que coincide con la zona actual
    info_zona = df_historico[df_historico["origen_id"] == datos.zona_id].iloc[0]
    
    # c. Construir la fila para el modelo (Respetando el orden del CSV inicial)
    entrada = pd.DataFrame([{
        "PULocationID": datos.zona_id,
        "hora_sen": h_sen,
        "hora_cos": h_cos,
        "es_finge": es_finge,
        "demanda_media": info_zona["demanda_media"],
        "lag_1h": info_zona["demanda_mediana"], # Ejemplo de mapeo
        "temp": 15.0, # Aquí podríais conectar una API de clima real
        "precipitacion": 0.0
    }])
    
    # d. Predicción
    pred_demanda = modelo_taxi.predict(entrada)[0]
    prob_exito = clasificador_exito.predict_proba(entrada)[0][1] # Probabilidad de "Punto Caliente"
    
    return {
        "demanda_estimada": round(float(pred_demanda), 2),
        "exito_prob": round(float(prob_exito * 100), 2),
        "recomendacion": "QUÉDATE" if prob_exito > 0.6 else "MUÉVETE"
    }

# Endpoint para VTC
@app.post("/api/vtc")
async def predict_vtc(datos: DatosVTC):
    # a. Datos temporales
    h_sen, h_cos, es_finge = obtener_caracteristicas_tiempo()
    
    # b. Construir entrada para XGBoost
    # Importante: Aquí aplicamos la "Ceguera de destino" no incluyendo DOLocationID si el modelo no lo usa
    entrada = pd.DataFrame([{
        "vendor_id": datos.tipo_vehiculo,
        "PULocationID": datos.origen_id,
        "fare_amount": datos.precio_base,
        "hora_sen": h_sen,
        "hora_cos": h_cos,
        "es_finge": es_finge,
        "temp": 15.0
    }])
    
    # c. Predicción de propina
    propina_estimada = modelo_vtc.predict(entrada)[0]
    
    return {
        "propina_esperada": round(float(propina_estimada), 2),
        "rentabilidad": "ALTA" if propina_estimada > (datos.precio_base * 0.15) else "MEDIA/BAJA"
    }

# 6. Servir los archivos HTML (index.html, taxi.html, etc)
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)