import os
# Usamos JAX para las redes neuronales
os.environ["KERAS_BACKEND"] = "jax"
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.concurrency import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import keras
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Annotated
import uvicorn
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from pathlib import Path

# Ruta base
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "modelos_finales"
DATA_PATH = BASE_DIR.parent / "data/processed/tlc_clean/datos_final.parquet"

# Cargar modelo al iniciar el servidor
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Cargamos los modelos de la carpeta modelos_finales
    app.modelo_p1 = joblib.load(MODEL_DIR / "modelo_p1_rf.joblib")
    app.modelo_p2 = keras.models.load_model(MODEL_DIR / "modelo_p2_mlp.keras")
    app.scaler_p2 = joblib.load(MODEL_DIR / "modelo_p2_mlp_scaler.pkl")
    app.encoder_p2 = joblib.load(MODEL_DIR / "modelo_p2_zona_encoder.pkl")
    app.modelo_p4 = keras.models.load_model(MODEL_DIR / "modelo_p4_red_neuronal.keras")
    app.scaler_p4 = joblib.load(MODEL_DIR / "modelo_p4_scaler_clima.joblib")
    app.encoder_p4 = joblib.load(MODEL_DIR / "modelo_p4_label_encoder_zonas.joblib")
    app.modelo_p5 = joblib.load(MODEL_DIR / "modelo_p5_xgboost.joblib")
    yield

# Instancia de la app
app = FastAPI(lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key="conducia_secret_key")

# Configuración de estáticos y plantillas
# Servir los archivos HTML (index.html, taxi.html, etc)
if (BASE_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory="templates")

# 2. Carga de Datos complementarios 
# Necesitamos los lags y medias históricas para que los modelos de los problemas 1 y 2 funcionen
try:
    df_historico = pd.read_parquet(DATA_PATH)
except Exception as e:
    print(f"Advertencia: No se pudo cargar el dataset histórico en {DATA_PATH}: {e}")
    df_historico = pd.DataFrame()


# 4. FUNCIONES DE APOYO (Lógica temporal)
def procesar_tiempo_despliegue(hora_usuario: int = None):
    """
    Si hora_usuario es None, usa el tiempo real.
    Si no, usa la hora seleccionada por el usuario.
    """
    if hora_usuario is None or hora_usuario == "actual":
        ahora = datetime.now()
        hora_float = ahora.hour + ahora.minute / 60.0
        dia_semana = ahora.weekday()
    else:
        hora_float = float(hora_usuario)
        # Para planificación, asumimos el día actual o podrías pedir fecha
        dia_semana = datetime.now().weekday() 

    hora_sen = np.sin(2 * np.pi * hora_float / 24)
    hora_cos = np.cos(2 * np.pi * hora_float / 24)
    es_fin_semana = 1 if dia_semana >= 5 else 0
    
    return {
        "hora": int(hora_float),
        "hora_sen": hora_sen,
        "hora_cos": hora_cos,
        "dia_semana": dia_semana,
        "es_fin_semana": es_fin_semana
    }

# RUTAS DE NAVEGACIÓN
@app.get("/", response_class=HTMLResponse)
async def pantalla_inicio(request: Request):
    return templates.TemplateResponse("inicio.html", {"request": request})

@app.get("/taxi", response_class=HTMLResponse)
async def panel_taxi(request: Request):
    return templates.TemplateResponse("taxi.html", {"request": request})

@app.get("/vtc", response_class=HTMLResponse)
async def panel_vtc(request: Request):
    return templates.TemplateResponse("vtc.html", {"request": request})

@app.get("/documentacion", response_class=HTMLResponse)
async def pantalla_doc(request: Request):
    return templates.TemplateResponse("documentacion.html", {"request": request})

# LÓGICA DE PREDICCIÓN  DE TAXI

# 5. ENDPOINTS
# Endpoint para TAXI
@app.post("/taxi", response_class=HTMLResponse)
async def predict_taxi(
    request: Request,
    zona_id: Annotated[int, Form(...)],
    planificacion_hora: Annotated[str, Form(None)] = "actual"
):
    # 1. Obtener tiempo
    tiempo = procesar_tiempo_despliegue(planificacion_hora)

    # 2. Buscar contexto histórico (Lags)
    # Buscamos en el dataframe la fila que coincide con la zona actual
    try:
        mask = (df_historico["origen_id"] == zona_id) & (df_historico["hora"] == tiempo["hora_int"])
        info_zona = df_historico[mask].iloc[0]
    except IndexError:
        # Si no hay datos exactos, buscamos la media de la zona
        info_zona = df_historico[df_historico["origen_id"] == zona_id].iloc[0]
    
    # 3. Construir la fila para el modelo (Respetando el orden del CSV inicial)
    entrada_p1 = pd.DataFrame([{
        "origen_id": zona_id,
        "hora_sen": tiempo["hora_sen"],
        "hora_cos": tiempo["hora_cos"],
        "es_fin_semana": tiempo["es_fin_semana"],
        "dia_semana": tiempo["dia_semana"],
        "temp_c": info_zona["temp_c"] if info_zona is not None else 15.0,
        "precipitation": info_zona["precipitation"] if info_zona is not None else 0.0,
        "oferta_inferida": info_zona["oferta_inferida"] if info_zona is not None else 1.0, # Este suele ser el Lag principal
        "tasa_historica": info_zona["tasa_historica"] if info_zona is not None else 0.5
    }])
    
    # 4. Predicción
    # Necesitamos el encoder y el scaler de P2
    demanda_pred = app.modelo_p1.predict(entrada_p1)[0]
    zona_enc = app.encoder_p2.transform([zona_id])[0]

    # Construcción del vector según el reentreno de P2
    entrada_p2_raw = np.array([[demanda_pred, info_zona["temp_c"], zona_enc]]) 
    entrada_p2_scaled = app.scaler_p2.transform(entrada_p2_raw)
    prob_exito = app.modelo_p2.predict(entrada_p2_scaled)[0][0]

    resultado = {
        "demanda_estimada": round(float(demanda_pred), 2),
        "exito_prob": round(float(prob_exito * 100), 2),
        "recomendacion": "QUÉDATE" if prob_exito > 0.6 else "MUÉVETE",
        "hora_analizada": f"{tiempo['hora_int']}:00"
    }
    
    return templates.TemplateResponse("taxi.html", {
        "request": request, 
        "resultado": resultado, 
        "mostrar_res": True
    })

# LÓGICA DE PREDICCIÓN DE VTC

# Endpoint para VTC
@app.post("/vtc", response_class=HTMLResponse)
async def predict_vtc(
    request: Request,
    origen_id: Annotated[int, Form(...)],
    destino_id: Annotated[int, Form(...)],
    precio_base: Annotated[float, Form(...)],
    tipo_vehiculo: Annotated[int, Form(...)],
    planificacion_hora: Annotated[str, Form(None)] = "actual"
):
    # 1. Datos temporales
    tiempo = procesar_tiempo_despliegue(planificacion_hora)

    # 2. Predicción P4: Tráfico
    # Para el modelo de tráfico, necesitamos la temperatura y la zona codificada
    zona_idx = app.encoder_p4.transform([origen_id])
    # Clima y tiempo para P4
    clima_vals = np.array([[
        15.0, 0.0, 10.0, 0, 0, 0, 0, 0, 0, 
        tiempo["dia_semana"], tiempo["es_fin_semana"], 
        tiempo["hora_sen"], tiempo["hora_cos"]
    ]])
    clima_scaled = app.scaler_p4.transform(clima_vals)

    # Predicción con lista de dos inputs [Embeddings, Numéricas]
    velocidad = app.modelo_p4.predict([zona_idx, clima_scaled])[0][0]
    
    # 3. Construir entrada para XGBoost
    # Como tiene un pipeline con preprocesado, ya se gestiona internamente el escalado y encoding, pero debemos respetar las columnas
    entrada_p5 = pd.DataFrame([{
        "tipo_vehiculo": tipo_vehiculo,
        "origen_id": origen_id,
        "destino_id": destino_id,
        "precio_base": precio_base,
        "hora_sen": tiempo["hora_sen"],
        "hora_cos": tiempo["hora_cos"],
        "es_fin_semana": tiempo["es_fin_semana"]
    }])
    
    propina_estimada = app.modelo_p5.predict(entrada_p5)[0]


    # 4. Predicción de Viaje encadenado (Lógica en Cascada)
    # IMPORTANTE: Evaluamos la zona de DESTINO para ver si habrá trabajo allí
    # Primero: Ejecutamos el Modelo 1 para predecir la demanda en el DESTINO
    entrada_p1_destino = pd.DataFrame([{
        "origen_id": destino_id, # Evaluamos la zona donde dejarás al pasajero
        "hora_sen": tiempo["hora_sen"],
        "hora_cos": tiempo["hora_cos"],
        "es_fin_semana": tiempo["es_fin_semana"],
        "dia_semana": tiempo["dia_semana"],
        "temp_c": 15.0, # O usar info_clima si tenéis la función de API
        "precipitation": 0.0,
        "oferta_inferida": 1.0, 
        "tasa_historica": 0.5
    }])
    
    # Predecimos demanda real en destino con Modelo 1
    demanda_pred_destino = app.modelo_p1.predict(entrada_p1_destino)[0]

    # Segundo: Usamos esa demanda predicha para alimentar el Modelo 2 (MLP)
    dest_enc = app.encoder_p2.transform([destino_id])[0]
    input_p2_dest = np.array([[demanda_pred_destino, 15.0, dest_enc]]) # demanda_p1, temp, zona
    input_p2_scaled = app.scaler_p2.transform(input_p2_dest)
    
    # Probabilidad final de éxito en la zona de destino
    prob_exito_destino = app.modelo_p2.predict(input_p2_scaled)[0][0]
    

    # 5. Lógica de la decisión final
    es_rentable = True
    razones = []

    if propina_estimada < (precio_base * 0.10):
        razones.append("Propina baja estimada")
        es_rentable = False

    if velocidad < 8: # Menos de 8 mph es mucho atasco
        razones.append("Tráfico intenso en la ruta")
        es_rentable = False

    if prob_exito_destino < 0.4:
        razones.append("Zona de destino con baja demanda (posible retorno vacío)")
        es_rentable = False
    else:
        razones.append("Alta probabilidad de viaje encadenado en destino")

    decision_final = "ACEPTAR" if es_rentable or prob_exito_destino > 0.7 else "RECHAZAR"

    resultado = {
        "decision": decision_final,
        "propina": round(float(propina_estimada), 2),
        "velocidad": round(float(velocidad), 2),
        "encadenado": "Muy probable" if prob_exito_destino > 0.6 else "Poco probable",
        "detalles": " | ".join(razones)
    }

    
    return templates.TemplateResponse("vtc.html", {
        "request": request, 
        "resultado": resultado, 
        "mostrar_res": True
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)