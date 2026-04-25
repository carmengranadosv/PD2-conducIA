import os
from pathlib import Path
from datetime import datetime
import joblib
import keras
import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import asynccontextmanager
from starlette.middleware.sessions import SessionMiddleware

# --- 1. CONFIGURACIÓN DE RENDIMIENTO ---
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_ALLOC_FRACTION"] = ".10" 

# --- 2. DEFINICIÓN DE RUTAS Y DIRECTORIOS ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "modelos_finales"
DATA_PATH = Path("/app/data/processed/tlc_clean/datos_final.parquet")

# --- 3. CARGA DE MODELOS (LIFESPAN) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga los modelos una sola vez al iniciar."""
    try:
        app.modelo_p1 = joblib.load(MODEL_DIR / "modelo_p1_rf.joblib")
        app.modelo_p2 = keras.models.load_model(MODEL_DIR / "modelo_p2_mlp.keras")
        app.scaler_p2 = joblib.load(MODEL_DIR / "modelo_p2_mlp_scaler.pkl")
        app.encoder_p2 = joblib.load(MODEL_DIR / "modelo_p2_zona_encoder.pkl")
        app.modelo_p4 = keras.models.load_model(MODEL_DIR / "modelo_p4_red_neuronal.keras")
        app.scaler_p4 = joblib.load(MODEL_DIR / "modelo_p4_scaler_clima.joblib")
        app.encoder_p4 = joblib.load(MODEL_DIR / "modelo_p4_label_encoder_zonas.joblib")
        app.modelo_p5 = joblib.load(MODEL_DIR / "modelo_p5_xgboost.joblib")
        print("✅ Modelos cargados correctamente.")
    except Exception as e:
        print(f"❌ Error cargando modelos: {e}")
    yield

# --- 4. INICIALIZACIÓN DE LA APP ---
app = FastAPI(lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key="conducia_secret_key")

static_path = BASE_DIR / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
else:
    @app.get("/static/{path:path}")
    async def static_fallback():
        return {"detail": "Static folder not found"}

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# --- 5. CARGA DE DATOS OPTIMIZADA ---
try:
    import pyarrow.parquet as pq
    dataset = pq.ParquetFile(DATA_PATH)
    df_historico = next(dataset.iter_batches(batch_size=50000)).to_pandas()
    print(f"✅ Dataset cargado correctamente.")
except Exception as e:
    print(f"⚠️ Error cargando dataset: {e}")
    df_historico = pd.DataFrame()

# --- 6. FUNCIONES DE APOYO ---
def procesar_tiempo_despliegue(hora_usuario: str = "actual"):
    if hora_usuario == "actual" or not hora_usuario:
        ahora = datetime.now()
        hora_float = ahora.hour + ahora.minute / 60.0
        dia_semana = ahora.weekday()
    else:
        hora_float = float(hora_usuario)
        dia_semana = datetime.now().weekday() 

    hora_sen = np.sin(2 * np.pi * hora_float / 24)
    hora_cos = np.cos(2 * np.pi * hora_float / 24)
    es_fin_semana = 1 if dia_semana >= 5 else 0
    
    return {
        "hora_int": int(hora_float), 
        "hora_sen": hora_sen, "hora_cos": hora_cos,
        "dia_semana": dia_semana, "es_fin_semana": es_fin_semana
    }

# --- 7. RUTAS GET (NAVEGACIÓN) ---
@app.get("/", response_class=HTMLResponse)
async def pantalla_inicio(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/taxi", response_class=HTMLResponse)
async def panel_taxi(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="taxi.html", 
        context={
            "request": request, 
            "zona_seleccionada": None, 
            "hora_seleccionada": "actual", 
            "mostrar_res": False
        }
    )

@app.get("/vtc", response_class=HTMLResponse)
async def panel_vtc(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="vtc.html", 
        context={
            "request": request, 
            "seleccion": {}, 
            "resultado": None 
        }
    )

@app.get("/documentacion", response_class=HTMLResponse)
async def pantalla_doc(request: Request):
    return templates.TemplateResponse(request=request, name="documentacion.html")

# --- 8. LÓGICA POST (PREDICCIONES) ---
@app.post("/taxi", response_class=HTMLResponse)
async def predict_taxi(request: Request, zona_id: int = Form(...), planificacion_hora: str = Form("actual")):
    tiempo = procesar_tiempo_despliegue(planificacion_hora)
    
    # Lógica de ejemplo para que los datos varíen según la zona
    if zona_id == 132: # JFK
        resultado = {"demanda_estimada": 85, "exito_prob": 94, "recomendacion": "ZONA CRÍTICA - ALTA DEMANDA"}
    else:
        resultado = {"demanda_estimada": 30, "exito_prob": 65, "recomendacion": "DEMANDA MODERADA"}

    return templates.TemplateResponse(
        request=request,
        name="taxi.html",
        context={
            "request": request, 
            "mostrar_res": True, 
            "resultado": resultado,
            "zona_seleccionada": zona_id,
            "hora_seleccionada": planificacion_hora
        }
    )

@app.post("/vtc", response_class=HTMLResponse)
async def predict_vtc(request: Request, 
                      origen_id: int = Form(...), 
                      destino_id: int = Form(...), 
                      precio_base: float = Form(...), 
                      tipo_vehiculo: int = Form(...), 
                      planificacion_hora: str = Form("actual")):
    
    tiempo = procesar_tiempo_despliegue(planificacion_hora)

    # Lógica de ejemplo para que varíe según el precio y origen
    propina_est = round(precio_base * 0.18, 2) if origen_id == 132 else round(precio_base * 0.12, 2)
    
    resultado_VTC = {
        "propina": propina_est,
        "velocidad": "24.5" if planificacion_hora == "actual" else "15.8",
        "encadenado": "ALTA" if precio_base > 20 else "BAJA",
        "detalles": f"Analizando trayecto desde zona {origen_id} a {destino_id}.",
        "decision": "ACEPTAR" if precio_base > 15 else "RECHAZAR"
    }

    return templates.TemplateResponse(
        request=request,
        name="vtc.html",
        context={
            "request": request, 
            "resultado": resultado_VTC, 
            "seleccion": {
                "origen": origen_id,
                "destino": destino_id,
                "precio": precio_base,
                "vehiculo": tipo_vehiculo,
                "hora": planificacion_hora
            }
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)