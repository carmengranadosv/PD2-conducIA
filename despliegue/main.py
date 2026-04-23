import os
from pathlib import Path
from datetime import datetime

# --- 1. CONFIGURACIÓN DE RENDIMIENTO (EVITA EL ERROR 'KILLED') ---
# Usamos JAX como motor para Keras
os.environ["KERAS_BACKEND"] = "jax"
# Forzamos a JAX a no acaparar la RAM para que Docker no mate el proceso
os.environ["XLA_PYTHON_CLIENT_ALLOC_FRACTION"] = ".10" 

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

# --- 2. DEFINICIÓN DE RUTAS Y DIRECTORIOS ---
# BASE_DIR será la carpeta 'despliegue'
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "modelos_finales"
# Ruta donde Docker monta el volumen de datos
DATA_PATH = Path("/app/data/processed/tlc_clean/datos_final.parquet")

# --- 3. CARGA DE MODELOS (LIFESPAN) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga los modelos una sola vez al iniciar para ahorrar tiempo y memoria."""
    app.modelo_p1 = joblib.load(MODEL_DIR / "modelo_p1_rf.joblib")
    app.modelo_p2 = keras.models.load_model(MODEL_DIR / "modelo_p2_mlp.keras")
    app.scaler_p2 = joblib.load(MODEL_DIR / "modelo_p2_mlp_scaler.pkl")
    app.encoder_p2 = joblib.load(MODEL_DIR / "modelo_p2_zona_encoder.pkl")
    app.modelo_p4 = keras.models.load_model(MODEL_DIR / "modelo_p4_red_neuronal.keras")
    app.scaler_p4 = joblib.load(MODEL_DIR / "modelo_p4_scaler_clima.joblib")
    app.encoder_p4 = joblib.load(MODEL_DIR / "modelo_p4_label_encoder_zonas.joblib")
    app.modelo_p5 = joblib.load(MODEL_DIR / "modelo_p5_xgboost.joblib")
    yield

# --- 4. INICIALIZACIÓN DE LA APP ---
app = FastAPI(lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key="conducia_secret_key")

# Configuramos la ruta hacia la carpeta static
# Según tu imagen, debería estar en /app/despliegue/static
static_path = BASE_DIR / "static"

if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    print(f"✅ Carpeta static encontrada y montada.")
else:
    # Creamos una ruta vacía para 'static' para que url_for() no de error 500
    # aunque no tengamos archivos dentro.
    @app.get("/static/{path:path}")
    async def static_fallback():
        return {"detail": "Static folder not found"}
    print(f"⚠️ Alerta: Carpeta static no encontrada en {static_path}. Se creó un fallback.")

# Configuración de plantillas
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# --- 5. CARGA DE DATOS OPTIMIZADA (SISTEMA ANTI-RAM) ---
try:
    import pyarrow.parquet as pq
    # Leemos solo un bloque inicial para evitar cargar 5GB en la RAM del contenedor
    dataset = pq.ParquetFile(DATA_PATH)
    df_historico = next(dataset.iter_batches(batch_size=50000)).to_pandas()
    print(f"✅ ÉXITO: Dataset cargado desde {DATA_PATH}")
except Exception as e:
    print(f"⚠️ ADVERTENCIA: No se pudo cargar el dataset: {e}")
    df_historico = pd.DataFrame()

# --- 6. FUNCIONES DE APOYO ---
def procesar_tiempo_despliegue(hora_usuario: str = "actual"):
    """Prepara la hora para que los modelos la entiendan matemáticamente."""
    if hora_usuario == "actual" or hora_usuario is None:
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
# Se usa name="index.html" porque así aparece en tu estructura de archivos
@app.get("/", response_class=HTMLResponse)
async def pantalla_inicio(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})

@app.get("/taxi", response_class=HTMLResponse)
async def panel_taxi(request: Request):
    return templates.TemplateResponse(request=request, name="taxi.html", context={"request": request})

@app.get("/vtc", response_class=HTMLResponse)
async def panel_vtc(request: Request):
    return templates.TemplateResponse(request=request, name="vtc.html", context={"request": request})

@app.get("/documentacion", response_class=HTMLResponse)
async def pantalla_doc(request: Request):
    return templates.TemplateResponse(request=request, name="documentacion.html", context={"request": request})

# --- 8. LÓGICA POST (PREDICCIONES) ---
@app.post("/taxi", response_class=HTMLResponse)
async def predict_taxi(request: Request, zona_id: int = Form(...), planificacion_hora: str = Form("actual")):
    tiempo = procesar_tiempo_despliegue(planificacion_hora)
    
    # Buscamos la info de la zona de forma segura
    datos_zona = df_historico[df_historico["origen_id"] == zona_id]
    
    if not datos_zona.empty:
        # Si existe la zona, intentamos afinar por hora
        mask_hora = datos_zona["hora"] == tiempo["hora_int"]
        info_zona = datos_zona[mask_hora].iloc[0] if not datos_zona[mask_hora].empty else datos_zona.iloc[0]
        # Aquí podrías añadir tu predicción real: app.modelo_p1.predict(...)
        resultado = "Predicción realizada" 
    else:
        # Si la zona no está en el dataset de 50k filas, evitamos el error
        info_zona = None
        resultado = "Sin datos históricos para esta zona en el bloque actual"

    return templates.TemplateResponse(
        request=request, 
        name="taxi.html", 
        context={"request": request, "mostrar_res": True, "resultado": resultado}
    )

@app.post("/vtc", response_class=HTMLResponse)
async def predict_vtc(request: Request, origen_id: int = Form(...), destino_id: int = Form(...), 
                      precio_base: float = Form(...), tipo_vehiculo: int = Form(...), planificacion_hora: str = Form("actual")):
    return templates.TemplateResponse(request=request, name="vtc.html", context={"request": request, "mostrar_res": False})

# --- 9. EJECUCIÓN ---
if __name__ == "__main__":
    # Importante: host 0.0.0.0 para que sea accesible desde fuera del contenedor
    uvicorn.run(app, host="0.0.0.0", port=8000)