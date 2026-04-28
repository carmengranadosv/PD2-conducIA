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
from contextlib import asynccontextmanager
from starlette.middleware.sessions import SessionMiddleware

# --- 1. CONFIGURACIÓN DE RENDIMIENTO ---
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_ALLOC_FRACTION"] = ".10" 

# --- 2. DEFINICIÓN DE RUTAS Y DIRECTORIOS ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "modelos_finales"
DATA_FULL_PATH = Path("/app/data/processed/tlc_clean/datos_final.parquet")
P2_FEATURES_DIR = Path("/app/data/processed/tlc_clean/problema2/features")

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
    df_historico = next(dataset.iter_batches(batch_size=100000)).to_pandas()
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

def obtener_contexto_zona(zona_id, h_int):
    try:
        mask = (df_historico["origen_id"] == zona_id) & (df_historico["hora"] == h_int)
        filtrado = df_historico[mask]
        if not filtrado.empty: return filtrado.iloc[0]
        filtrado_z = df_historico[df_historico["origen_id"] == zona_id]
        if not filtrado_z.empty: return filtrado_z.iloc[0]
    except: pass
    return None

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
    t = procesar_tiempo_despliegue(planificacion_hora)
    info = obtener_contexto_zona(zona_id, t["hora_int"])
    oferta = float(info["oferta_inferida"]) if info is not None else 1.0
    temp = float(info["temp_c"]) if info is not None else 15.0
    
    # P1
    df_p1 = pd.DataFrame([{
        "origen_id": zona_id, "hora": t["hora_int"], "dia_semana": t["dia_semana"], "dia_mes": 15, "mes_num": 6,
        "es_finde": t["es_fin_semana"], "demanda": oferta, "lag_1h": oferta, "lag_2h": oferta, "lag_3h": oferta,
        "lag_6h": oferta, "lag_12h": oferta, "lag_24h": oferta, "roll_mean_3h": oferta, "roll_std_3h": 0.1,
        "roll_mean_24h": oferta, "roll_std_24h": 0.1, "media_hist": oferta, "temp_c": temp,
        "precipitation": 0.0, "viento_kmh": 10.0, "velocidad_mph": 10.0, "lluvia": 0, "nieve": 0, "es_festivo": 0, "num_eventos": 0
    }])
    demanda_pred = float(app.modelo_p1.predict(df_p1)[0])

    # P2
    cols_p2 = ['demanda_p1', 'temp_c', 'precipitation', 'viento_kmh', 'lluvia', 'nieve', 'es_festivo', 'num_eventos', 'hora', 'dia_semana', 'es_finde', 'mes_num', 'lag_1h', 'roll_mean_3h', 'roll_std_3h', 'roll_mean_24h', 'media_hist', 'velocidad_mph', 'zona_enc']
    z_enc = app.encoder_p2.transform([zona_id])[0]
    df_p2 = pd.DataFrame(0.1, index=[0], columns=cols_p2)
    df_p2.update(pd.DataFrame([{"demanda_p1": demanda_pred, "temp_c": temp, "hora": t["hora_int"], "dia_semana": t["dia_semana"], "zona_enc": z_enc}]))
    prob = float(app.modelo_p2.predict(app.scaler_p2.transform(df_p2.values), verbose=0)[0][0])

    res = {"demanda_estimada": round(demanda_pred, 1), "exito_prob": round(prob*100, 1), "recomendacion": "ALTA" if prob > 0.6 else "MODERADA"}
    return templates.TemplateResponse(request=request, name="taxi.html", context={"request": request, "mostrar_res": True, "resultado": res, "zona_seleccionada": zona_id, "hora_seleccionada": planificacion_hora})

@app.post("/vtc", response_class=HTMLResponse)
async def predict_vtc(request: Request, origen_id: int = Form(...), destino_id: int = Form(...), precio_base: float = Form(...), tipo_vehiculo: int = Form(...), planificacion_hora: str = Form("actual")):
    t = procesar_tiempo_despliegue(planificacion_hora)
    
    # P4
    z_idx = app.encoder_p4.transform([origen_id])
    c_num = app.scaler_p4.transform([[15.0, 0.0, 10.0, 0, 0, 0, 0, 0, 0, t["dia_semana"], t["es_fin_semana"], t["hora_sen"], t["hora_cos"]]])
    vel = float(app.modelo_p4.predict([z_idx, c_num], verbose=0)[0][0])

    # P5
    df_p5 = pd.DataFrame([{
        "tipo_vehiculo": int(tipo_vehiculo), "origen_zona": "Midtown Center", "origen_barrio": "Manhattan", "evento_tipo": "Ninguno", "franja_horaria": "tarde", "precio_base": float(precio_base), "hora_sen": float(t["hora_sen"]), "hora_cos": float(t["hora_cos"]), "es_fin_semana": int(t["es_fin_semana"]),
        "num_pasajeros": 1, "velocidad_mph": vel, "lluvia": 0, "temp_c": 15.0, "es_festivo": 0, "distancia": 3.0, "nieve": 0, "espera_min": 1.0, "precipitation": 0.0, "trafico_denso": 0, "duracion_min": 10.0, "num_eventos": 0, "mes_num": 6, "hora": t["hora_int"], "rentabilidad_base_min": 1.0, "viento_kmh": 10.0, "precio_total_est": precio_base + 2.0, "dia_semana": t["dia_semana"]
    }])
    propina = float(app.modelo_p5.predict(df_p5)[0])

    res = {"propina": round(propina, 2), "velocidad": round(vel, 1), "encadenado": "ALTA", "decision": "ACEPTAR" if propina > 2 else "RECHAZAR", "detalles": "Análisis OK"}
    return templates.TemplateResponse(request=request, name="vtc.html", context={"request": request, "resultado": res, "seleccion": {"origen": origen_id, "destino": destino_id, "precio": precio_base, "vehiculo": tipo_vehiculo, "hora": planificacion_hora}})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)