import os
# --- 1. CONFIGURACIÓN DE RENDIMIENTO ---
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_ALLOC_FRACTION"] = ".10" 

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

# --- 2. DEFINICIÓN DE RUTAS Y DIRECTORIOS ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "modelos_finales"
DATA_ROOT = Path(os.getenv("CONDUCIA_DATA_DIR", "/app/data"))
if not DATA_ROOT.exists() and (BASE_DIR.parent / "data").exists():
    DATA_ROOT = BASE_DIR.parent / "data"

DATA_FULL_PATH = DATA_ROOT / "processed/tlc_clean/datos_final.parquet"
CONTEXT_DIR = DATA_ROOT / "processed/tlc_clean/contexto_web"
P2_CONTEXT_PATH = CONTEXT_DIR / "contexto_p2.parquet"
P5_CONTEXT_PATH = CONTEXT_DIR / "contexto_p5.parquet"
P2_FALLBACK_PATH = DATA_ROOT / "processed/tlc_clean/problema2/features/train.parquet"
P5_FALLBACK_PATH = DATA_ROOT / "processed/tlc_clean/problema5/train.parquet"

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
def cargar_contexto(path: Path, fallback_path: Path, nombre: str) -> pd.DataFrame:
    try:
        if path.exists():
            df = pd.read_parquet(path)
            print(f"✅ Contexto {nombre} cargado: {len(df):,} filas.")
            return df

        import pyarrow.parquet as pq

        dataset = pq.ParquetFile(fallback_path)
        df = next(dataset.iter_batches(batch_size=100000)).to_pandas()
        print(
            f"⚠️ No existe {path}. "
            f"Usando primer batch de {fallback_path.name} para {nombre}."
        )
        return df

    except Exception as e:
        print(f"⚠️ Error cargando contexto {nombre}: {e}")
        return pd.DataFrame()


df_historico = pd.DataFrame()
df_p2_contexto = cargar_contexto(P2_CONTEXT_PATH, P2_FALLBACK_PATH, "P2")
df_p5_contexto = cargar_contexto(P5_CONTEXT_PATH, P5_FALLBACK_PATH, "P5")

# --- 6. FUNCIONES DE APOYO ---
def procesar_tiempo_despliegue(hora_usuario: str = "actual"):
    ahora = datetime.now()

    if hora_usuario == "actual" or not hora_usuario:
        hora_float = ahora.hour + ahora.minute / 60.0
    else:
        try:
            hora_float = float(hora_usuario)
        except ValueError:
            hora_float = ahora.hour + ahora.minute / 60.0

    # Evita horas fuera de rango
    hora_float = max(0, min(hora_float, 23.99))

    dia_semana = ahora.weekday()
    hora_int = int(hora_float)

    hora_sen = float(np.sin(2 * np.pi * hora_float / 24))
    hora_cos = float(np.cos(2 * np.pi * hora_float / 24))
    es_fin_semana = 1 if dia_semana >= 5 else 0

    return {
        "hora_int": hora_int,
        "hora_sen": hora_sen,
        "hora_cos": hora_cos,
        "dia_semana": dia_semana,
        "dia_mes": ahora.day,
        "mes_num": ahora.month,
        "es_fin_semana": es_fin_semana,
        "es_finde": es_fin_semana,
    }

def resumir_contexto(filtrado: pd.DataFrame):
    """Resume varias filas historicas en una sola fila compatible con valor_contexto."""
    if filtrado.empty:
        return None

    if len(filtrado) == 1:
        return filtrado.iloc[0]

    resumen = {}
    for col in filtrado.columns:
        serie = filtrado[col].dropna()
        if serie.empty:
            resumen[col] = np.nan
        elif pd.api.types.is_numeric_dtype(serie):
            resumen[col] = serie.mean()
        else:
            resumen[col] = serie.mode().iloc[0]

    return pd.Series(resumen)

def buscar_contexto_historico(df: pd.DataFrame, zona_id, h_int, mes_num, dia_semana):
    """Busca contexto historico comparable, de mas especifico a mas general."""
    try:
        if df.empty:
            return None

        zona_id = int(zona_id)
        h_int = int(h_int)
        mes_num = int(mes_num)
        dia_semana = int(dia_semana)

        filtros = [
            {"origen_id": zona_id, "mes_num": mes_num, "dia_semana": dia_semana, "hora": h_int},
            {"origen_id": zona_id, "mes_num": mes_num, "hora": h_int},
            {"origen_id": zona_id, "dia_semana": dia_semana, "hora": h_int},
            {"origen_id": zona_id, "hora": h_int},
            {"origen_id": zona_id},
        ]

        for filtro in filtros:
            mask = pd.Series(True, index=df.index)
            for col, valor in filtro.items():
                if col not in df.columns:
                    mask = pd.Series(False, index=df.index)
                    break
                mask &= df[col].astype(int) == int(valor)

            filtrado = df.loc[mask]
            if not filtrado.empty:
                return resumir_contexto(filtrado)

    except Exception as e:
        print("Error buscando contexto historico:", e)

    return None

def obtener_contexto_zona(zona_id, t):
    """Busca contexto historico P2 para taxi."""
    return buscar_contexto_historico(
        df_p2_contexto,
        zona_id,
        t["hora_int"],
        t["mes_num"],
        t["dia_semana"],
    )

def obtener_contexto_p5(origen_id, t):
    """Busca contexto historico P5 para VTC."""
    return buscar_contexto_historico(
        df_p5_contexto,
        origen_id,
        t["hora_int"],
        t["mes_num"],
        t["dia_semana"],
    )

def valor_contexto(info, columna, defecto=0.0, tipo=float):
    """
    Devuelve un valor seguro desde una fila de contexto.
    Si no existe la columna o viene nula, devuelve defecto.
    """
    try:
        if info is None:
            return tipo(defecto)

        if columna not in info.index:
            return tipo(defecto)

        valor = info[columna]

        if pd.isna(valor):
            return tipo(defecto)

        return tipo(valor)

    except Exception:
        return tipo(defecto)

def transformar_label_encoder_seguro(encoder, valor):
    """
    Evita que LabelEncoder falle si llega una zona no vista.
    Si la zona no existe, usa la primera clase conocida.
    """
    try:
        valor = int(valor)

        if valor in encoder.classes_:
            return int(encoder.transform([valor])[0])

        return int(encoder.transform([encoder.classes_[0]])[0])

    except Exception as e:
        print("Error en LabelEncoder:", e)
        return 0

def normalizar_tipo_vehiculo(valor):
    """Convierte los valores del formulario a las categorias vistas por P5."""
    mapa = {
        "1": "Yellow Taxi",
        "2": "VTC",
        "yellow": "Yellow Taxi",
        "taxi": "Yellow Taxi",
        "vtc": "VTC",
    }
    return mapa.get(str(valor).strip().lower(), str(valor))

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
            "mostrar_res": False,
            "resultado": None
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
async def predict_taxi(
    request: Request,
    zona_id: int = Form(...),
    planificacion_hora: str = Form("actual")
):
    t = procesar_tiempo_despliegue(planificacion_hora)
    info = obtener_contexto_zona(zona_id, t)

    # Valores de contexto P2
    oferta = valor_contexto(info, "oferta_inferida", 1.0, float)
    tasa_historica = valor_contexto(info, "tasa_historica", 1.0, float)
    espera_media = valor_contexto(info, "espera_media", 0.0, float)

    temp = valor_contexto(info, "temp_c", 15.0, float)
    precipitation = valor_contexto(info, "precipitation", 0.0, float)
    viento_kmh = valor_contexto(info, "viento_kmh", 10.0, float)
    lluvia = valor_contexto(info, "lluvia", 0, int)
    nieve = valor_contexto(info, "nieve", 0, int)
    es_festivo = valor_contexto(info, "es_festivo", 0, int)
    num_eventos = valor_contexto(info, "num_eventos", 0, float)
    mes_num = int(t["mes_num"])

    oferta_media = float(df_p2_contexto["oferta_inferida"].mean()) if not df_p2_contexto.empty else oferta

    # Predecir la demanda - modelo 1
    df_p1 = pd.DataFrame([{
        "origen_id": int(zona_id),
        "hora": int(t["hora_int"]),
        "dia_semana": int(t["dia_semana"]),
        "dia_mes": int(t["dia_mes"]),
        "mes_num": int(mes_num),
        "es_finde": int(t["es_fin_semana"]),

        "demanda": oferta,
        "lag_1h": oferta,
        "lag_2h": oferta,
        "lag_3h": oferta,
        "lag_6h": oferta,
        "lag_12h": oferta,
        "lag_24h": tasa_historica * oferta_media,

        "roll_mean_3h": oferta,
        "roll_std_3h": oferta * 0.1,
        "roll_mean_24h": oferta,
        "roll_std_24h": oferta * 0.1,
        "media_hist": tasa_historica * oferta_media,

        "temp_c": temp,
        "precipitation": precipitation,
        "viento_kmh": viento_kmh,
        "velocidad_mph": viento_kmh * 0.621,

        "lluvia": lluvia,
        "nieve": nieve,
        "es_festivo": es_festivo,
        "num_eventos": num_eventos
    }])

    demanda_pred = float(app.modelo_p1.predict(df_p1)[0])
    demanda_pred = max(demanda_pred, 0.0)

    # Predecir probabilidad de alta demanda - modelo 2
    z_enc = transformar_label_encoder_seguro(app.encoder_p2, zona_id)
    n_viajes = valor_contexto(info, "n_viajes", oferta, float)

    df_p2 = pd.DataFrame([{
        "hora": int(t["hora_int"]),
        "dia_semana": int(t["dia_semana"]),
        "es_finde": int(t["es_fin_semana"]),
        "hora_sen": float(t["hora_sen"]),
        "hora_cos": float(t["hora_cos"]),
        "mes_num": int(mes_num),
        "temp_c": temp,
        "precipitation": precipitation,
        "viento_kmh": viento_kmh,
        "lluvia": lluvia,
        "nieve": nieve,
        "es_festivo": es_festivo,
        "num_eventos": num_eventos,
        "oferta_inferida": oferta,
        "tasa_historica": tasa_historica,
        "demanda_p1": demanda_pred,
        "n_viajes": n_viajes,
        "espera_media": espera_media,
        "zona_enc": z_enc
    }])

    # Colocamos las columnas en el orden esperado por el modelo
    cols_p2 = [
        "hora",
        "dia_semana",
        "es_finde",
        "hora_sen",
        "hora_cos",
        "mes_num",
        "temp_c",
        "precipitation",
        "viento_kmh",
        "lluvia",
        "nieve",
        "es_festivo",
        "num_eventos",
        "oferta_inferida",
        "tasa_historica",
        "demanda_p1",
        "n_viajes",
        "espera_media",
        "zona_enc"
    ]

    df_p2 = df_p2[cols_p2].fillna(0).astype(np.float32)

    X_p2 = app.scaler_p2.transform(df_p2.values)
    prob = float(app.modelo_p2.predict(X_p2, verbose=0)[0][0])

    # Resultado y decisión
    res = {
        "demanda_estimada": round(demanda_pred, 1),
        "exito_prob": round(prob * 100, 1),
        "recomendacion": "ALTA" if prob > 0.6 else "MODERADA"
    }

    return templates.TemplateResponse(
        request=request,
        name="taxi.html",
        context={
            "request": request,
            "mostrar_res": True,
            "resultado": res,
            "zona_seleccionada": zona_id,
            "hora_seleccionada": planificacion_hora
        }
    )
    
 
@app.post("/vtc", response_class=HTMLResponse)
async def predict_vtc(
    request: Request,
    origen_id: int = Form(...),
    destino_id: int = Form(...),
    precio_base: float = Form(...),
    tipo_vehiculo: str = Form(...),
    planificacion_hora: str = Form("actual")
):
    t = procesar_tiempo_despliegue(planificacion_hora)
    info_p5 = obtener_contexto_p5(origen_id, t)

    # Valores de contexto P5
    temp = valor_contexto(info_p5, "temp_c", 15.0, float)
    precipitation = valor_contexto(info_p5, "precipitation", 0.0, float)
    viento_kmh = valor_contexto(info_p5, "viento_kmh", 10.0, float)
    lluvia = valor_contexto(info_p5, "lluvia", 0, int)
    nieve = valor_contexto(info_p5, "nieve", 0, int)
    es_festivo = valor_contexto(info_p5, "es_festivo", 0, int)
    num_eventos = valor_contexto(info_p5, "num_eventos", 0, int)
    mes_num = int(t["mes_num"])

    hay_lluvia = 1 if lluvia == 1 or precipitation > 0 else 0
    hay_nieve = 1 if nieve == 1 else 0
    
    # Predecir la velocidad - modelo 4
    z_idx = np.array(
        [transformar_label_encoder_seguro(app.encoder_p4, origen_id)],
        dtype=np.int32
    )

    cols_p4 = [
        "temp_c",
        "precipitation",
        "viento_kmh",
        "lluvia",
        "nieve",
        "hay_lluvia",
        "hay_nieve",
        "es_festivo",
        "num_eventos",
        "dia_semana",
        "es_fin_semana",
        "hora_sen",
        "hora_cos"
    ]

    df_clima_p4 = pd.DataFrame([{
        "temp_c": temp,
        "precipitation": precipitation,
        "viento_kmh": viento_kmh,
        "lluvia": lluvia,
        "nieve": nieve,
        "hay_lluvia": hay_lluvia,
        "hay_nieve": hay_nieve,
        "es_festivo": es_festivo,
        "num_eventos": num_eventos,
        "dia_semana": int(t["dia_semana"]),
        "es_fin_semana": int(t["es_fin_semana"]),
        "hora_sen": float(t["hora_sen"]),
        "hora_cos": float(t["hora_cos"])
    }], columns=cols_p4)

    c_num = app.scaler_p4.transform(df_clima_p4)
    vel = float(app.modelo_p4.predict([z_idx, c_num], verbose=0)[0][0])
    vel = max(vel, 0.0)

    # Predecir la propina - modelo 5
    origen_zona = (
        str(info_p5["origen_zona"])
        if info_p5 is not None and "origen_zona" in info_p5.index and pd.notna(info_p5["origen_zona"])
        else "desconocido"
    )

    origen_barrio = (
        str(info_p5["origen_barrio"])
        if info_p5 is not None and "origen_barrio" in info_p5.index and pd.notna(info_p5["origen_barrio"])
        else "desconocido"
    )

    evento_tipo = (
        str(info_p5["evento_tipo"])
        if info_p5 is not None and "evento_tipo" in info_p5.index and pd.notna(info_p5["evento_tipo"])
        else "No hay"
    )

    franja_horaria = (
        str(info_p5["franja_horaria"])
        if info_p5 is not None and "franja_horaria" in info_p5.index and pd.notna(info_p5["franja_horaria"])
        else "Tarde"
    )

    num_pasajeros = valor_contexto(info_p5, "num_pasajeros", 1, int)
    distancia = valor_contexto(info_p5, "distancia", 3.0, float)
    espera_min = valor_contexto(info_p5, "espera_min", 1.0, float)
    duracion_min = valor_contexto(info_p5, "duracion_min", 10.0, float)
    trafico_denso = valor_contexto(info_p5, "trafico_denso", 0, int)
    rentabilidad_base_min = valor_contexto(info_p5, "rentabilidad_base_min", 1.0, float)

    precio_total_est = valor_contexto(
        info_p5,
        "precio_total_est",
        float(precio_base) + 2.0,
        float
    )

    df_p5 = pd.DataFrame([{
        "tipo_vehiculo": normalizar_tipo_vehiculo(tipo_vehiculo),
        "origen_zona": origen_zona,
        "origen_barrio": origen_barrio,
        "evento_tipo": evento_tipo,
        "franja_horaria": franja_horaria,

        "precio_base": float(precio_base),
        "hora_sen": float(t["hora_sen"]),
        "hora_cos": float(t["hora_cos"]),
        "es_fin_semana": int(t["es_fin_semana"]),

        "num_pasajeros": num_pasajeros,
        "velocidad_mph": vel,
        "lluvia": lluvia,
        "temp_c": temp,
        "es_festivo": es_festivo,
        "distancia": distancia,
        "nieve": nieve,
        "espera_min": espera_min,
        "precipitation": precipitation,
        "trafico_denso": trafico_denso,
        "duracion_min": duracion_min,
        "num_eventos": num_eventos,
        "mes_num": mes_num,
        "hora": int(t["hora_int"]),
        "rentabilidad_base_min": rentabilidad_base_min,
        "viento_kmh": viento_kmh,
        "precio_total_est": precio_total_est,
        "dia_semana": int(t["dia_semana"])
    }])

    # Colocamos las columnas en el orden esperado por el modelo
    cols_p5 = [
        "tipo_vehiculo",
        "precio_base",
        "precio_total_est",
        "espera_min",
        "origen_zona",
        "origen_barrio",
        "temp_c",
        "precipitation",
        "viento_kmh",
        "lluvia",
        "nieve",
        "es_festivo",
        "evento_tipo",
        "num_eventos",
        "mes_num",
        "hora",
        "dia_semana",
        "es_fin_semana",
        "franja_horaria",
        "hora_sen",
        "hora_cos",
        "num_pasajeros",
        "distancia",
        "duracion_min",
        "velocidad_mph",
        "rentabilidad_base_min",
        "trafico_denso"
    ]

    df_p5 = df_p5[cols_p5]

    columnas_categoricas = [
        "tipo_vehiculo",
        "origen_zona",
        "origen_barrio",
        "evento_tipo",
        "franja_horaria"
    ]

    for col in columnas_categoricas:
        df_p5[col] = df_p5[col].astype(str).fillna("desconocido")

    propina = float(app.modelo_p5.predict(df_p5)[0])
    propina = max(propina, 0.0)

    # Resultado y decisión
    decision = "ACEPTAR" if propina > 2 else "RECHAZAR"

    res = {
        "propina": round(propina, 2),
        "velocidad": round(vel, 1),
        "encadenado": "ALTA" if propina > 2 and vel > 8 else "MODERADA",
        "decision": decision,
        "detalles": "Análisis realizado con contexto histórico de la zona."
    }

    return templates.TemplateResponse(
        request=request,
        name="vtc.html",
        context={
            "request": request,
            "resultado": res,
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
