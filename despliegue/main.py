import os
# --- 1. CONFIGURACIÓN DE RENDIMIENTO ---
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_ALLOC_FRACTION"] = ".10" 

from pathlib import Path
from datetime import datetime
from typing import Optional
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
from src.funcionalidades.demanda_zona_franja import (
    cargar_resumen_para_consulta,
    consultar_demanda,
)
from src.funcionalidades.max_demanda import predecir_top_3_zonas

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
FUNCIONALIDADES_DIR = DATA_ROOT / "funcionalidades"
MAPA_HTML_PATH = FUNCIONALIDADES_DIR / "mapa_poder_barrios.html"
DEMANDA_CACHE_DIR = CONTEXT_DIR / "funcionalidades"

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

        path_zonas = DATA_ROOT / "external/taxi_zone_lookup.csv"
        if path_zonas.exists():
            app.df_zonas = pd.read_csv(path_zonas)
        else:
            # Fallback por si la ruta es distinta en tu local
            app.df_zonas = pd.DataFrame(columns=['LocationID', 'Zone', 'Borough'])

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
        "hora_float": hora_float,
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

def desplazar_tiempo(t, minutos: float):
    """Aproxima la hora de llegada manteniendo mes/dia del contexto actual."""
    hora_float = float(t["hora_float"]) + max(float(minutos), 0.0) / 60.0
    dias_extra = int(hora_float // 24)
    hora_float = hora_float % 24
    dia_semana = (int(t["dia_semana"]) + dias_extra) % 7
    es_fin_semana = 1 if dia_semana >= 5 else 0

    return {
        "hora_float": hora_float,
        "hora_int": int(hora_float),
        "hora_sen": float(np.sin(2 * np.pi * hora_float / 24)),
        "hora_cos": float(np.cos(2 * np.pi * hora_float / 24)),
        "dia_semana": dia_semana,
        "dia_mes": int(t["dia_mes"]),
        "mes_num": int(t["mes_num"]),
        "es_fin_semana": es_fin_semana,
        "es_finde": es_fin_semana,
    }

def predecir_potencial_zona(zona_id: int, t):
    """Predice demanda P1 y probabilidad P2 para una zona y momento."""
    info = obtener_contexto_zona(zona_id, t)

    oferta = valor_contexto(info, "oferta_inferida", 1.0, float)
    tasa_historica = valor_contexto(info, "tasa_historica", 1.0, float)
    espera_media = valor_contexto(info, "espera_media", 0.0, float)

    temp = valor_contexto(info, "temp_c", 15.0, float)
    precipitation = valor_contexto(info, "precipitation", 0.0, float)
    viento_kmh = valor_contexto(info, "viento_kmh", 10.0, float)
    lluvia = valor_contexto(info, "lluvia", 0.0, float)
    nieve = valor_contexto(info, "nieve", 0.0, float)
    es_festivo = valor_contexto(info, "es_festivo", 0.0, float)
    num_eventos = valor_contexto(info, "num_eventos", 0, float)
    mes_num = int(t["mes_num"])

    oferta_media = float(df_p2_contexto["oferta_inferida"].mean()) if not df_p2_contexto.empty else oferta

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

    z_enc = transformar_label_encoder_seguro(app.encoder_p2, zona_id)
    n_viajes = valor_contexto(info, "n_viajes", oferta, float)

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

    df_p2 = df_p2[cols_p2].fillna(0).astype(np.float32)
    X_p2 = app.scaler_p2.transform(df_p2.values)
    prob = float(app.modelo_p2.predict(X_p2, verbose=0)[0][0])

    return {
        "demanda_estimada": demanda_pred,
        "exito_prob": prob,
        "contexto": info,
    }

def predecir_demanda_zona_p1(zona_id: int, t):
    info = obtener_contexto_zona(zona_id, t)

    oferta = valor_contexto(info, "oferta_inferida", 1.0, float)
    tasa_historica = valor_contexto(info, "tasa_historica", 1.0, float)

    temp = valor_contexto(info, "temp_c", 15.0, float)
    precipitation = valor_contexto(info, "precipitation", 0.0, float)
    viento_kmh = valor_contexto(info, "viento_kmh", 10.0, float)
    lluvia = valor_contexto(info, "lluvia", 0.0, float)
    nieve = valor_contexto(info, "nieve", 0.0, float)
    es_festivo = valor_contexto(info, "es_festivo", 0.0, float)
    num_eventos = valor_contexto(info, "num_eventos", 0, float)

    oferta_media = (
        float(df_p2_contexto["oferta_inferida"].mean())
        if not df_p2_contexto.empty and "oferta_inferida" in df_p2_contexto.columns
        else oferta
    )

    df_p1 = pd.DataFrame([{
        "origen_id": int(zona_id),
        "hora": int(t["hora_int"]),
        "dia_semana": int(t["dia_semana"]),
        "dia_mes": int(t["dia_mes"]),
        "mes_num": int(t["mes_num"]),
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
        "num_eventos": num_eventos,
    }])

    return max(float(app.modelo_p1.predict(df_p1)[0]), 0.0)

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


def obtener_resumen_demanda_contexto():
    """Usa contexto_p2 como fuente ligera para la funcionalidad zona-franja."""
    try:
        return cargar_resumen_para_consulta(
            out_dir=DEMANDA_CACHE_DIR,
            inputs=[P2_CONTEXT_PATH],
        )
    except Exception as e:
        print(f"⚠️ Error cargando resumen demanda_zona_franja: {e}")
        return pd.DataFrame()


def obtener_opciones_zona(resumen: pd.DataFrame):
    if resumen.empty:
        return []
    cols = [c for c in ["origen_id", "origen_zona", "origen_barrio"] if c in resumen.columns]
    if "origen_id" not in cols:
        return []
    base = resumen[cols].drop_duplicates().sort_values("origen_id")
    opciones = []
    for _, row in base.iterrows():
        zona = str(row["origen_zona"]) if "origen_zona" in base.columns and pd.notna(row.get("origen_zona")) else "Zona"
        barrio = str(row["origen_barrio"]) if "origen_barrio" in base.columns and pd.notna(row.get("origen_barrio")) else ""
        label = f"{int(row['origen_id'])} · {zona}"
        if barrio:
            label += f" ({barrio})"
        opciones.append({"id": int(row["origen_id"]), "label": label})
    return opciones


@app.get("/funcionalidades", response_class=HTMLResponse)
async def pantalla_funcionalidades(request: Request):
    resumen = obtener_resumen_demanda_contexto()
    opciones = obtener_opciones_zona(resumen)
    resultados = []
    if not resumen.empty:
        resultados = (
            resumen[resumen["nivel_demanda"] == "alta"]
            .nlargest(20, "demanda_media")
            .to_dict(orient="records")
        )
    return templates.TemplateResponse(
        request=request,
        name="funcionalidades.html",
        context={
            "request": request,
            "mapa_disponible": MAPA_HTML_PATH.exists(),
            "filtros": {"top": 20},
            "resultados": resultados,
            "zona_options": opciones,
            "error": None,
        },
    )


@app.post("/funcionalidades", response_class=HTMLResponse)
async def consultar_funcionalidades(
    request: Request,
    zona_id: Optional[str] = Form(None),
    zona: Optional[str] = Form(None),
    franja: Optional[str] = Form(None),
    nivel: Optional[str] = Form(None),
    demanda_min: Optional[str] = Form(None),
    demanda_max: Optional[str] = Form(None),
    top: int = Form(20),
):
    filtros = {
        "zona_id": (zona_id or "").strip(),
        "zona": (zona or "").strip(),
        "franja": (franja or "").strip(),
        "nivel": (nivel or "").strip(),
        "demanda_min": (demanda_min or "").strip(),
        "demanda_max": (demanda_max or "").strip(),
        "top": top,
    }

    try:
        resumen = obtener_resumen_demanda_contexto()
        if resumen.empty:
            raise ValueError("No se pudo cargar el resumen de demanda desde contexto_p2.")
        opciones = obtener_opciones_zona(resumen)

        zona_consulta = filtros["zona_id"] if filtros["zona_id"] else (filtros["zona"] or None)

        if not any([zona_consulta, filtros["franja"], filtros["nivel"], filtros["demanda_min"], filtros["demanda_max"]]):
            resultado = resumen.nlargest(int(top), "demanda_media")
        else:
            resultado = consultar_demanda(
                resumen=resumen,
                zona=zona_consulta,
                franja=filtros["franja"] or None,
                nivel=filtros["nivel"] or None,
                demanda_min=float(filtros["demanda_min"]) if filtros["demanda_min"] else None,
                demanda_max=float(filtros["demanda_max"]) if filtros["demanda_max"] else None,
                top=int(top),
            )

        resultados = resultado.to_dict(orient="records")
        error = None
    except Exception as e:
        resultados = []
        opciones = []
        error = str(e)

    return templates.TemplateResponse(
        request=request,
        name="funcionalidades.html",
        context={
            "request": request,
            "mapa_disponible": MAPA_HTML_PATH.exists(),
            "filtros": filtros,
            "resultados": resultados,
            "zona_options": opciones,
            "error": error,
        },
    )


@app.post("/funcionalidades/max-demanda", response_class=HTMLResponse)
async def post_max_demanda(
    request: Request, 
    dia: str = Form(...), 
    hora: int = Form(...)
):
    try:
        dias_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2,
            "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
        }

        t = procesar_tiempo_despliegue(str(hora))
        t["dia_semana"] = dias_map[dia]
        t["es_fin_semana"] = 1 if t["dia_semana"] >= 5 else 0
        t["es_finde"] = t["es_fin_semana"]

        resultados = []

        for zona_id in app.df_zonas["LocationID"].dropna().unique():
            try:
                demanda = predecir_demanda_zona_p1(int(zona_id), t)

                info_zona = app.df_zonas[app.df_zonas["LocationID"] == zona_id].iloc[0]

                resultados.append({
                    "nombre": info_zona["Zone"],
                    "barrio": info_zona["Borough"],
                    "demanda": demanda,
                })

            except Exception:
                continue

        top_3 = sorted(resultados, key=lambda x: x["demanda"], reverse=True)[:3]
        error_max = None

    except Exception as e:
        top_3 = []
        error_max = f"Error en predicción: {str(e)}"

    resumen = obtener_resumen_demanda_contexto()
    opciones = obtener_opciones_zona(resumen)

    return templates.TemplateResponse(
        request=request,
        name="funcionalidades.html",
        context={
            "request": request,
            "mapa_disponible": MAPA_HTML_PATH.exists(),
            "zona_options": opciones,
            "top_zonas": top_3,
            "dia_sel": dia,
            "hora_sel": hora,
            "error_max": error_max,
            "resultados": [],
            "filtros": {"top": 20},
            "error": None,
        },
    )

@app.get("/funcionalidades/mapa", response_class=HTMLResponse)
async def ver_mapa_funcionalidades():
    if not MAPA_HTML_PATH.exists():
        return HTMLResponse(
            "<h3>Mapa no disponible.</h3>"
            "<p>Generalo con: uv run python -m src.funcionalidades.mapa_coropletico</p>",
            status_code=404,
        )
    return HTMLResponse(MAPA_HTML_PATH.read_text(encoding="utf-8"))

# --- 8. LÓGICA POST (PREDICCIONES) ---
@app.post("/taxi", response_class=HTMLResponse)
async def predict_taxi(
    request: Request,
    zona_id: int = Form(...),
    planificacion_hora: str = Form("actual")
):
    t = procesar_tiempo_despliegue(planificacion_hora)
    potencial = predecir_potencial_zona(zona_id, t)
    demanda_pred = potencial["demanda_estimada"]
    prob = potencial["exito_prob"]

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
    lluvia = valor_contexto(info_p5, "lluvia", 0.0, float)
    nieve = valor_contexto(info_p5, "nieve", 0.0, float)
    es_festivo = valor_contexto(info_p5, "es_festivo", 0.0, float)
    num_eventos = valor_contexto(info_p5, "num_eventos", 0, int)
    mes_num = int(t["mes_num"])

    hay_lluvia = 1 if lluvia > 0 or precipitation > 0 else 0
    hay_nieve = 1 if nieve > 0 else 0
    
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
    trafico_denso = valor_contexto(info_p5, "trafico_denso", 0.0, float)
    rentabilidad_base_min = valor_contexto(info_p5, "rentabilidad_base_min", 1.0, float)

    precio_base_contexto = valor_contexto(info_p5, "precio_base", float(precio_base), float)
    precio_total_contexto = valor_contexto(info_p5, "precio_total_est", float(precio_base) + 2.0, float)
    extras_historicos = max(precio_total_contexto - precio_base_contexto, 0.0)
    precio_total_est = float(precio_base) + extras_historicos

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
    t_llegada = desplazar_tiempo(t, duracion_min)
    retorno = predecir_potencial_zona(destino_id, t_llegada)
    retorno_prob = float(retorno["exito_prob"])

    propina_score = min(propina / 4.0, 1.0)
    velocidad_score = min(vel / 20.0, 1.0)
    rentabilidad_score = (
        0.45 * propina_score
        + 0.25 * velocidad_score
        + 0.30 * retorno_prob
    )
    decision = "ACEPTAR" if rentabilidad_score >= 0.55 else "RECHAZAR"

    res = {
        "propina": round(propina, 2),
        "velocidad": round(vel, 1),
        "retorno_prob": round(retorno_prob * 100, 1),
        "encadenado": "ALTA" if retorno_prob > 0.6 else "MODERADA",
        "rentabilidad_score": round(rentabilidad_score * 100, 1),
        "decision": decision,
        "detalles": (
            f"Retorno calculado en destino a las {t_llegada['hora_int']}:00 "
            "con contexto histórico comparable."
        )
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
