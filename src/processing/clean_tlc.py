from pathlib import Path
import pandas as pd
import numpy as np

# Importación de configuraciones externas
from src.processing.columnas import COLUMNAS_YELLOW, COLUMNAS_FHVHV

# Definición de columnas troncales
COMMON_COLS = ["fecha_inicio", "fecha_fin", "origen_id", "destino_id", "distancia", "propina", "tipo_pago"]

def _obtener_mapeo_columnas(service: str) -> dict:
    service = service.lower()
    if service == "yellow":
        return COLUMNAS_YELLOW
    if service == "fhvhv":
        return COLUMNAS_FHVHV
    raise ValueError(f"Servicio no reconocido: {service}")


def _convertir_tipos_base(df: pd.DataFrame) -> None:
    # Conversión de fechas
    for col in ["fecha_inicio", "fecha_fin", "fecha_solicitud"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Conversión de flotantes
    cols_float = ["distancia", "duracion_min", "duracion_seg", "espera_min", 
                  "tarifa_base", "precio_total", "peajes", "propina", "precio_base", "precio_total_est"]
    
    for col in cols_float:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    # Conversión de enteros
    cols_int = ["origen_id", "destino_id", "num_pasajeros"]
    for col in cols_int:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")


def _aplicar_limpieza_comun(df: pd.DataFrame, service: str) -> pd.DataFrame:
    _convertir_tipos_base(df)

    # 1. Filtro de consistencia temporal
    if "fecha_fin" in df.columns and "fecha_inicio" in df.columns:
        df = df[df["fecha_fin"] > df["fecha_inicio"]].copy() # Estricto mayor que
        df["duracion_min"] = (df["fecha_fin"] - df["fecha_inicio"]).dt.total_seconds() / 60

    # 2. Filtros Físicos (Hard Limits)
    df = df[
        (df["distancia"] > 0.1) & (df["distancia"] < 200) & # 1000 millas es demasiado para un taxi urbano
        (df["duracion_min"] > 1) & (df["duracion_min"] < 180) # Más de 3 horas es raro/error
    ].copy()

    # 3. Filtro de Velocidad
    df["velocidad_mph"] = df["distancia"] / (df["duracion_min"] / 60)

    # 4. Eliminamos las zonas desconocidas
    # También eliminamos si el ID es nulo.
    if "origen_id" in df.columns and "destino_id" in df.columns:
        df = df[~df["origen_id"].isin([264, 265])]
        df = df[~df["destino_id"].isin([264, 265])]

    # Limpieza de nulos en columnas críticas
    cols_criticas = [c for c in COMMON_COLS if c in df.columns] + ["duracion_min"]
    df = df.dropna(subset=cols_criticas).copy()

    df["tipo_vehiculo"] = service

    return df.dropna().copy()


def _procesar_logica_yellow(df: pd.DataFrame) -> pd.DataFrame:
    """
    LOGICA ORIGINAL (La que te funcionaba):
    Usa 'Blacklist': Elimina columnas específicas y deja pasar el resto.
    """
    if "num_pasajeros" in df.columns:
        df = df[(df["num_pasajeros"] >= 0) & (df["num_pasajeros"] <= 8)].copy()

    if "tarifa_base" in df.columns and "precio_total" in df.columns:
        df = df[(df["tarifa_base"] >= 0) & (df["tarifa_base"] < 500)].copy()
        df = df[(df["precio_total"] > 0) & (df["precio_total"] < 500)].copy()
    else:
        return df.iloc[0:0].copy()

    df["precio_base"] = df["tarifa_base"]
    df["precio_total_est"] = df["precio_total"]

    # Eliminamos solo lo que sabemos que sobra
    cols_a_eliminar = [
        "tarifa_base", "precio_total", "peajes", "extra", 
        "mta_tax", "recargo_mejora", "recargo_congestion", "ehail_fee", "codigo_tarifa", "tipo_viaje"
    ]
    df = df.drop(columns=[c for c in cols_a_eliminar if c in df.columns], errors='ignore')

    return df.dropna().copy()


def _procesar_logica_fhvhv(df: pd.DataFrame) -> pd.DataFrame:
    """
    LOGICA NUEVA (Estricta):
    Usa 'Whitelist': Solo se queda con las columnas que definimos aquí.
    Esto elimina las columnas extra que te daban problemas.
    """
    if "tarifa_base" not in df.columns:
        return df.iloc[0:0].copy()

    df = df[(df["tarifa_base"] > 0) & (df["tarifa_base"] < 500)].copy()

    if "duracion_seg" in df.columns:
        df = df[(df["duracion_seg"] > 30) & (df["duracion_seg"] < 6 * 3600)].copy()

    if "fecha_solicitud" in df.columns:
        df["espera_min"] = (df["fecha_inicio"] - df["fecha_solicitud"]).dt.total_seconds() / 60
        df = df[(df["espera_min"] >= 0) & (df["espera_min"] <= 120)].copy()

    # Reconstrucción del Precio Total
    componentes_precio = [
        "peajes", "black_car_fund", "impuesto_ventas",
        "recargo_congestion", "recargo_aeropuerto", "recargo_cbd"
    ]
    
    for comp in componentes_precio:
        if comp in df.columns:
            df[comp] = df[comp].fillna(0)
            df = df[df[comp] >= 0].copy()
        else:
            df[comp] = 0.0

    df["precio_base"] = df["tarifa_base"]
    df["precio_total_est"] = df["tarifa_base"] + df[componentes_precio].sum(axis=1)
    
    df = df[(df["precio_total_est"] > 0) & (df["precio_total_est"] < 500)].copy()

    # --- LISTA BLANCA (WHITELIST) ---
    # Solo estas columnas sobrevivirán. El resto se borra.
    cols_finales = [
        "fecha_inicio", "fecha_fin", "origen_id", "destino_id", 
        "distancia", "duracion_min", "velocidad_mph", "tipo_vehiculo",
        "precio_base", "precio_total_est", "espera_min", "propina", "tipo_pago"
    ]

    cols_a_mantener = [c for c in cols_finales if c in df.columns]

    return df[cols_a_mantener].dropna().copy()

def _imputacion_negocio(df: pd.DataFrame, service: str) -> pd.DataFrame:
    """
    Imputación determinista y simple que sí corresponde a 'clean':
    - Yellow: espera_min = 0
    - Yellow: num_pasajeros: fillna(1) opcional
    """
    df = df.copy()
    service = service.lower()

    # Asegurar columna espera_min homogénea
    if "espera_min" not in df.columns:
        df["espera_min"] = pd.NA

    if service == "yellow":
        df["espera_min"] = df["espera_min"].fillna(0.0)

        # opcional: si te interesa evitar NA en pasajeros en yellow
        if "num_pasajeros" in df.columns:
            df["num_pasajeros"] = df["num_pasajeros"].fillna(1)

    # Para fhvhv: espera_min ya se calcula si hay fecha_solicitud.
    # Si no existe o falta, mejor dejar NA (no inventar).

    return df


def clean_df(df: pd.DataFrame, service: str, month_filter: int = None) -> pd.DataFrame:
    service = service.lower()
    
    mapa_cols = _obtener_mapeo_columnas(service)
    cols_in = [c for c in mapa_cols.keys() if c in df.columns]
    
    if not cols_in:
        raise ValueError(f"El dataset no contiene las columnas esperadas para {service}.")

    df_procesado = df[cols_in].rename(columns=mapa_cols).copy()

    # --- FILTRO DE MES DINÁMICO ---
    # Convertimos a datetime antes de limpiar para poder filtrar
    if "fecha_inicio" in df_procesado.columns:
        df_procesado["fecha_inicio"] = pd.to_datetime(df_procesado["fecha_inicio"], errors="coerce")
        # Si nos pasan un mes, filtramos rigurosamente
        if month_filter:
            df_procesado = df_procesado[df_procesado["fecha_inicio"].dt.month == month_filter].copy()

    df_procesado = _aplicar_limpieza_comun(df_procesado, service)

    if service == "yellow":
        df_procesado = _procesar_logica_yellow(df_procesado)
    elif service == "fhvhv":
        df_procesado = _procesar_logica_fhvhv(df_procesado)
       # Imputación de negocio (determinista)
    df_procesado = _imputacion_negocio(df_procesado, service)

    return df_procesado



def clean_file(in_path: Path, out_path: Path, service: str, overwrite: bool = False) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        return f"SKIP {out_path.name} (Ya existe)"

    try:
        # Extraer el mes del nombre del archivo (ej: '...2025-11.parquet' -> 11)
        try:
            nombre_archivo = in_path.stem # 'yellow_tripdata_2025-11'
            mes_str = nombre_archivo.split('-')[-1] # '11'
            mes_a_filtrar = int(mes_str)
        except:
            mes_a_filtrar = None # Si el nombre no tiene el formato, no filtramos por mes
            print(f"⚠️ No se pudo extraer el mes de {in_path.name}, se procesará sin filtro mensual.")

        df = pd.read_parquet(in_path)
        
        # Pasamos el mes detectado a la limpieza
        df_clean = clean_df(df, service=service, month_filter=mes_a_filtrar)
        
        df_clean.to_parquet(out_path, index=False)
        return f"OK   {out_path.name} ({len(df_clean):,} registros procesados)"
    except Exception as e:
        return f"ERROR en {out_path.name}: {str(e)}"