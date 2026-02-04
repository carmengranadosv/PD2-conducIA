from pathlib import Path
import pandas as pd

from src.processing.columnas import COLUMNAS_YELLOW, COLUMNAS_FHVHV

# columnas comunes ya renombradas (tras aplicar el mapping)
COMMON_COLS = ["fecha_inicio", "fecha_fin", "origen_id", "destino_id", "distancia"]


def _columnas_por_servicio(service: str) -> dict:
    service = service.lower()
    if service == "yellow":
        return COLUMNAS_YELLOW
    if service == "fhvhv":
        return COLUMNAS_FHVHV
    raise ValueError(f"Servicio no soportado: {service}")


def _to_datetime(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


def _to_float(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")


def _to_int32_nullable(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")

def _clean_common(df: pd.DataFrame, service: str) -> pd.DataFrame:
    """
    Limpieza base para ambos: fechas, zonas, distancia, duración.
    Además: elimina cualquier fila con nulos (en todas las columnas presentes).
    """
    # fechas
    _to_datetime(df, "fecha_inicio")
    _to_datetime(df, "fecha_fin")

    # duración
    df["duracion_min"] = (df["fecha_fin"] - df["fecha_inicio"]).dt.total_seconds() / 60

    # 1) eliminar nulos en las comunes + duracion (primero)
    df = df.dropna(subset=COMMON_COLS + ["duracion_min"]).copy()

    # coherencia temporal
    df = df[df["fecha_inicio"].dt.year == 2025].copy()
    df = df[df["fecha_fin"] >= df["fecha_inicio"]].copy()

    # filtros generales anti-basura
    df = df[
        (df["distancia"] > 0) & (df["distancia"] < 1000) &
        (df["duracion_min"] > 1) & (df["duracion_min"] < 300)
    ].copy()

    # tipo
    df["tipo_vehiculo"] = service

    # tipos comunes (ojo: convertir puede generar NaN si había strings raros)
    _to_int32_nullable(df, "origen_id")
    _to_int32_nullable(df, "destino_id")
    _to_float(df, "distancia")
    _to_float(df, "duracion_min")

    # 2) AHORA sí: elimina cualquier fila con nulos en cualquier columna existente
    # (incluye las columnas específicas que ya venían desde el mapping)
    df = df.dropna().copy()

    return df



def _clean_yellow_specific(df: pd.DataFrame) -> pd.DataFrame:
    """
    Yellow:
    - precio_base = tarifa_base
    - precio_total_est = precio_total (ya es el total real)
    - Elimina columnas de componentes de precio (solo deja los 2 precios)
    """

    # num_pasajeros
    if "num_pasajeros" in df.columns:
        _to_int32_nullable(df, "num_pasajeros")
        df = df[(df["num_pasajeros"] >= 0) & (df["num_pasajeros"] <= 8)].copy()

    # convertir y filtrar precio base/total
    if "tarifa_base" in df.columns:
        _to_float(df, "tarifa_base")
        df = df[(df["tarifa_base"] >= 0) & (df["tarifa_base"] < 500)].copy()
    else:
        return df.iloc[0:0].copy()

    if "precio_total" in df.columns:
        _to_float(df, "precio_total")
        df = df[(df["precio_total"] > 0) & (df["precio_total"] < 500)].copy()
    else:
        return df.iloc[0:0].copy()

    # ✅ precios estándar
    df["precio_base"] = df["tarifa_base"]
    df["precio_total_est"] = df["precio_total"]

    # ✅ eliminar componentes de precio (y las columnas originales)
    cols_drop = [
        "tarifa_base", "precio_total",
        "propina", "peajes", "extra", "mta_tax",
        "recargo_mejora", "recargo_congestion", "ehail_fee",
        "tipo_pago", "codigo_tarifa", "tipo_viaje"
    ]
    df = df.drop(columns=[c for c in cols_drop if c in df.columns])

    # elimina nulos (ahora ya son poquitas columnas)
    df = df.dropna().copy()

    return df




def _clean_fhvhv_specific(df: pd.DataFrame) -> pd.DataFrame:
    """
    FHVHV:
    - precio_base = tarifa_base
    - precio_total_est = suma de componentes (aprox total pagado)
    - Después borra componentes (solo deja los 2 precios)
    """

    # tarifa_base obligatoria
    if "tarifa_base" not in df.columns:
        return df.iloc[0:0].copy()

    _to_float(df, "tarifa_base")
    df = df[(df["tarifa_base"] > 0) & (df["tarifa_base"] < 500)].copy()

    # duracion_seg
    if "duracion_seg" in df.columns:
        _to_float(df, "duracion_seg")
        df = df[(df["duracion_seg"] > 30) & (df["duracion_seg"] < 6 * 3600)].copy()

    # espera_min
    if "fecha_solicitud" in df.columns:
        _to_datetime(df, "fecha_solicitud")
        df["espera_min"] = (df["fecha_inicio"] - df["fecha_solicitud"]).dt.total_seconds() / 60
        _to_float(df, "espera_min")
        df = df[(df["espera_min"] >= 0) & (df["espera_min"] <= 120)].copy()

    # componentes de precio (si no existen o vienen NaN -> 0)
    componentes = [
        "peajes", "black_car_fund", "impuesto_ventas",
        "recargo_congestion", "recargo_aeropuerto", "recargo_cbd", "propina"
    ]
    for c in componentes:
        if c in df.columns:
            _to_float(df, c)
            df[c] = df[c].fillna(0)
            df = df[df[c] >= 0].copy()
        else:
            df[c] = 0.0

    # ✅ precios estándar
    df["precio_base"] = df["tarifa_base"]
    df["precio_total_est"] = (
        df["tarifa_base"]
        + df["peajes"]
        + df["black_car_fund"]
        + df["impuesto_ventas"]
        + df["recargo_congestion"]
        + df["recargo_aeropuerto"]
        + df["recargo_cbd"]
        + df["propina"]
    )

    df = df[(df["precio_total_est"] > 0) & (df["precio_total_est"] < 500)].copy()

    # ✅ eliminar componentes (y originales)
    cols_drop = ["tarifa_base"] + componentes + ["pago_conductor"]
    df = df.drop(columns=[c for c in cols_drop if c in df.columns])

    # flags (si quieres conservarlas, no las borres; si no, puedes dropearlas también)
    # Aquí las dejo tal cual.

    df = df.dropna().copy()
    return df



def clean_df(df: pd.DataFrame, service: str) -> pd.DataFrame:
    service = service.lower()
    cols_map = _columnas_por_servicio(service)

    # 1) Seleccionar + renombrar TODAS las columnas del mapping que existan
    cols_in = [c for c in cols_map.keys() if c in df.columns]
    if not cols_in:
        raise ValueError("No se encontraron columnas esperadas para este servicio.")

    df = df[cols_in].rename(columns=cols_map).copy()

    # 2) Limpieza común
    df = _clean_common(df, service=service)

    # 3) Limpieza específica
    if service == "yellow":
        df = _clean_yellow_specific(df)
    elif service == "fhvhv":
        df = _clean_fhvhv_specific(df)

    return df


def clean_file(
    in_path: Path,
    out_path: Path,
    service: str,
    overwrite: bool = False
) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        return f"SKIP {out_path.name}"

    df = pd.read_parquet(in_path)
    df_clean = clean_df(df, service=service)
    df_clean.to_parquet(out_path, index=False)

    return f"OK   {out_path.name} ({len(df_clean):,} filas)"
