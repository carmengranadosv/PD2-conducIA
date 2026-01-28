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
    """
    # fechas
    _to_datetime(df, "fecha_inicio")
    _to_datetime(df, "fecha_fin")

    # duración
    df["duracion_min"] = (df["fecha_fin"] - df["fecha_inicio"]).dt.total_seconds() / 60

    # nulos críticos comunes
    df = df.dropna(subset=COMMON_COLS + ["duracion_min"]).copy()

    # coherencia temporal
    df = df[df["fecha_fin"] >= df["fecha_inicio"]].copy()

    # filtros generales anti-basura
    df = df[
        (df["distancia"] > 0) & (df["distancia"] < 1000) &
        (df["duracion_min"] > 1) & (df["duracion_min"] < 300)
    ].copy()

    # tipo
    df["tipo_vehiculo"] = service

    # tipos comunes
    _to_int32_nullable(df, "origen_id")
    _to_int32_nullable(df, "destino_id")
    _to_float(df, "distancia")
    _to_float(df, "duracion_min")

    return df


def _clean_yellow_specific(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reglas específicas de Yellow:
    - precio_total y tarifa_base razonables
    - num_pasajeros razonable (0..6 típico, pero permitimos más)
    - tipos y limpieza de columnas económicas
    """
    # num_pasajeros
    if "num_pasajeros" in df.columns:
        _to_int32_nullable(df, "num_pasajeros")
        df = df[(df["num_pasajeros"].isna()) | ((df["num_pasajeros"] >= 0) & (df["num_pasajeros"] <= 8))].copy()

    # precios (yellow sí tiene precio_total)
    if "precio_total" in df.columns:
        _to_float(df, "precio_total")
        df = df[(df["precio_total"] > 0) & (df["precio_total"] < 500)].copy()

    if "tarifa_base" in df.columns:
        _to_float(df, "tarifa_base")
        df = df[(df["tarifa_base"] >= 0) & (df["tarifa_base"] < 500)].copy()

    # propina / peajes / recargos (pueden ser 0)
    for c in ["propina", "peajes", "extra", "mta_tax", "recargo_mejora", "recargo_congestion", "recargo_cbd", "ehail_fee"]:
        if c in df.columns:
            _to_float(df, c)
            df = df[df[c].isna() | (df[c] >= 0)].copy()

    # códigos (nullable)
    for c in ["tipo_pago", "codigo_tarifa", "vendor_id", "tipo_viaje"]:
        _to_int32_nullable(df, c)

    return df


def _clean_fhvhv_specific(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reglas específicas de FHVHV:
    - espera_min con fecha_solicitud (si existe)
    - duracion_seg razonable (si existe)
    - tarifa_base razonable (sí existe)
    - flags (shared/wav) limpiar strings tipo 'Y'/'N'
    """
    # tarifa_base (principal variable económica)
    if "tarifa_base" in df.columns:
        _to_float(df, "tarifa_base")
        df = df[(df["tarifa_base"] > 0) & (df["tarifa_base"] < 500)].copy()

    # duracion_seg
    if "duracion_seg" in df.columns:
        _to_float(df, "duracion_seg")
        df = df[(df["duracion_seg"] > 30) & (df["duracion_seg"] < 6 * 3600)].copy()  # 30s a 6h

    # espera_min (solo si tenemos fecha_solicitud)
    if "fecha_solicitud" in df.columns:
        _to_datetime(df, "fecha_solicitud")
        df["espera_min"] = (df["fecha_inicio"] - df["fecha_solicitud"]).dt.total_seconds() / 60
        df = df.dropna(subset=["espera_min"])
        df = df[(df["espera_min"] >= 0) & (df["espera_min"] <= 120)].copy()
        _to_float(df, "espera_min")

    # dinero extra (>=0)
    for c in ["propina", "peajes", "black_car_fund", "impuesto_ventas", "recargo_congestion", "recargo_aeropuerto", "recargo_cbd", "pago_conductor"]:
        if c in df.columns:
            _to_float(df, c)
            df = df[df[c].isna() | (df[c] >= 0)].copy()

    # flags Y/N -> category (no obligatorio, pero limpia)
    for c in ["solicitud_compartida", "viaje_compartido", "wav_solicitado", "wav_realizado", "access_a_ride"]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.upper()
            df.loc[~df[c].isin(["Y", "N", "<NA>"]), c] = pd.NA

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
