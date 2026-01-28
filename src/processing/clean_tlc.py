from pathlib import Path
import pandas as pd

from src.processing.schema import COLUMNAS_YELLOW, COLUMNAS_FHVHV


def _columnas_por_servicio(service: str) -> dict:
    if service == "yellow":
        return COLUMNAS_YELLOW
    if service == "fhvhv":
        return COLUMNAS_FHVHV
    raise ValueError(f"Servicio no soportado: {service}")


def clean_df(df: pd.DataFrame, service: str) -> pd.DataFrame:
    cols_map = _columnas_por_servicio(service)

    cols = [c for c in cols_map if c in df.columns]
    if not cols:
        raise ValueError("No se encontraron columnas esperadas")

    df = df[cols].rename(columns=cols_map).copy()

    df["fecha_inicio"] = pd.to_datetime(df["fecha_inicio"], errors="coerce")
    df["fecha_fin"] = pd.to_datetime(df["fecha_fin"], errors="coerce")
    df["duracion_min"] = (df["fecha_fin"] - df["fecha_inicio"]).dt.total_seconds() / 60

    df = df.dropna(subset=["origen_id", "destino_id", "precio", "distancia", "duracion_min"])

    df = df[
        (df["precio"] > 0) & (df["precio"] < 500) &
        (df["distancia"] > 0) & (df["distancia"] < 1000) &
        (df["duracion_min"] > 1) & (df["duracion_min"] < 300)
    ].copy()

    df["tipo_vehiculo"] = service
    df["origen_id"] = df["origen_id"].astype("int32")
    df["destino_id"] = df["destino_id"].astype("int32")
    df["precio"] = df["precio"].astype("float32")
    df["distancia"] = df["distancia"].astype("float32")
    df["duracion_min"] = df["duracion_min"].astype("float32")

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
