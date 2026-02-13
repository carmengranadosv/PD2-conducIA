# weather.py
import pandas as pd
import urllib.request
import json
import time
from pathlib import Path

# Coordenadas representativas
BOROUGH_COORDS = {
    "Manhattan":     {"lat": 40.7831, "lon": -73.9712},  # Central Park
    "Brooklyn":      {"lat": 40.6782, "lon": -73.9442},  # Barclays Center aprox
    "Queens":        {"lat": 40.7282, "lon": -73.7949},  # Flushing Meadows aprox
    "Bronx":         {"lat": 40.8448, "lon": -73.8648},  # Bronx Zoo aprox
    "Staten Island": {"lat": 40.5795, "lon": -74.1502},  # Centro de la isla
    "EWR":           {"lat": 40.6895, "lon": -74.1745},  # Newark Airport
}

def _iter_months(start_yyyy_mm: str, end_yyyy_mm: str) -> list[str]:
    """Devuelve lista de meses ['YYYY-MM', ...] entre start y end (inclusive)."""
    start = pd.to_datetime(f"{start_yyyy_mm}-01")
    end = pd.to_datetime(f"{end_yyyy_mm}-01")
    months = pd.date_range(start=start, end=end, freq="MS")  # Month Start
    return [d.strftime("%Y-%m") for d in months]

def _month_start_end(yyyy_mm: str) -> tuple[str, str]:
    """Devuelve (start_date, end_date) en formato ISO YYYY-MM-DD para un mes."""
    start = pd.to_datetime(f"{yyyy_mm}-01")
    end = start + pd.offsets.MonthEnd(1)
    return start.date().isoformat(), end.date().isoformat()

def download_weather_data(
    start_date: str,
    end_date: str,
    output_path: Path,
    overwrite: bool = False,
    api: str = "archive",  # "archive" o "historical_forecast"
):
    """
    Descarga clima horario para varios boroughs y genera dataset enriquecible por hora.

    Params:
      - start_date, end_date: formato 'YYYY-MM' (ej: '2025-01', '2025-12')
      - output_path: Path del parquet de salida
      - overwrite: si False y existe, no regenera
      - api:
          "archive" -> https://archive-api.open-meteo.com/v1/archive
          "historical_forecast" -> https://historical-forecast-api.open-meteo.com/v1/forecast
    """
    if output_path.exists() and not overwrite:
        print(f"SKIP Weather ({output_path.name})")
        return

    if api == "historical_forecast":
        base_url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    else:
        base_url = "https://archive-api.open-meteo.com/v1/archive"

    print(" Descargando Clima ...")
    months = _iter_months(start_date, end_date)

    dfs = []

    for borough, coords in BOROUGH_COORDS.items():
        print(f"   -> {borough}...", end=" ")

        try:
            dfs_borough = []

            for mm in months:
                s, e = _month_start_end(mm)

                url = (
                    f"{base_url}?"
                    f"latitude={coords['lat']}&longitude={coords['lon']}&"
                    f"start_date={s}&end_date={e}&"
                    f"hourly=temperature_2m,precipitation,rain,snowfall,windspeed_10m&"
                    f"timezone=America%2FNew_York"
                )

                with urllib.request.urlopen(url) as response:
                    data = json.loads(response.read().decode())

                # Si no hay hourly (raro), skip
                if "hourly" not in data or "time" not in data["hourly"]:
                    continue

                df_temp = pd.DataFrame(data["hourly"])
                dfs_borough.append(df_temp)

                # Pausa para no saturar API
                time.sleep(0.35)

            if not dfs_borough:
                print("NO DATA")
                continue

            df_b = pd.concat(dfs_borough, ignore_index=True)
            df_b["fecha_hora"] = pd.to_datetime(df_b["time"])
            df_b.drop(columns=["time"], inplace=True)
            df_b["borough"] = borough

            dfs.append(df_b)
            print("OK")

        except Exception as e:
            print(f"ERROR: {e}")

    if not dfs:
        print(" No se pudo descargar clima (dfs vacío).")
        return

    df_final = pd.concat(dfs, ignore_index=True)

    # Flags binarias
    # (si quieres más sensibilidad: usa > 0.0)
    df_final["lluvia"] = (df_final["rain"] > 0.1).astype("int8")
    df_final["nieve"] = (df_final["snowfall"] > 0.1).astype("int8")

    # Quitamos numéricas que ya codificamos, pero dejamos precipitation (útil)
    df_final.drop(columns=["rain", "snowfall"], inplace=True)

    # Renombrar columnas para que queden bonitas
    df_final.rename(
        columns={
            "temperature_2m": "temp_c",
            "windspeed_10m": "viento_kmh",
        },
        inplace=True,
    )

    # Guardar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(output_path, index=False)
    print(f" Guardado Weather: {output_path}")
