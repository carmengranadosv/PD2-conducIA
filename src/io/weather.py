import pandas as pd
import urllib.request
import json
import time
from pathlib import Path

# Coordenadas representativas 
BOROUGH_COORDS = {
    "Manhattan":     {"lat": 40.7831, "lon": -73.9712}, # Central Park
    "Brooklyn":      {"lat": 40.6782, "lon": -73.9442}, # Cerca de Barclays Center
    "Queens":        {"lat": 40.7282, "lon": -73.7949}, # Flushing Meadows / Puntos medios
    "Bronx":         {"lat": 40.8448, "lon": -73.8648}, # Bronx Zoo area
    "Staten Island": {"lat": 40.5795, "lon": -74.1502}, # Centro de la isla
    "EWR":           {"lat": 40.6895, "lon": -74.1745}  # Newark Airport
}

def download_weather_data(start_date: str, end_date: str, output_path: Path, overwrite: bool = False):
    """
    Descarga clima horario y añade columnas binarias (SI/NO) para lluvia y nieve.
    """
    if output_path.exists() and not overwrite:
        print(f"SKIP Weather ({output_path.name})")
        return

    print(" Descargando Clima ...")
    dfs = []

    for borough, coords in BOROUGH_COORDS.items():
        print(f"   -> {borough}...", end=" ")
        try:
            url = (
                f"https://archive-api.open-meteo.com/v1/archive?"
                f"latitude={coords['lat']}&longitude={coords['lon']}&"
                f"start_date={start_date}-01&end_date={end_date}-28&"
                f"hourly=temperature_2m,precipitation,rain,snowfall,windspeed_10m&"
                f"timezone=America%2FNew_York"
            )
            
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
            
            df_temp = pd.DataFrame(data['hourly'])
            df_temp['fecha_hora'] = pd.to_datetime(df_temp['time'])
            df_temp.drop(columns=['time'], inplace=True)
            df_temp['borough'] = borough 
            
            dfs.append(df_temp)
            print("OK")
            time.sleep(0.5) # Pausa para la API

        except Exception as e:
            print(f"ERROR: {e}")

    if dfs:
        # Concatenamos todos (Manhattan, Queens, etc.)
        df_final = pd.concat(dfs, ignore_index=True)
        
        # 1. Calculamos las columnas binarias (1 = Sí, 0 = No)
        # Usamos un umbral mínimo
        df_final['lluvia'] = (df_final['rain'] > 0.1).astype(int)
        df_final['nieve'] = (df_final['snowfall'] > 0.1).astype(int)
        
        # 2. ELIMINAMOS las columnas numéricas originales
        cols_to_drop = ['rain', 'snowfall']
        df_final.drop(columns=cols_to_drop, inplace=True)

        # 3. Renombramos lo que queda para que quede bonito
        df_final.rename(columns={
            'temperature_2m': 'temp_c',
            'windspeed_10m': 'viento_kmh'
        }, inplace=True)

        # Guardamos
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_parquet(output_path, index=False)