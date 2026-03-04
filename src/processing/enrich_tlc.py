"""Aumentador de contexto: no limpia datos sino que añade
información externa (clima, barrios, festivos) para dar más
valor"""

import pandas as pd
from pathlib import Path

def enrich_data(
    file_path: Path, 
    lookup_path: Path, 
    weather_lookup_path: Path = None,
    holidays_path: Path = None,
    events_path: Path = None,
    traffic_path: Path = None
) -> str:
    """
    Enriquece datos con zonas, clima, festivos, eventos y tráfico.
    """
    try:
        df = pd.read_parquet(file_path)
        log_msg = []

        # ===== ZONAS =====
        if not lookup_path.exists():
            return f" No se encuentra {lookup_path.name}"
            
        zones = pd.read_csv(lookup_path)
        zones = zones.drop_duplicates(subset=['LocationID'])
        
        # Merge origen
        df = df.merge(
            zones[['LocationID', 'Zone', 'Borough']], 
            left_on='origen_id', 
            right_on='LocationID', 
            how='left'
        )
        df.rename(columns={'Zone': 'origen_zona', 'Borough': 'origen_barrio'}, inplace=True)
        df.drop(columns=['LocationID'], inplace=True, errors='ignore')

        # Merge destino
        df = df.merge(
            zones[['LocationID', 'Zone', 'Borough']], 
            left_on='destino_id', 
            right_on='LocationID', 
            how='left'
        )
        df.rename(columns={'Zone': 'destino_zona', 'Borough': 'destino_barrio'}, inplace=True)
        df.drop(columns=['LocationID'], inplace=True, errors='ignore')
        
        log_msg.append("Zones")

        # ===== CLIMA =====
        if weather_lookup_path and weather_lookup_path.exists() and "origen_barrio" in df.columns:
            df_weather = pd.read_parquet(weather_lookup_path)

            # Normaliza timezone (evita merges que no matchean por tz)
            df_weather["fecha_hora"] = pd.to_datetime(df_weather["fecha_hora"]).dt.tz_localize(None)

            # Asegura fecha_inicio sin tz + floor hora
            df["temp_hora_join"] = pd.to_datetime(df["fecha_inicio"]).dt.tz_localize(None).dt.floor("h")

            df = df.merge(
                df_weather,
                left_on=["temp_hora_join", "origen_barrio"],
                right_on=["fecha_hora", "borough"],
                how="left",
            )

            df.drop(columns=["temp_hora_join", "fecha_hora", "borough"], inplace=True, errors="ignore")
            log_msg.append("Weather")


        # ===== IMPUTACIÓN POST-ENRICH (CLIMA) =====

        # flags lluvia/nieve -> 0
        for col in ["lluvia", "nieve"]:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype("int8")

        # precipitación -> 0
        if "precipitation" in df.columns:
            df["precipitation"] = df["precipitation"].fillna(0).astype("float32")

        # temp y viento -> mediana del mes (si todo fuese NaN, cae a 0)
        for col in ["temp_c", "viento_kmh"]:
            if col in df.columns:
                med = df[col].median()
                if pd.isna(med):
                    med = 0.0
                df[col] = df[col].fillna(med).astype("float32")


        # ===== FESTIVOS =====
        if holidays_path and holidays_path.exists():
            df_holidays = pd.read_parquet(holidays_path)
            df['fecha_date'] = df['fecha_inicio'].dt.date
            df_holidays['fecha_date'] = df_holidays['fecha'].dt.date
            
            # SOLO tomar fecha_date y es_festivo
            df = df.merge(
                df_holidays[['fecha_date', 'es_festivo']],
                on='fecha_date',
                how='left'
            )
            
            # Limpiar columnas temporales
            df.drop(columns=['fecha_date'], inplace=True, errors='ignore')
            
            # Convertir a binario (0 o 1)
            df['es_festivo'] = df['es_festivo'].fillna(0).astype('int8')
            
            log_msg.append("Holidays")
            

        # ===== ORDENAR =====
        if 'fecha_inicio' in df.columns:
            df.sort_values(by='fecha_inicio', inplace=True)
            log_msg.append("Sorted")

        # Guardar
        df.to_parquet(file_path, index=False)
        
        return f" {file_path.name} ({', '.join(log_msg)})"

    except Exception as e:
        return f" {file_path.name}: {str(e)}"