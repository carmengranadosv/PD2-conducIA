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
        if weather_lookup_path and weather_lookup_path.exists() and 'origen_barrio' in df.columns:
            df_weather = pd.read_parquet(weather_lookup_path)
            df['temp_hora_join'] = df['fecha_inicio'].dt.floor('h')

            df = df.merge(
                df_weather,
                left_on=['temp_hora_join', 'origen_barrio'],
                right_on=['fecha_hora', 'borough'],
                how='left'
            )

            df.drop(columns=['temp_hora_join', 'fecha_hora', 'borough'], inplace=True, errors='ignore')
            log_msg.append("Weather")


        # ===== FESTIVOS (solo columna es_festivo) =====
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

        # ===== EVENTOS (solo columna hay_evento) =====
        if events_path and events_path.exists():
            df_events = pd.read_parquet(events_path)
            df['fecha_date'] = df['fecha_inicio'].dt.date
            df_events['fecha_date'] = df_events['fecha'].dt.date
            
            # SOLO tomar fecha_date y hay_evento (nada más)
            df = df.merge(
                df_events[['fecha_date', 'hay_evento']],
                on='fecha_date',
                how='left'
            )
            
            # Limpiar columnas temporales
            df.drop(columns=['fecha_date'], inplace=True, errors='ignore')
            
            # Convertir a binario (0 o 1)
            df['hay_evento'] = df['hay_evento'].fillna(0).astype('int8')
            
            log_msg.append("Events")

        # ===== ORDENAR =====
        if 'fecha_inicio' in df.columns:
            df.sort_values(by='fecha_inicio', inplace=True)
            log_msg.append("Sorted")

        # Guardar
        df.to_parquet(file_path, index=False)
        
        return f" {file_path.name} ({', '.join(log_msg)})"

    except Exception as e:
        return f" {file_path.name}: {str(e)}"