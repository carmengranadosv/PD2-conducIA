import pandas as pd
from pathlib import Path

def enrich_data(file_path: Path, lookup_path: Path, weather_lookup_path: Path = None) -> str:
    """
    Lee un archivo parquet limpio.
    1. Añade nombres de zonas (Barrio/Zona) usando el lookup de TLC.
    2. Añade datos de clima correspondientes a la hora y el barrio de origen.
    3. Ordena cronológicamente
    Sobreescribe el archivo original.
    """
    try:
        # 1. Cargar datos limpios y tabla de zonas
        df = pd.read_parquet(file_path)
        log_msg = []

        # PARTE 1: ENRIQUECIMIENTO DE ZONAS
        if not lookup_path.exists():
            return f" No se encuentra {lookup_path.name}"
            
        zones = pd.read_csv(lookup_path)
        
        # Asegurar que no hay duplicados en el ID
        zones = zones.drop_duplicates(subset=['LocationID'])
        
        # Preparamos el diccionario para mapear
        
        # 2. MERGE PARA ORIGEN
        df = df.merge(
            zones[['LocationID', 'Zone', 'Borough']], 
            left_on='origen_id', 
            right_on='LocationID', 
            how='left'
        )
        # Renombramos y limpiamos
        df.rename(columns={'Zone': 'origen_zona', 'Borough': 'origen_barrio'}, inplace=True)
        df.drop(columns=['LocationID'], inplace=True)

        # 3. MERGE PARA DESTINO
        df = df.merge(
            zones[['LocationID', 'Zone', 'Borough']], 
            left_on='destino_id', 
            right_on='LocationID', 
            how='left'
        )
        df.rename(columns={'Zone': 'destino_zona', 'Borough': 'destino_barrio'}, inplace=True)
        df.drop(columns=['LocationID'], inplace=True)

        log_msg.append("Zones")

        # PARTE 2: ENRIQUECIMIENTO DE CLIMA
        # Solo ejecutamos si tenemos el archivo de clima y si ya tenemos el barrio de origen
        if weather_lookup_path and weather_lookup_path.exists() and 'origen_barrio' in df.columns:
            
            df_weather = pd.read_parquet(weather_lookup_path)
            
            # 1. Creamos una columna temporal redondeada a la hora para hacer el match
            df['temp_hora_join'] = df['fecha_inicio'].dt.floor('h')

            # 2. MERGE DOBLE:
            # Cruzamos por HORA y por BARRIO DE ORIGEN.
            df = df.merge(
                df_weather,
                left_on=['temp_hora_join', 'origen_barrio'], # Claves en tus viajes
                right_on=['fecha_hora', 'borough'],          # Claves en el clima
                how='left'
            )

            # 3. Limpieza de columnas auxiliares que sobran tras el merge
            cols_to_drop = ['temp_hora_join', 'fecha_hora', 'borough']
            df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
            
            log_msg.append("Weather")

        # 3. Ordenar cronológicamente
        df.sort_values(by='fecha_inicio', inplace=True)

        # PARTE 3: ORDENAR POR FECHA
        if 'fecha_inicio' in df.columns:
            df['fecha_inicio'] = pd.to_datetime(df['fecha_inicio'])
            df.sort_values(by='fecha_inicio', inplace=True)
            log_msg.append("Sorted")

        # Guardar 
        df.to_parquet(file_path, index=False)
        
        return f"Archivo modificado: {file_path.name} ({', '.join(log_msg)})"

    except Exception as e:
        return f"Error: {file_path.name}: {str(e)}"