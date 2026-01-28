import pandas as pd
from pathlib import Path

def enrich_with_zones(file_path: Path, lookup_path: Path) -> str:
    """
    Lee un archivo parquet limpio, le añade los nombres de las zonas
    cruzando con la tabla de lookup, y sobreescribe el archivo.
    """
    try:
        # 1. Cargar datos limpios y tabla de zonas
        df = pd.read_parquet(file_path)
        
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

        # 4. Guardar 
        df.to_parquet(file_path, index=False)
        
        return f"Archivo modificado: {file_path.name} (w/ Zones)"

    except Exception as e:
        return f"Error: {file_path.name}: {e}"