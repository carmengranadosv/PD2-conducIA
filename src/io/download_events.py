import pandas as pd
from pathlib import Path

def create_major_events(output_path: Path, overwrite: bool = False):
    """
    Crea calendario simple de eventos: fecha + hay_evento (1 o 0)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists() and not overwrite:
        print(f"SKIP Events ({output_path.name})")
        return
    
    print(" Creando calendario de eventos mayores...")
    
    # Eventos importantes 2025
    events = [
        '2025-01-01',  # New Year aftermath
        '2025-02-09',  # Super Bowl
        '2025-03-17',  # St Patrick's Parade
        '2025-06-29',  # Pride Parade
        '2025-07-04',  # July 4th Fireworks
        '2025-11-02',  # NYC Marathon
        '2025-11-27',  # Thanksgiving Parade
        '2025-12-31',  # NYE Times Square
    ]
    
    # DataFrame de eventos
    df_events = pd.DataFrame({'fecha': events})
    df_events['fecha'] = pd.to_datetime(df_events['fecha'])
    df_events['hay_evento'] = 1
    
    # Rango completo 2025
    date_range = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
    df_complete = pd.DataFrame({'fecha': date_range})
    df_complete = df_complete.merge(df_events, on='fecha', how='left')
    df_complete['hay_evento'] = df_complete['hay_evento'].fillna(0).astype(int)
    
    # Guardar
    df_complete.to_parquet(output_path, index=False)
    
    n_eventos = df_complete['hay_evento'].sum()
    print(f"    Calendario: {len(df_complete)} días, {n_eventos} eventos")