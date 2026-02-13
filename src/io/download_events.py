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
    # Eventos masivos NYC 2025
    events = [

        # ENERO
        '2025-01-01',  # New Year aftermath (resaca NYE)

        # FEBRERO
        '2025-02-09',  # Super Bowl

        # MARZO
        '2025-03-16',  # NYC Half Marathon
        '2025-03-17',  # St Patrick's Day Parade

        # ABRIL
        '2025-04-19',  # Tribeca Film Festival (inicio aprox)
        
        # MAYO
        '2025-05-04',  # Five Boro Bike Tour
        '2025-05-26',  # Fleet Week NYC (inicio aprox)

        # JUNIO
        '2025-06-08',  # Puerto Rican Day Parade
        '2025-06-29',  # NYC Pride Parade

        # JULIO
        '2025-07-04',  # Macy’s July 4th Fireworks

        # AGOSTO
        '2025-08-25',  # US Open (inicio aprox)

        # SEPTIEMBRE
        '2025-09-07',  # US Open Finals (aprox)

        # OCTUBRE
        '2025-10-31',  # Halloween Parade (Greenwich Village)

        # NOVIEMBRE
        '2025-11-02',  # NYC Marathon
        '2025-11-27',  # Macy’s Thanksgiving Parade
        '2025-11-28',  # Black Friday (shopping peak Manhattan + outlets)
        '2025-11-29',  # Black Friday weekend effect
        '2025-11-30',  # Cyber Weekend mobility impact

        # DICIEMBRE
        '2025-12-03',  # Rockefeller Tree Lighting (aprox)
        '2025-12-31',  # New Year’s Eve Times Square
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