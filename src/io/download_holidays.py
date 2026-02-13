import pandas as pd
from pathlib import Path

def create_holidays_calendar(start_year: int, end_year: int, output_path: Path, overwrite: bool = False):
    """
    Crea calendario simple de festivos: fecha + es_festivo (1 o 0)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists() and not overwrite:
        print(f"SKIP Holidays ({output_path.name})")
        return
    
    print(f" Creando calendario de festivos {start_year}-{end_year}...")
    
    # Lista de festivos importantes
    holidays = []
    for year in range(start_year, end_year + 1):
        holidays.extend([
            f'{year}-01-01',  # New Year's Day
            f'{year}-01-20',  # Martin Luther King Jr. Day (3er lunes enero 2025)
            f'{year}-02-17',  # Presidents Day (3er lunes febrero 2025)
            f'{year}-05-26',  # Memorial Day (último lunes mayo 2025)
            f'{year}-07-04',  # Independence Day
            f'{year}-09-01',  # Labor Day (1er lunes septiembre 2025)
            f'{year}-10-13',  # Columbus Day (2º lunes octubre 2025)
            f'{year}-11-11',  # Veterans Day
            f'{year}-11-27',  # Thanksgiving Day (4º jueves noviembre 2025)
            f'{year}-12-25',  # Christmas Day
        ])
    
    # DataFrame de festivos
    df_holidays = pd.DataFrame({'fecha': holidays})
    df_holidays['fecha'] = pd.to_datetime(df_holidays['fecha'])
    df_holidays['es_festivo'] = 1
    
    # Rango completo de fechas del período
    start_date = f'{start_year}-01-01'
    end_date = f'{end_year}-12-31'
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # DataFrame completo
    df_complete = pd.DataFrame({'fecha': date_range})
    df_complete = df_complete.merge(df_holidays, on='fecha', how='left')
    df_complete['es_festivo'] = df_complete['es_festivo'].fillna(0).astype(int)
    
    # Guardar
    df_complete.to_parquet(output_path, index=False)
    
    n_festivos = df_complete['es_festivo'].sum()
    print(f"    Calendario: {len(df_complete)} días, {n_festivos} festivos")