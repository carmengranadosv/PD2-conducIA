import pandas as pd
from pathlib import Path
import io
import urllib.request

NYC_EVENTS_CSV = "https://data.cityofnewyork.us/api/views/6v4b-5gp4/rows.csv?accessType=DOWNLOAD"

def classify_by_impact(row):
    try:
        asistencia = float(row.get('attendance', 0))
    except (ValueError, TypeError):
        asistencia = 0
    e_type = str(row.get('event_type', '')).upper()
    if asistencia >= 5000 or any(x in e_type for x in ['PARADE', 'MARATHON', 'STREET FESTIVAL']):
        return "Masivo"
    if 1000 <= asistencia < 5000 or any(x in e_type for x in ['STREET FAIR', 'SPECIAL EVENT']):
        return "Medio"
    return "No hay"

def create_major_events(output_path: Path, year: int = 2025, overwrite: bool = False):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        print(f"SKIP Events ({output_path.name})")
        return

    try:
        print(f" Descargando base de datos completa de eventos NYC...")
        req = urllib.request.Request(NYC_EVENTS_CSV, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            df_raw = pd.read_csv(io.BytesIO(response.read()))

        df_raw.columns = [col.lower().replace(' ', '_') for col in df_raw.columns]
        fecha_col = 'date_and_time' 
        
        df_raw['fecha_dt'] = pd.to_datetime(df_raw[fecha_col], errors='coerce').dt.normalize()
        df_year = df_raw[df_raw['fecha_dt'].dt.year == year].copy()

        if df_year.empty:
            df_daily = pd.DataFrame(columns=['fecha', 'evento_tipo', 'num_eventos'])
        else:
            df_year['evento_tipo'] = df_year.apply(classify_by_impact, axis=1)
            impact_order = {"Masivo": 2, "Medio": 1, "No hay": 0}
            df_year['prioridad'] = df_year['evento_tipo'].map(impact_order)
            
            # --- NUEVA LÓGICA DE AGREGACIÓN ---
            # Agrupamos por fecha y calculamos el máximo de prioridad y el conteo de eventos
            df_daily = df_year.groupby('fecha_dt').agg(
                prioridad_max=('prioridad', 'max'),
                num_eventos=('fecha_dt', 'count')
            ).reset_index()

            # Mapeo inverso para recuperar el nombre de la categoría más alta
            inv_map = {2: "Masivo", 1: "Medio", 0: "No hay"}
            df_daily['evento_tipo'] = df_daily['prioridad_max'].map(inv_map)
            df_daily = df_daily.rename(columns={'fecha_dt': 'fecha'})[['fecha', 'evento_tipo', 'num_eventos']]

        # 4. Crear calendario completo
        date_range = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
        df_final = pd.DataFrame({'fecha': date_range})
        df_final = df_final.merge(df_daily, on='fecha', how='left')
        
        # Rellenar valores para días sin eventos
        df_final['evento_tipo'] = df_final['evento_tipo'].fillna("No hay").astype(str)
        df_final['num_eventos'] = df_final['num_eventos'].fillna(0).astype(int)

        df_final.to_parquet(output_path, index=False)
        print(f" EVENTOS Y CONTEO {year} CARGADOS: {df_final['evento_tipo'].value_counts().to_dict()}")

    except Exception as e:
        print(f" ERROR EN DOWNLOAD_EVENTS: {e}")
        raise