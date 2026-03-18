# src/io/download_holidays.py
import pandas as pd
import requests
from pathlib import Path

def _coerce_date(s: str, is_end: bool) -> pd.Timestamp:
    # si viene YYYY-MM → primer/último día del mes
    if len(s) == 7:
        ts = pd.to_datetime(s + "-01")
        if is_end:
            ts = ts + pd.offsets.MonthEnd(0)
        return ts.normalize()
    return pd.to_datetime(s).normalize()

def _get_holidays_us_year(year: int) -> pd.DataFrame:
    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/US"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json())

    # Normalizamos fecha a día
    df["fecha"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return df

def create_holidays_calendar_range(start_date: str, end_date: str, output_path: Path, overwrite: bool = False):
    """
    Calendario de festivos NYC:
      - Federal (global=True)
      - específicos de NY (US-NY)
    Output:
      - fecha
      - es_festivo (0/1)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        print(f"SKIP Holidays ({output_path.name})")
        return

    start = _coerce_date(start_date, is_end=False)
    end   = _coerce_date(end_date,   is_end=True)
    years = range(start.year, end.year + 1)

    print(f" Descargando festivos (NYC/NY) {start_date} → {end_date}...")

    df_all = pd.concat([_get_holidays_us_year(y) for y in years], ignore_index=True)
    df_all = df_all.dropna(subset=["fecha"])

    # Filtro/NY:
    # - global True 
    # - counties contiene US-NY
    df_ny = df_all[
        (df_all.get("global", False) == True)
        | (df_all.get("counties").astype(str).str.contains("US-NY", na=False))
    ].copy()

    # Filtrar rango exacto
    df_ny = df_ny[(df_ny["fecha"] >= start) & (df_ny["fecha"] <= end)].copy()

    # Calendario completo
    date_range = pd.date_range(start=start, end=end, freq="D")
    df_complete = pd.DataFrame({"fecha": date_range})

    df_complete["es_festivo"] = df_complete["fecha"].isin(df_ny["fecha"].drop_duplicates()).astype(int)

    df_complete.to_parquet(output_path, index=False)
    print(f"    Calendario: {len(df_complete)} días, {df_complete['es_festivo'].sum()} festivos")