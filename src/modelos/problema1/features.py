# src/modelos/problema1/features.py
# (puedes borrar agregacion.py — este lo reemplaza)
"""
Calcula lags, rolling stats y media histórica sobre los datos
ya agregados (zona×hora) que vienen de preparar_datos.py.
Sin data leakage: lags calculados sobre el dataset completo,
media histórica solo desde train.
"""

import pandas as pd
import numpy as np
import json, os
from pathlib import Path

RAW_DIR = Path('data/processed/tlc_clean/problema1/raw')
OUT_DIR = Path('data/processed/tlc_clean/problema1/features')
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(RAW_DIR / 'metadata.json') as f:
    meta_raw = json.load(f)

CORTE_TRAIN = pd.Timestamp(meta_raw['cortes']['corte_train'])
CORTE_VAL   = pd.Timestamp(meta_raw['cortes']['corte_val'])

# ── 1. Cargar y concatenar (ya son zona×hora, caben en RAM) ──────────────────
print("Cargando datos agregados...")
agg = pd.concat([
    pd.read_parquet(RAW_DIR / 'train.parquet'),
    pd.read_parquet(RAW_DIR / 'val.parquet'),
    pd.read_parquet(RAW_DIR / 'test.parquet'),
], ignore_index=True)

agg['timestamp_hora'] = pd.to_datetime(agg['timestamp_hora'])
print(f"  Total registros zona×hora: {len(agg):,}")

# ── 2. Lags y rolling sobre el dataset COMPLETO (sin leakage) ────────────────
print("Calculando lags y rolling...")
agg = agg.sort_values(['origen_id', 'timestamp_hora']).reset_index(drop=True)

agg['target'] = agg.groupby('origen_id')['demanda'].shift(-1)

for lag in [1, 2, 3, 6, 12, 24]:
    agg[f'lag_{lag}h'] = agg.groupby('origen_id')['demanda'].shift(lag)

for window in [3, 24]:
    agg[f'roll_mean_{window}h'] = agg.groupby('origen_id')['demanda'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    agg[f'roll_std_{window}h'] = agg.groupby('origen_id')['demanda'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0)
    )

# ── 3. Partir por fecha DESPUÉS de calcular lags ─────────────────────────────
print("Partiendo en train / val / test...")
agg_train = agg[agg['timestamp_hora'] <  CORTE_TRAIN].copy()
agg_val   = agg[(agg['timestamp_hora'] >= CORTE_TRAIN) & (agg['timestamp_hora'] < CORTE_VAL)].copy()
agg_test  = agg[agg['timestamp_hora'] >= CORTE_VAL].copy()

# ── 4. Media histórica solo desde train ──────────────────────────────────────
print("Calculando media histórica (solo train)...")
media_train = (
    agg_train.groupby(['origen_id', 'hora', 'dia_semana'])['demanda']
    .mean().reset_index().rename(columns={'demanda': 'media_hist'})
)
global_mean = float(agg_train['demanda'].mean())

def aplicar_media_hist(df):
    df = df.merge(media_train, on=['origen_id', 'hora', 'dia_semana'], how='left')
    df['media_hist'] = df['media_hist'].fillna(global_mean)
    return df

agg_train = aplicar_media_hist(agg_train)
agg_val   = aplicar_media_hist(agg_val)
agg_test  = aplicar_media_hist(agg_test)

# ── 5. Limpieza ───────────────────────────────────────────────────────────────
print("Limpieza...")
for name, df in [('train', agg_train), ('val', agg_val), ('test', agg_test)]:
    antes = len(df)
    df.dropna(subset=['target', 'lag_1h', 'lag_2h', 'lag_3h'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  {name}: {antes:,} → {len(df):,} ({antes-len(df):,} eliminadas)")

for df in [agg_train, agg_val, agg_test]:
    for col in ['lag_6h', 'lag_12h', 'lag_24h']:
        df[col] = df[col].fillna(df['media_hist'])

# ── 6. Guardar ────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'origen_id', 'hora', 'dia_semana', 'dia_mes', 'mes_num', 'es_finde',
    'demanda',
    'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h',
    'roll_mean_3h', 'roll_std_3h', 'roll_mean_24h', 'roll_std_24h',
    'media_hist',
    'temp_c', 'precipitation', 'viento_kmh', 'velocidad_mph',
    'lluvia', 'nieve', 'es_festivo', 'num_eventos',
]

agg_train.to_parquet(OUT_DIR / 'train.parquet', index=False)
agg_val.to_parquet(  OUT_DIR / 'val.parquet',   index=False)
agg_test.to_parquet( OUT_DIR / 'test.parquet',  index=False)

metadata = {
    'target': 'target',
    'feature_cols': FEATURE_COLS,
    'media_hist_global_fallback': round(global_mean, 4),
    'split': {
        s: {
            'filas':        len(d),
            'fecha_inicio': str(d['timestamp_hora'].min()),
            'fecha_fin':    str(d['timestamp_hora'].max()),
        }
        for s, d in [('train', agg_train), ('val', agg_val), ('test', agg_test)]
    },
    'demanda_stats': {
        'media':  round(float(agg_train['demanda'].mean()), 2),
        'std':    round(float(agg_train['demanda'].std()),  2),
        'min':    int(agg_train['demanda'].min()),
        'max':    int(agg_train['demanda'].max()),
        'median': round(float(agg_train['demanda'].median()), 2),
    },
}

with open(OUT_DIR / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"\nFeatures guardadas en {OUT_DIR}/")
for s, d in [('train', agg_train), ('val', agg_val), ('test', agg_test)]:
    print(f"  {s}.parquet: {len(d):,} filas")
print("\nSiguiente paso: baseline.py / lstm.py")