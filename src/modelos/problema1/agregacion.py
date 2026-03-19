# ============================================================
# FEATURES — PROBLEMA 1
# ============================================================
# Responsabilidad: a partir de los splits raw, construir las
# features necesarias para los modelos tabulares (baseline,
# Random Forest). Los modelos de secuencias (LSTM, Transformer)
# leerán estos parquets y construirán sus propias secuencias.
#
# IMPORTANTE: la media_hist se calcula SOLO sobre train
# y se aplica a val y test → sin data leakage.
#
# Entrada:  data/processed/tlc_clean/problema1/raw/
# Salida:   data/processed/tlc_clean/problema1/features/
#               train.parquet
#               val.parquet
#               test.parquet
#               metadata.json
# ============================================================

import pandas as pd
import numpy as np
import json
import os

os.makedirs('data/processed/tlc_clean/problema1/features', exist_ok=True)

RAW_DIR = 'data/processed/tlc_clean/problema1/raw'
OUT_DIR = 'data/processed/tlc_clean/problema1/features'

# ============================================================
# PASO 1: CARGAR SPLITS RAW
# ============================================================

print(" Cargando splits raw...")

df_train = pd.read_parquet(f'{RAW_DIR}/train.parquet')
df_val   = pd.read_parquet(f'{RAW_DIR}/val.parquet')
df_test  = pd.read_parquet(f'{RAW_DIR}/test.parquet')

for name, split in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
    print(f"   {name}: {len(split):>10,} viajes")

# ============================================================
# PASO 2: FUNCIÓN DE AGREGACIÓN
# ============================================================

def agregar_por_zona_hora(df):
    """Agrega viajes individuales a demanda por (zona, hora)."""
    df = df.copy()
    df['fecha_inicio']   = pd.to_datetime(df['fecha_inicio'])
    df['timestamp_hora'] = df['fecha_inicio'].dt.floor('1h')
    df['hora']           = df['fecha_inicio'].dt.hour
    df['dia_semana']     = df['fecha_inicio'].dt.dayofweek  # 0=Lun, 6=Dom
    df['dia_mes']        = df['fecha_inicio'].dt.day
    df['es_finde']       = (df['dia_semana'] >= 5).astype(int)

    return (
        df.groupby(['origen_id', 'timestamp_hora'])
        .agg(
            demanda       = ('fecha_inicio',  'count'),
            hora          = ('hora',          'first'),
            dia_semana    = ('dia_semana',    'first'),
            dia_mes       = ('dia_mes',       'first'),
            mes_num       = ('mes_num',       'first'),
            es_finde      = ('es_finde',      'first'),
            temp_c        = ('temp_c',        'mean'),
            precipitation = ('precipitation', 'mean'),
            viento_kmh    = ('viento_kmh',    'mean'),
            velocidad_mph = ('velocidad_mph', 'mean'),
            lluvia        = ('lluvia',        'max'),
            nieve         = ('nieve',         'max'),
            es_festivo    = ('es_festivo',    'max'),
            num_eventos   = ('num_eventos',   'mean'),
        )
        .reset_index()
    )

# ============================================================
# PASO 3: AGREGAR CADA SPLIT
# ============================================================

print("\n Agregando por zona y hora...")

agg_train = agregar_por_zona_hora(df_train)
agg_val   = agregar_por_zona_hora(df_val)
agg_test  = agregar_por_zona_hora(df_test)

for name, agg in [('Train', agg_train), ('Val', agg_val), ('Test', agg_test)]:
    print(f"   {name}: {len(agg):>8,} registros (zona×hora)")

# ============================================================
# PASO 4: FUNCIÓN DE FEATURES TEMPORALES Y LAGS
# ============================================================

def añadir_lags_y_rolling(df):
    """
    Añade lag features y rolling stats.
    Se aplica sobre el dataframe de cada split de forma
    independiente, ordenando por (zona, tiempo).
    """
    df = df.sort_values(['origen_id', 'timestamp_hora']).reset_index(drop=True)

    # Target: demanda en la siguiente hora
    df['target'] = df.groupby('origen_id')['demanda'].shift(-1)

    # Lags
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'lag_{lag}h'] = df.groupby('origen_id')['demanda'].shift(lag)

    # Rolling stats (shift(1) para no incluir el instante actual)
    for window in [3, 24]:
        df[f'roll_mean_{window}h'] = df.groupby('origen_id')['demanda'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'roll_std_{window}h'] = df.groupby('origen_id')['demanda'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0)
        )

    return df

print("\n Calculando lags y rolling stats...")

agg_train = añadir_lags_y_rolling(agg_train)
agg_val   = añadir_lags_y_rolling(agg_val)
agg_test  = añadir_lags_y_rolling(agg_test)

# ============================================================
# PASO 5: MEDIA HISTÓRICA — SOLO DESDE TRAIN (sin leakage)
# ============================================================

print("\n Calculando media histórica (solo desde train)...")

media_train = (
    agg_train.groupby(['origen_id', 'hora', 'dia_semana'])['demanda']
    .mean()
    .reset_index()
    .rename(columns={'demanda': 'media_hist'})
)
global_mean = float(agg_train['demanda'].mean())

def aplicar_media_hist(df):
    merged = df.merge(media_train, on=['origen_id', 'hora', 'dia_semana'], how='left')
    df['media_hist'] = merged['media_hist'].fillna(global_mean).values
    return df

agg_train = aplicar_media_hist(agg_train)
agg_val   = aplicar_media_hist(agg_val)
agg_test  = aplicar_media_hist(agg_test)

# ============================================================
# PASO 6: LIMPIEZA FINAL
# ============================================================

print("\n Limpieza final...")

# Eliminar filas sin target ni lags mínimos
for name, df in [('train', agg_train), ('val', agg_val), ('test', agg_test)]:
    antes = len(df)
    df.dropna(subset=['target', 'lag_1h', 'lag_2h', 'lag_3h'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"   {name}: {antes:,} → {len(df):,} filas (eliminadas {antes - len(df):,} sin target/lags)")

# Rellenar lags largos con media histórica
for df in [agg_train, agg_val, agg_test]:
    for col in ['lag_6h', 'lag_12h', 'lag_24h']:
        df[col] = df[col].fillna(df['media_hist'])

# ============================================================
# PASO 7: GUARDAR
# ============================================================

print("\n Guardando features...")

agg_train.to_parquet(f'{OUT_DIR}/train.parquet', index=False)
agg_val.to_parquet(  f'{OUT_DIR}/val.parquet',   index=False)
agg_test.to_parquet( f'{OUT_DIR}/test.parquet',  index=False)

FEATURE_COLS = [
    'origen_id',
    'hora', 'dia_semana', 'dia_mes', 'mes_num', 'es_finde',
    'demanda',
    'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h',
    'roll_mean_3h', 'roll_std_3h', 'roll_mean_24h', 'roll_std_24h',
    'media_hist',
    'temp_c', 'precipitation', 'viento_kmh', 'velocidad_mph',
    'lluvia', 'nieve', 'es_festivo', 'num_eventos',
]

metadata = {
    'target': 'target',
    'feature_cols': FEATURE_COLS,
    'media_hist_global_fallback': round(global_mean, 4),
    'split': {
        'train': {
            'filas':        len(agg_train),
            'fecha_inicio': str(agg_train['timestamp_hora'].min()),
            'fecha_fin':    str(agg_train['timestamp_hora'].max()),
        },
        'val': {
            'filas':        len(agg_val),
            'fecha_inicio': str(agg_val['timestamp_hora'].min()),
            'fecha_fin':    str(agg_val['timestamp_hora'].max()),
        },
        'test': {
            'filas':        len(agg_test),
            'fecha_inicio': str(agg_test['timestamp_hora'].min()),
            'fecha_fin':    str(agg_test['timestamp_hora'].max()),
        },
    },
    'demanda_stats': {
        'media':  round(float(agg_train['demanda'].mean()), 2),
        'std':    round(float(agg_train['demanda'].std()),  2),
        'min':    int(agg_train['demanda'].min()),
        'max':    int(agg_train['demanda'].max()),
        'median': round(float(agg_train['demanda'].median()), 2),
    },
}

with open(f'{OUT_DIR}/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"    train.parquet  ({len(agg_train):,} filas)")
print(f"    val.parquet    ({len(agg_val):,} filas)")
print(f"    test.parquet   ({len(agg_test):,} filas)")
print(f"    metadata.json")
print(f"\n Siguiente paso: ejecutar baseline.py / lstm.py / ...")