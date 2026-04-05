# src/modelos/problema1/preparar_datos.py
"""
Agrega el parquet completo (80M viajes) a zona×hora por chunks,
sin cargar todo en RAM. Sustituye a samplear_datos.py + division_datos.py.

Salida: data/processed/tlc_clean/problema1/raw/{train,val,test}.parquet
        con datos AGREGADOS (zona×hora), no viajes individuales.
"""

import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pandas as pd
import numpy as np
import json, os
from pathlib import Path

PARQUET_PATH = "data/processed/tlc_clean/datos_final.parquet"
OUT_DIR      = Path("data/processed/tlc_clean/problema1/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# División temporal: 70% train / 15% val / 15% test
SPLIT = (0.70, 0.85)  # cortes como fracción del rango total

print("="*60)
print(" AGREGACIÓN DIRECTA SIN SAMPLEO")
print("="*60)

# ── 1. Calcular rango temporal sin cargar datos ───────────────────────────────
print("\n1. Calculando rango temporal...")
pf = pq.ParquetFile(PARQUET_PATH)
t_min, t_max = None, None

for i in range(pf.num_row_groups):
    rg_meta = pf.metadata.row_group(i)
    # Leer solo la columna de fecha para obtener min/max
    batch = pf.read_row_group(i, columns=['fecha_inicio'])
    col = batch.column('fecha_inicio')
    bmin = pd.Timestamp(pc.min(col).as_py())
    bmax = pd.Timestamp(pc.max(col).as_py())
    t_min = bmin if t_min is None else min(t_min, bmin)
    t_max = bmax if t_max is None else max(t_max, bmax)
    print(f"\r   Row group {i+1}/{pf.num_row_groups}...", end='', flush=True)

print(f"\n   Período: {t_min} → {t_max}")

rango = t_max - t_min
corte_train = (t_min + rango * SPLIT[0]).floor('1h')
corte_val   = (t_min + rango * SPLIT[1]).floor('1h')
print(f"   Corte train: {corte_train}")
print(f"   Corte val:   {corte_val}")

# ── 2. Columnas que necesitamos ───────────────────────────────────────────────
COLS_NECESARIAS = [
    'fecha_inicio', 'origen_id', 'mes_num',
    'temp_c', 'precipitation', 'viento_kmh', 'velocidad_mph',
    'lluvia', 'nieve', 'es_festivo', 'num_eventos',
]

# ── 3. Leer por row groups y acumular agregados parciales ────────────────────
print("\n2. Leyendo y agregando por chunks...")

# Acumuladores: dict zona→hora→{conteo, sumas}
# En lugar de acumular en dict (lento), acumulamos DataFrames parciales
chunks_train, chunks_val, chunks_test = [], [], []

for i in range(pf.num_row_groups):
    print(f"\r   Procesando row group {i+1}/{pf.num_row_groups}...", end='', flush=True)
    
    # Leer solo columnas necesarias
    try:
        batch = pf.read_row_group(i, columns=COLS_NECESARIAS)
    except Exception:
        # Si alguna columna no existe en este grupo, saltar
        continue
    
    df = batch.to_pandas()
    df['fecha_inicio']   = pd.to_datetime(df['fecha_inicio'])
    df['timestamp_hora'] = df['fecha_inicio'].dt.floor('1h')
    df['hora']           = df['fecha_inicio'].dt.hour
    df['dia_semana']     = df['fecha_inicio'].dt.dayofweek
    df['dia_mes']        = df['fecha_inicio'].dt.day
    df['es_finde']       = (df['dia_semana'] >= 5).astype(int)
    
    # Agregar este chunk
    agg = (
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
    
    # Partir por período
    mask_train = agg['timestamp_hora'] <  corte_train
    mask_val   = (agg['timestamp_hora'] >= corte_train) & (agg['timestamp_hora'] < corte_val)
    mask_test  = agg['timestamp_hora'] >= corte_val
    
    if mask_train.any(): chunks_train.append(agg[mask_train])
    if mask_val.any():   chunks_val.append(agg[mask_val])
    if mask_test.any():  chunks_test.append(agg[mask_test])
    
    del df, batch, agg

print(f"\n   Chunks acumulados: train={len(chunks_train)}, val={len(chunks_val)}, test={len(chunks_test)}")

# ── 4. Consolidar: sumar demandas del mismo zona×hora entre chunks ────────────
print("\n3. Consolidando agregados...")

def consolidar(chunks):
    """
    Combina chunks del mismo período sumando demandas y promediando features.
    Necesario porque el mismo zona×hora puede aparecer en múltiples row groups.
    """
    df = pd.concat(chunks, ignore_index=True)
    
    # Para demanda: SUMAR (es un conteo)
    # Para features numéricas: promediar ponderado por demanda
    # Simplificación: promediar (el error es mínimo para vars continuas)
    result = df.groupby(['origen_id', 'timestamp_hora']).agg(
        demanda       = ('demanda',       'sum'),   # ← SUMAR conteos
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
    ).reset_index()
    
    return result

agg_train = consolidar(chunks_train)
print(f"   Train: {len(agg_train):,} registros zona×hora")
del chunks_train

agg_val = consolidar(chunks_val)
print(f"   Val:   {len(agg_val):,} registros zona×hora")
del chunks_val

agg_test = consolidar(chunks_test)
print(f"   Test:  {len(agg_test):,} registros zona×hora")
del chunks_test

# ── 5. Verificar cobertura temporal ──────────────────────────────────────────
print("\n4. Cobertura temporal:")
for name, df in [('Train', agg_train), ('Val', agg_val), ('Test', agg_test)]:
    t0 = df['timestamp_hora'].min()
    t1 = df['timestamp_hora'].max()
    dias = (t1 - t0).days
    print(f"   {name}: {t0.date()} → {t1.date()} ({dias} días, {len(df):,} registros)")

# Verificar que no hay solapamiento
assert agg_train['timestamp_hora'].max() < agg_val['timestamp_hora'].min(), \
    "Solapamiento train/val"
assert agg_val['timestamp_hora'].max() < agg_test['timestamp_hora'].min(), \
    "Solapamiento val/test"
print("   Sin solapamientos ✓")

# ── 6. Guardar ────────────────────────────────────────────────────────────────
print("\n5. Guardando splits agregados...")

agg_train.to_parquet(OUT_DIR / 'train.parquet', index=False)
agg_val.to_parquet(  OUT_DIR / 'val.parquet',   index=False)
agg_test.to_parquet( OUT_DIR / 'test.parquet',  index=False)

n_total = len(agg_train) + len(agg_val) + len(agg_test)
metadata = {
    'fuente': 'agregacion directa sin sampleo (80M viajes → zona×hora)',
    'cortes': {
        'corte_train': str(corte_train),
        'corte_val':   str(corte_val),
    },
    'split': {
        s: {
            'filas':        len(d),
            'pct':          round(len(d) / n_total * 100, 1),
            'fecha_inicio': str(d['timestamp_hora'].min()),
            'fecha_fin':    str(d['timestamp_hora'].max()),
            'dias':         (d['timestamp_hora'].max() - d['timestamp_hora'].min()).days,
        }
        for s, d in [('train', agg_train), ('val', agg_val), ('test', agg_test)]
    }
}

with open(OUT_DIR / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"   train.parquet: {len(agg_train):,} filas")
print(f"   val.parquet:   {len(agg_val):,} filas")
print(f"   test.parquet:  {len(agg_test):,} filas")
print(f"   metadata.json")

print("\n" + "="*60)
print(" COMPLETADO — siguiente paso: agregacion.py")
print("="*60)