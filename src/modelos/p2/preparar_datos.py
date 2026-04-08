# src/modelos/problema2/preparar_datos.py
"""
Agrega el dataset raw a zona×ventana10min y añade:
  - oferta_inferida: taxis que terminaron viaje en esa zona en los 20min previos
  - target: 1 si la zona tiene tasa_exito > media de su ventana temporal
            (zona relativamente buena en ese momento — ~50% positivos)

División temporal: 70/15/15 por fecha (igual que Problema 1).
"""

import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
import numpy as np
import json, os
from pathlib import Path

PARQUET_PATH = 'data/processed/tlc_clean/datos_final.parquet'
OUT_DIR      = Path('data/processed/tlc_clean/problema2/raw')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT = (0.70, 0.85)

print("="*60)
print(" PROBLEMA 2 — PREPARACIÓN DE DATOS")
print("="*60)

# ── 1. Rango temporal ─────────────────────────────────────────────────────────
print("\n1. Calculando rango temporal...")
pf = pq.ParquetFile(PARQUET_PATH)
t_min, t_max = None, None

for i in range(pf.num_row_groups):
    batch = pf.read_row_group(i, columns=['fecha_inicio'])
    col   = batch.column('fecha_inicio')
    bmin  = pd.Timestamp(pc.min(col).as_py())
    bmax  = pd.Timestamp(pc.max(col).as_py())
    t_min = bmin if t_min is None else min(t_min, bmin)
    t_max = bmax if t_max is None else max(t_max, bmax)
    print(f"\r   Row group {i+1}/{pf.num_row_groups}...", end='', flush=True)

rango       = t_max - t_min
corte_train = (t_min + rango * SPLIT[0]).floor('10min')
corte_val   = (t_min + rango * SPLIT[1]).floor('10min')
print(f"\n   Período: {t_min} → {t_max}")
print(f"   Corte train: {corte_train}")
print(f"   Corte val:   {corte_val}")

# ── 2. Calcular oferta inferida global ────────────────────────────────────────
print("\n2. Calculando oferta inferida (destino × ventana_fin)...")

chunks_oferta = []
for i in range(pf.num_row_groups):
    batch = pf.read_row_group(i, columns=['destino_id', 'fecha_fin'])
    df_o  = batch.to_pandas()
    df_o['fecha_fin']      = pd.to_datetime(df_o['fecha_fin'])
    df_o['ventana_oferta'] = df_o['fecha_fin'].dt.floor('10min')
    agg = (df_o.groupby(['destino_id', 'ventana_oferta'])
               .size().reset_index(name='oferta_inferida'))
    chunks_oferta.append(agg)
    print(f"\r   Row group {i+1}/{pf.num_row_groups}...", end='', flush=True)

oferta = (pd.concat(chunks_oferta, ignore_index=True)
            .groupby(['destino_id', 'ventana_oferta'])['oferta_inferida']
            .sum().reset_index())
del chunks_oferta
print(f"\n   Registros oferta: {len(oferta):,}")

# ── 3. Agregar viajes a zona×ventana10min ─────────────────────────────────────
print("\n3. Agregando viajes por zona y ventana de 10 min...")

COLS = [
    'fecha_inicio', 'origen_id', 'espera_min',
    'hora', 'dia_semana', 'es_fin_semana', 'hora_sen', 'hora_cos',
    'mes_num', 'franja_horaria',
    'temp_c', 'precipitation', 'viento_kmh', 'lluvia', 'nieve',
    'es_festivo', 'num_eventos',
]

chunks_train, chunks_val, chunks_test = [], [], []

for i in range(pf.num_row_groups):
    batch = pf.read_row_group(i, columns=COLS)
    df    = batch.to_pandas()
    df['fecha_inicio']   = pd.to_datetime(df['fecha_inicio'])
    df['ventana_inicio'] = df['fecha_inicio'].dt.floor('10min')
    df['exito']          = (df['espera_min'] <= 10).astype(np.int8)

    agg = df.groupby(['origen_id', 'ventana_inicio']).agg(
        n_viajes      = ('espera_min',    'count'),
        tasa_exito    = ('exito',         'mean'),
        espera_media  = ('espera_min',    'mean'),
        hora          = ('hora',          'first'),
        dia_semana    = ('dia_semana',    'first'),
        es_finde      = ('es_fin_semana', 'first'),
        hora_sen      = ('hora_sen',      'first'),
        hora_cos      = ('hora_cos',      'first'),
        mes_num       = ('mes_num',       'first'),
        temp_c        = ('temp_c',        'mean'),
        precipitation = ('precipitation', 'mean'),
        viento_kmh    = ('viento_kmh',    'mean'),
        lluvia        = ('lluvia',        'max'),
        nieve         = ('nieve',         'max'),
        es_festivo    = ('es_festivo',    'max'),
        num_eventos   = ('num_eventos',   'mean'),
    ).reset_index()

    # Merge con oferta
    agg = agg.merge(
        oferta,
        left_on  = ['origen_id', 'ventana_inicio'],
        right_on = ['destino_id', 'ventana_oferta'],
        how='left'
    ).drop(columns=['destino_id', 'ventana_oferta'], errors='ignore')
    agg['oferta_inferida'] = agg['oferta_inferida'].fillna(0).astype(np.float32)

    # Partir por período
    mask_tr = agg['ventana_inicio'] <  corte_train
    mask_v  = (agg['ventana_inicio'] >= corte_train) & (agg['ventana_inicio'] < corte_val)
    mask_te = agg['ventana_inicio'] >= corte_val

    if mask_tr.any(): chunks_train.append(agg[mask_tr])
    if mask_v.any():  chunks_val.append(agg[mask_v])
    if mask_te.any(): chunks_test.append(agg[mask_te])

    del df, batch, agg
    print(f"\r   Row group {i+1}/{pf.num_row_groups}...", end='', flush=True)

print()

# ── 4. Consolidar ─────────────────────────────────────────────────────────────
print("\n4. Consolidando...")

def consolidar(chunks):
    df = pd.concat(chunks, ignore_index=True)
    return df.groupby(['origen_id', 'ventana_inicio']).agg(
        n_viajes        = ('n_viajes',        'sum'),
        tasa_exito      = ('tasa_exito',      'mean'),
        espera_media    = ('espera_media',    'mean'),
        hora            = ('hora',            'first'),
        dia_semana      = ('dia_semana',      'first'),
        es_finde        = ('es_finde',        'first'),
        hora_sen        = ('hora_sen',        'first'),
        hora_cos        = ('hora_cos',        'first'),
        mes_num         = ('mes_num',         'first'),
        temp_c          = ('temp_c',          'mean'),
        precipitation   = ('precipitation',   'mean'),
        viento_kmh      = ('viento_kmh',      'mean'),
        lluvia          = ('lluvia',          'max'),
        nieve           = ('nieve',           'max'),
        es_festivo      = ('es_festivo',      'max'),
        num_eventos     = ('num_eventos',     'mean'),
        oferta_inferida = ('oferta_inferida', 'sum'),
    ).reset_index()

agg_train = consolidar(chunks_train); del chunks_train
agg_val   = consolidar(chunks_val);   del chunks_val
agg_test  = consolidar(chunks_test);  del chunks_test

# ── 5. TARGET: top-33% zonas por tasa_exito en cada ventana ──────────────────
# Solo el tercio mejor de zonas en cada momento = 1.
# Más exigente → mejor discriminación → ~33% positivos.
print("\n5. Calculando target relativo...")

for df in [agg_train, agg_val, agg_test]:
    p67 = df.groupby('ventana_inicio')['tasa_exito'].transform(
        lambda x: x.quantile(0.67)
    )
    df['target'] = (df['tasa_exito'] >= p67).astype(np.int8)

print(f"   Train: {len(agg_train):,} | target positivo: {agg_train['target'].mean():.1%}")
print(f"   Val:   {len(agg_val):,}   | target positivo: {agg_val['target'].mean():.1%}")
print(f"   Test:  {len(agg_test):,}  | target positivo: {agg_test['target'].mean():.1%}")

# Verificar cobertura temporal
print("\n   Cobertura temporal:")
for name, df in [('Train', agg_train), ('Val', agg_val), ('Test', agg_test)]:
    t0 = df['ventana_inicio'].min()
    t1 = df['ventana_inicio'].max()
    print(f"   {name}: {t0.date()} → {t1.date()} ({(t1-t0).days} días)")

assert agg_train['ventana_inicio'].max() < agg_val['ventana_inicio'].min(), \
    "Solapamiento train/val"
assert agg_val['ventana_inicio'].max() < agg_test['ventana_inicio'].min(), \
    "Solapamiento val/test"
print("   Sin solapamientos ✓")

# ── 6. Guardar ────────────────────────────────────────────────────────────────
print("\n6. Guardando...")
agg_train.to_parquet(OUT_DIR / 'train.parquet', index=False)
agg_val.to_parquet(  OUT_DIR / 'val.parquet',   index=False)
agg_test.to_parquet( OUT_DIR / 'test.parquet',  index=False)

n_total  = len(agg_train) + len(agg_val) + len(agg_test)
metadata = {
    'cortes': {
        'corte_train': str(corte_train),
        'corte_val':   str(corte_val),
    },
    'target': 'target (1 = zona con tasa_exito > media de su ventana temporal)',
    'split': {
        s: {
            'filas':        len(d),
            'pct':          round(len(d) / n_total * 100, 1),
            'fecha_inicio': str(d['ventana_inicio'].min()),
            'fecha_fin':    str(d['ventana_inicio'].max()),
            'dias':         (d['ventana_inicio'].max() - d['ventana_inicio'].min()).days,
            'pct_positivos': round(float(d['target'].mean()) * 100, 1),
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
print(" COMPLETADO — siguiente paso: features.py")
print("="*60)