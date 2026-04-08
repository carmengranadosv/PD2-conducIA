# src/modelos/problema2/features.py
"""
Añade tasa histórica (calculada solo desde train) y
el cascading input del Problema 1 (predicción de demanda RF).
"""

import pandas as pd
import numpy as np
import json, joblib
from pathlib import Path

RAW_DIR = Path('data/processed/tlc_clean/problema2/raw')
OUT_DIR = Path('data/processed/tlc_clean/problema2/features')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Cargar ─────────────────────────────────────────────────────────────────
print("Cargando datos raw...")
df_train = pd.read_parquet(RAW_DIR / 'train.parquet')
df_val   = pd.read_parquet(RAW_DIR / 'val.parquet')
df_test  = pd.read_parquet(RAW_DIR / 'test.parquet')

for df in [df_train, df_val, df_test]:
    df['ventana_inicio'] = pd.to_datetime(df['ventana_inicio'])

print(f"  Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

# ── 2. Tasa histórica: P(éxito) por zona×hora×dia (solo desde train) ─────────
print("Calculando tasa histórica...")
tasa = (
    df_train.groupby(['origen_id', 'hora', 'dia_semana'])['target']
    .mean().reset_index().rename(columns={'target': 'tasa_historica'})
)
global_tasa = float(df_train['target'].mean())
print(f"  Tasa global de éxito: {global_tasa:.3f}")

def aplicar_tasa(df):
    df = df.merge(tasa, on=['origen_id', 'hora', 'dia_semana'], how='left')
    df['tasa_historica'] = df['tasa_historica'].fillna(global_tasa)
    return df

df_train = aplicar_tasa(df_train)
df_val   = aplicar_tasa(df_val)
df_test  = aplicar_tasa(df_test)

# ── 3. Cascading input del Problema 1 ─────────────────────────────────────────
# Usamos el RF (mejor modelo P1) para predecir demanda en cada zona×ventana.
# El RF predice demanda por zona×hora — la mapeamos a zona×ventana10min.
print("Generando cascading input desde Problema 1 (RF)...")

rf_p1      = joblib.load('models/problema1/baseline_random_forest.pkl')
feat_p1    = json.load(open('reports/problema1/resultados/baseline_results.json'))

# Features que necesita el RF del P1 — reconstruimos desde los datos de P2
# El RF usa: origen_id, hora, dia_semana, dia_mes, mes_num, es_finde,
#            demanda, lags, rolling, media_hist, clima

# Como no tenemos lags en P2, usamos la tasa_historica como proxy de demanda
# y rellenamos los lags con la media de la zona

def generar_cascading(df):
    """
    Genera una estimación de demanda usando variables disponibles.
    El RF de P1 espera lag features — usamos oferta_inferida y tasa_historica
    como proxies razonables para las más importantes (lag_1h, media_hist).
    """
    # Features disponibles directamente
    feats = pd.DataFrame({
        'origen_id':    df['origen_id'],
        'hora':         df['hora'],
        'dia_semana':   df['dia_semana'],
        'dia_mes':      df['ventana_inicio'].dt.day,
        'mes_num':      df['mes_num'],
        'es_finde':     df['es_finde'],
        # Proxies para lag features — oferta como proxy de demanda reciente
        'demanda':      df['oferta_inferida'],
        'lag_1h':       df['oferta_inferida'],
        'lag_2h':       df['oferta_inferida'],
        'lag_3h':       df['oferta_inferida'],
        'lag_6h':       df['oferta_inferida'],
        'lag_12h':      df['oferta_inferida'],
        'lag_24h':      df['tasa_historica'] * df['oferta_inferida'].mean(),
        'roll_mean_3h': df['oferta_inferida'],
        'roll_std_3h':  df['oferta_inferida'] * 0.1,
        'roll_mean_24h':df['oferta_inferida'],
        'roll_std_24h': df['oferta_inferida'] * 0.1,
        'media_hist':   df['tasa_historica'] * df['oferta_inferida'].mean(),
        'temp_c':       df['temp_c'],
        'precipitation':df['precipitation'],
        'viento_kmh':   df['viento_kmh'],
        'velocidad_mph':df['viento_kmh'] * 0.621,  # conversión aproximada
        'lluvia':       df['lluvia'],
        'nieve':        df['nieve'],
        'es_festivo':   df['es_festivo'],
        'num_eventos':  df['num_eventos'],
    })
    pred = rf_p1.predict(feats)
    return np.clip(pred, 0, None)

df_train['demanda_p1'] = generar_cascading(df_train)
df_val['demanda_p1']   = generar_cascading(df_val)
df_test['demanda_p1']  = generar_cascading(df_test)

print(f"  Demanda P1 — media train: {df_train['demanda_p1'].mean():.2f}")

# ── 4. Guardar ────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'origen_id',
    'hora', 'dia_semana', 'es_finde', 'hora_sen', 'hora_cos', 'mes_num',
    'temp_c', 'precipitation', 'viento_kmh', 'lluvia', 'nieve',
    'es_festivo', 'num_eventos',
    'oferta_inferida',
    'tasa_historica',
    'demanda_p1',       # cascading input del Problema 1
    'n_viajes',
    'espera_media',
]

df_train.to_parquet(OUT_DIR / 'train.parquet', index=False)
df_val.to_parquet(  OUT_DIR / 'val.parquet',   index=False)
df_test.to_parquet( OUT_DIR / 'test.parquet',  index=False)

metadata = {
    'target': 'target',
    'feature_cols': FEATURE_COLS,
    'tasa_global': round(global_tasa, 4),
    'cascading_input': 'demanda_p1 generada por RF del Problema 1',
    'split': {
        s: {'filas': len(d)}
        for s, d in [('train', df_train), ('val', df_val), ('test', df_test)]
    }
}
with open(OUT_DIR / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nFeatures guardadas en {OUT_DIR}/")
for s, d in [('train', df_train), ('val', df_val), ('test', df_test)]:
    print(f"  {s}: {len(d):,} filas | target positivo: {d['target'].mean():.1%}")
print("\nSiguiente: baseline.py")