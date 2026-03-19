# src/modelos/problema1/exportar_predicciones.py

"""
Exporta predicciones en formato normalizado (0-1) para integración
con otros módulos del proyecto
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from sklearn.preprocessing import MinMaxScaler

print("="*80)
print("📤 EXPORTAR PREDICCIONES (demanda_score 0-1)")
print("="*80)

# ============================================================
# CONFIGURACIÓN
# ============================================================

# Directorios
DATA_DIR = Path('data/processed/tlc_clean/problema1/features')
MODEL_DIR = Path('models/problema1')
OUTPUT_DIR = Path('data/outputs/problema1')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Qué splits exportar
EXPORT_SPLITS = ['test']  # Cambia a ['train', 'val', 'test'] si necesitas todos

# Método de normalización
NORMALIZATION_METHOD = 'minmax'  # 'minmax' o 'percentile'

print(f"\n📋 Configuración:")
print(f"   Splits: {EXPORT_SPLITS}")
print(f"   Normalización: {NORMALIZATION_METHOD}")

# ============================================================
# CARGAR MODELO Y METADATA
# ============================================================

print(f"\n⏳ Cargando modelo y metadata...")

# Cargar Random Forest (mejor modelo)
rf = joblib.load(MODEL_DIR / 'baseline_random_forest.pkl')

with open(DATA_DIR / 'metadata.json') as f:
    meta = json.load(f)

FEATURE_COLS = meta['feature_cols']
TARGET = meta['target']

print(f"✅ Modelo cargado: Random Forest")
print(f"   Features: {len(FEATURE_COLS)}")

# ============================================================
# FUNCIÓN: NORMALIZAR DEMANDA A SCORE 0-1
# ============================================================

def normalizar_demanda(demanda_pred, method='minmax'):
    """
    Normaliza predicciones de demanda a score 0-1
    
    Args:
        demanda_pred: array de predicciones
        method: 'minmax' o 'percentile'
    
    Returns:
        array normalizado 0-1
    """
    demanda_pred = np.array(demanda_pred).reshape(-1, 1)
    
    if method == 'minmax':
        # Min-Max: 0 = mínimo, 1 = máximo
        scaler = MinMaxScaler(feature_range=(0, 1))
        score = scaler.fit_transform(demanda_pred).flatten()
        
    elif method == 'percentile':
        # Basado en percentiles (más robusto a outliers)
        p_min = np.percentile(demanda_pred, 5)   # Percentil 5
        p_max = np.percentile(demanda_pred, 95)  # Percentil 95
        
        score = (demanda_pred.flatten() - p_min) / (p_max - p_min)
        score = np.clip(score, 0, 1)  # Forzar rango 0-1
        
    else:
        raise ValueError(f"Método '{method}' no reconocido")
    
    return score

# ============================================================
# PROCESAR CADA SPLIT
# ============================================================

for split_name in EXPORT_SPLITS:
    print(f"\n{'='*80}")
    print(f"📊 PROCESANDO SPLIT: {split_name.upper()}")
    print(f"{'='*80}")
    
    # Cargar datos
    df = pd.read_parquet(DATA_DIR / f'{split_name}.parquet')
    
    print(f"\n⏳ Datos cargados: {len(df):,} registros")
    print(f"   Período: {df['timestamp_hora'].min()} → {df['timestamp_hora'].max()}")
    
    # Predecir
    print(f"\n🔮 Generando predicciones...")
    X = df[FEATURE_COLS]
    y_pred = np.clip(rf.predict(X), 0, None)  # No permitir negativos
    
    print(f"✅ Predicciones generadas")
    print(f"   Min: {y_pred.min():.2f} viajes/hora")
    print(f"   Max: {y_pred.max():.2f} viajes/hora")
    print(f"   Media: {y_pred.mean():.2f} viajes/hora")
    
    # Normalizar a score 0-1
    print(f"\n📏 Normalizando a score 0-1 ({NORMALIZATION_METHOD})...")
    demanda_score = normalizar_demanda(y_pred, method=NORMALIZATION_METHOD)
    
    print(f"✅ Normalización completada")
    print(f"   Min score: {demanda_score.min():.4f}")
    print(f"   Max score: {demanda_score.max():.4f}")
    print(f"   Media score: {demanda_score.mean():.4f}")
    
    # Crear DataFrame output
    df_export = pd.DataFrame({
        'fecha_hora': df['timestamp_hora'],
        'id_zona': df['origen_id'],
        'demanda_score': demanda_score
    })
    
    # Ordenar por fecha y zona
    df_export = df_export.sort_values(['fecha_hora', 'id_zona']).reset_index(drop=True)
    
    # ============================================================
    # EXPORTAR EN MÚLTIPLES FORMATOS
    # ============================================================
    
    print(f"\n💾 Guardando outputs...")
    
    # 1. CSV (más compatible)
    csv_path = OUTPUT_DIR / f'predicciones_{split_name}.csv'
    df_export.to_csv(csv_path, index=False)
    print(f"   ✅ CSV: {csv_path}")
    
    # 2. Parquet (más eficiente)
    parquet_path = OUTPUT_DIR / f'predicciones_{split_name}.parquet'
    df_export.to_parquet(parquet_path, index=False)
    print(f"   ✅ Parquet: {parquet_path}")
    
    # 3. JSON (para APIs)
    json_path = OUTPUT_DIR / f'predicciones_{split_name}.json'
    df_export.to_json(json_path, orient='records', date_format='iso', indent=2)
    print(f"   ✅ JSON: {json_path}")
    
    # ============================================================
    # ESTADÍSTICAS POR ZONA
    # ============================================================
    
    print(f"\n📊 Generando estadísticas por zona...")
    
    stats_zona = df_export.groupby('id_zona').agg({
        'demanda_score': ['mean', 'min', 'max', 'std', 'count']
    }).round(4)
    
    stats_zona.columns = ['score_medio', 'score_min', 'score_max', 'score_std', 'n_horas']
    stats_zona = stats_zona.sort_values('score_medio', ascending=False).reset_index()
    
    stats_path = OUTPUT_DIR / f'estadisticas_zonas_{split_name}.csv'
    stats_zona.to_csv(stats_path, index=False)
    print(f"   ✅ Estadísticas: {stats_path}")
    
    # Mostrar top 10
    print(f"\n   📍 Top 10 zonas por score medio:")
    print(f"\n   {'Zona':>8} {'Score Medio':>12} {'Score Max':>12} {'Horas':>8}")
    print(f"   {'-'*44}")
    for _, row in stats_zona.head(10).iterrows():
        print(f"   {row['id_zona']:>8.0f} {row['score_medio']:>12.4f} {row['score_max']:>12.4f} {row['n_horas']:>8.0f}")

# ============================================================
# MUESTRA DEL OUTPUT
# ============================================================

print(f"\n{'='*80}")
print("👁️  MUESTRA DEL OUTPUT")
print("="*80)

# Leer el último archivo exportado
df_muestra = pd.read_csv(OUTPUT_DIR / f'predicciones_{EXPORT_SPLITS[-1]}.csv')

print(f"\n📋 Primeras 20 filas:")
print(df_muestra.head(20).to_string(index=False))

print(f"\n📊 Resumen estadístico:")
print(df_muestra['demanda_score'].describe())

# ============================================================
# METADATA DEL EXPORT
# ============================================================

metadata_export = {
    'fecha_generacion': pd.Timestamp.now().isoformat(),
    'modelo': 'Random Forest',
    'modelo_path': str(MODEL_DIR / 'baseline_random_forest.pkl'),
    'normalizacion': NORMALIZATION_METHOD,
    'splits_exportados': EXPORT_SPLITS,
    'columnas': ['fecha_hora', 'id_zona', 'demanda_score'],
    'formato_fecha': 'ISO 8601 (YYYY-MM-DD HH:MM:SS)',
    'rango_score': [0.0, 1.0],
    'descripcion': {
        'fecha_hora': 'Timestamp hora inicio (UTC)',
        'id_zona': 'ID zona TLC (LocationID)',
        'demanda_score': 'Score normalizado 0-1 (0=baja, 1=alta demanda)'
    },
    'estadisticas': {
        split: {
            'n_registros': len(pd.read_csv(OUTPUT_DIR / f'predicciones_{split}.csv')),
            'zonas_unicas': pd.read_csv(OUTPUT_DIR / f'predicciones_{split}.csv')['id_zona'].nunique(),
            'periodo': {
                'inicio': str(pd.read_csv(OUTPUT_DIR / f'predicciones_{split}.csv')['fecha_hora'].min()),
                'fin': str(pd.read_csv(OUTPUT_DIR / f'predicciones_{split}.csv')['fecha_hora'].max())
            }
        }
        for split in EXPORT_SPLITS
    }
}

metadata_path = OUTPUT_DIR / 'predicciones_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata_export, f, indent=2)

print(f"\n✅ Metadata guardado: {metadata_path}")

# ============================================================
# RESUMEN FINAL
# ============================================================

print("\n" + "="*80)
print("✅ EXPORTACIÓN COMPLETADA")
print("="*80)

print(f"\n📁 Archivos generados:")
for split in EXPORT_SPLITS:
    print(f"\n   Split: {split}")
    print(f"   ├─ predicciones_{split}.csv")
    print(f"   ├─ predicciones_{split}.parquet")
    print(f"   ├─ predicciones_{split}.json")
    print(f"   └─ estadisticas_zonas_{split}.csv")

print(f"\n   📋 predicciones_metadata.json")

print(f"\n📊 Total registros exportados:")
for split in EXPORT_SPLITS:
    n = len(pd.read_csv(OUTPUT_DIR / f'predicciones_{split}.csv'))
    print(f"   {split}: {n:,} registros")

print(f"\n🎯 Formato de salida:")
print(f"   Columnas: fecha_hora, id_zona, demanda_score")
print(f"   Score range: 0.0 (baja demanda) - 1.0 (alta demanda)")
print(f"   Normalización: {NORMALIZATION_METHOD}")

print("\n" + "="*80)