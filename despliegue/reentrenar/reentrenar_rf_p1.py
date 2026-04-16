import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# ============================================================
# 1. CONFIGURACIÓN DE RUTAS 
# ============================================================
# Esto busca la carpeta raíz 'PD2-conducIA' sin importar dónde estés
BASE_DIR = Path(__file__).resolve().parents[2] 

DATA_DIR = BASE_DIR / 'data/processed/tlc_clean/problema1/features'
SAVE_DIR = BASE_DIR / 'despliegue/modelos_finales'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def reentrenar_p1_completo():
    print("=" * 60)
    print("REENTRENAMIENTO FINAL - PROBLEMA 1 (RANDOM FOREST)")
    print("=" * 60)
    print(f"Buscando datos en: {DATA_DIR}")

    # Comprobación de seguridad
    if not (DATA_DIR / 'train.parquet').exists():
        print(f"ERROR: No se encuentran los archivos en {DATA_DIR}")
        print("Asegúrate de que la ruta es correcta desde la raíz del proyecto.")
        return

    # ============================================================
    # 2. CARGAR Y UNIFICAR TODO EL DATASET (100%)
    # ============================================================
    print("\nCargando todos los conjuntos de datos...")
    df_train = pd.read_parquet(DATA_DIR / 'train.parquet')
    df_val   = pd.read_parquet(DATA_DIR / 'val.parquet')
    df_test  = pd.read_parquet(DATA_DIR / 'test.parquet')

    df_full = pd.concat([df_train, df_val, df_test], ignore_index=True)
    print(f"Dataset unificado con éxito: {len(df_full):,} filas.")

    # Cargamos metadata
    with open(DATA_DIR / 'metadata.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    FEATURE_COLS = meta['feature_cols']
    TARGET = meta['target']

    missing_cols = [col for col in FEATURE_COLS + [TARGET] if col not in df_full.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas esperadas en los datos de P1: {missing_cols}")

    X = df_full[FEATURE_COLS]
    y = df_full[TARGET]

    # ============================================================
    # 3. CONFIGURACIÓN DEL MODELO
    # ============================================================
    rf_final = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    )

    # ============================================================
    # 4. ENTRENAMIENTO Y GUARDADO
    # ============================================================
    print(f"\nEntrenando Random Forest...")
    rf_final.fit(X, y)
    
    model_path = SAVE_DIR / 'modelo_p1_rf.joblib'
    joblib.dump(rf_final, model_path)
    
    print(f"\nMODELO FINAL GUARDADO EN: {model_path}")
    print("=" * 60)

if __name__ == "__main__":
    reentrenar_p1_completo()
