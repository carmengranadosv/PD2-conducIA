import pandas as pd
import numpy as np
import os
import json
import joblib
import gc
from pathlib import Path
import pyarrow.parquet as pq

# Scikit-Learn y XGBoost
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor

# ============================================================
# 1. CONFIGURACIÓN DE RUTAS 
# ============================================================
# Siguiendo la estructura de tu ejemplo para coherencia del proyecto
BASE_DIR = Path(__file__).resolve().parents[2] 

DATA_DIR = BASE_DIR / 'data' / 'processed' / 'tlc_clean' / 'problema5'
SAVE_DIR = BASE_DIR / 'despliegue' / 'modelos_finales'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Rutas de archivos de datos
TRAIN_FILE = DATA_DIR / 'train.parquet'
VAL_FILE = DATA_DIR / 'val.parquet'
TEST_FILE = DATA_DIR / 'test.parquet'

def optimizar_tipos(df):
    """Reduce el uso de memoria convirtiendo tipos de datos a 32 bits."""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

def leer_datos_despliegue(ruta, sample_size=None):
    """Lee el archivo optimizando memoria. Si sample_size es None, lee todo."""
    print(f"Leyendo {ruta.name}...")
    parquet_file = pq.ParquetFile(ruta)
    total_rows = parquet_file.metadata.num_rows
    
    if total_rows > sample_size:
        print(f"  Archivo muy grande ({total_rows:,} filas). Muestreando {sample_size:,}...")
        df = pd.read_parquet(ruta).sample(n=sample_size, random_state=42)
    else:
        df = pd.read_parquet(ruta)
    return optimizar_tipos(df)

def reentrenar_p5_despliegue():
    print("DESPLIEGUE FINAL - PROBLEMA 5 (XGBOOST)")

    # 2. CARGAR Y UNIFICAR TODO EL DATASET (100%)
    print("\nCargando todos los bloques de datos...")
    df_train = leer_datos_despliegue(TRAIN_FILE, sample_size=28_000_000)
    df_val   = leer_datos_despliegue(VAL_FILE, sample_size=6_000_000)
    df_test  = leer_datos_despliegue(TEST_FILE, sample_size=6_000_000)

    print("\nUnificando sets de datos para el entrenamiento global...")
    df_full = pd.concat([df_train, df_val, df_test], ignore_index=True)
    
    # Liberar memoria de los bloques individuales
    del df_train, df_val, df_test
    gc.collect()
    
    print(f"Dataset unificado con éxito: {len(df_full):,} filas.")

    # 3. PREPARACIÓN DE X e y 
    columnas_a_ignorar = ['propina', 'origen_id', 'destino_id', 'destino_zona', 'destino_barrio']
    
    X = df_full.drop(columns=[col for col in columnas_a_ignorar if col in df_full.columns])
    y = df_full['propina']
    
    del df_full
    gc.collect()

    # Definir columnas por tipo para el Pipeline
    columnas_categoricas = ['tipo_vehiculo', 'origen_zona', 'origen_barrio', 'evento_tipo', 'franja_horaria']
    columnas_numericas = [col for col in X.columns if col not in columnas_categoricas]

    # 4. CONFIGURACIÓN DEL PIPELINE FINAL
    preprocesador = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), columnas_numericas),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), columnas_categoricas)
        ],
        remainder='passthrough'
    )

    pipeline_final = Pipeline(steps=[
        ('preprocesamiento', preprocesador),
        ('modelo', XGBRegressor(
            n_estimators=250,      
            learning_rate=0.1,     
            max_depth=8,           
            subsample=0.8,         
            colsample_bytree=0.8,  
            random_state=42,
            n_jobs=-1,
            tree_method='hist'    
        ))
    ])

    # 5. ENTRENAMIENTO Y GUARDADO
    print(f"\nEntrenando XGBoost sobre el 100% de los datos...")
    pipeline_final.fit(X, y)
    
    model_path = SAVE_DIR / 'modelo_p5_xgboost.joblib'
    joblib.dump(pipeline_final, model_path)
    
    print(f"MODELO DE DESPLIEGUE GUARDADO EN: {model_path}")
    print(f"Total de registros procesados: {len(X):,}")

if __name__ == "__main__":
    reentrenar_p5_despliegue()