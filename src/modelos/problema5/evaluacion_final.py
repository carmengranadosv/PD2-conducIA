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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parents[3] 
DATA_DIR = BASE_DIR / 'data' / 'processed' / 'tlc_clean' / 'problema5'
MODELS_DIR = BASE_DIR / 'models' / 'problema5'
REPORTS_DIR = BASE_DIR / 'reports' / 'problema5'

# Asegurar directorios
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / 'train.parquet'
VAL_FILE = DATA_DIR / 'val.parquet'
TEST_FILE = DATA_DIR / 'test.parquet'

def optimizar_tipos(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

def leer_datos_finales(ruta, sample_size):
    print(f"Leyendo {ruta.name}...")
    parquet_file = pq.ParquetFile(ruta)
    total_rows = parquet_file.metadata.num_rows
    
    if total_rows > sample_size:
        print(f"  Archivo muy grande ({total_rows:,} filas). Muestreando {sample_size:,}...")
        df = pd.read_parquet(ruta).sample(n=sample_size, random_state=42)
    else:
        df = pd.read_parquet(ruta)
    return optimizar_tipos(df)

def calcular_metricas(y_true, y_pred, nombre_set):
    # Convertimos a NumPy para evitar desalineación de índices por el muestreo
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    
    # MAPE: Solo sobre propinas significativas (> 0.5$)
    mask = y_true > 0.5
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0
    
    return {
        f"{nombre_set}_mae": float(mae),
        f"{nombre_set}_rmse": float(rmse),
        f"{nombre_set}_r2": float(r2),
        f"{nombre_set}_bias": float(bias),
        f"{nombre_set}_mape": float(mape)
    }

def evaluacion_final():
    print("INICIANDO EVALUACIÓN FINAL DEL PROYECTO (XGBOOST)...")

    # 1. LEER TODOS LOS DATOS (El primer bloque de cada uno)
    print("Leyendo todos los datos...")
    df_train = leer_datos_finales(TRAIN_FILE, 28_000_000)
    df_val = leer_datos_finales(VAL_FILE, 6_000_000)
    df_test = leer_datos_finales(TEST_FILE, 6_000_000)

    # 2. JUNTAR TRAIN Y VAL (Maximizando el conocimiento)
    print("Combinando Train y Validación para el reentrenamiento...")
    df_total_train = pd.concat([df_train, df_val], ignore_index=True)
    del df_train, df_val
    gc.collect()

    # 3. SEPARAR X e y (Siguiendo la lógica "Pre-viaje" del profesor)
    columnas_a_ignorar = [
        'propina', 'origen_id', 'destino_id', 
        'destino_zona', 'destino_barrio' 
    ]

    def prepare_xy(df):
        X = df.drop(columns=[col for col in columnas_a_ignorar if col in df.columns])
        y = df['propina']
        return X, y

    X_train_final, y_train_final = prepare_xy(df_total_train)
    X_test, y_test = prepare_xy(df_test)

    del df_total_train, df_test
    gc.collect()

    # 4. DEFINIR COLUMNAS
    columnas_categoricas = [
        'tipo_vehiculo', 'origen_zona', 'origen_barrio', 
        'evento_tipo', 'franja_horaria'
    ]
    columnas_numericas = [col for col in X_train_final.columns if col not in columnas_categoricas]

    # 5. CREAR EL PIPELINE DEL MEJOR MODELO
    preprocesador = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), columnas_numericas),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), columnas_categoricas)
        ],
        remainder='passthrough'
    )

    pipeline_xgb_final = Pipeline(steps=[
        ('preprocesamiento', preprocesador),
        ('modelo', XGBRegressor(
            n_estimators=250,      
            learning_rate=0.1,     
            max_depth=8,           
            subsample=0.8,         
            colsample_bytree=0.8,  
            random_state=42,
            n_jobs=-1              
        ))
    ])

    # 6. ENTRENAMIENTO DEFINITIVO
    print("Reentrenando XGBoost con todos los datos...")
    pipeline_xgb_final.fit(X_train_final, y_train_final)

    # 7. EL EXAMEN FINAL (TEST)
    print("\nEnfrentando el modelo al conjunto de Test...")
    y_pred = pipeline_xgb_final.predict(X_test)
    
    # 8. CÁLCULO DE MÉTRICAS OFICIALES
    metrics_finales_dict = calcular_metricas(y_test, y_pred, "test")
    mae = metrics_finales_dict["test_mae"]
    rmse = metrics_finales_dict["test_rmse"]
    r2 = metrics_finales_dict["test_r2"]
    bias = metrics_finales_dict["test_bias"]
    mape = metrics_finales_dict["test_mape"]
    
    print("RESULTADOS FINALES (SOBRE TEST)")
    print("="*50)
    print(f"MAE:  ${mae:.4f} (Error absoluto medio)")
    print(f"RMSE: ${rmse:.4f} (Raíz del error cuadrático)")
    print(f"R2:    {r2:.4f} (Varianza explicada)")
    print(f"BIAS: ${bias:.4f} ({'Al alza' if bias > 0 else 'A la baja'})")
    print(f"MAPE: {mape:.4f} (Error porcentual absoluto medio)")

    # 9. GUARDADO DE SEGURIDAD
    final_metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "bias": float(bias),
        "mape": float(mape)
    }
    
    with open(os.path.join(REPORTS_DIR, 'final_test_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)

    joblib.dump(pipeline_xgb_final, MODELS_DIR / 'model_final.joblib')
if __name__ == "__main__":
    evaluacion_final()