import pandas as pd
import numpy as np
import os
import json
import joblib
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
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'tlc_clean', 'problema5')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(DATA_DIR, 'train.parquet')
VAL_FILE = os.path.join(DATA_DIR, 'val.parquet')
TEST_FILE = os.path.join(DATA_DIR, 'test.parquet') 

def calcular_metricas(y_true, y_pred, nombre_set):
    # Convertimos a series de pandas para facilitar el filtrado si vienen como arrays
    y_true_s = pd.Series(y_true)
    y_pred_s = pd.Series(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    
    # Solo lo calculamos donde la propina es mayor a 0.50$
    mask = y_true_s > 0.5
    if mask.any():
        mape = np.mean(np.abs((y_true_s[mask] - y_pred_s[mask]) / y_true_s[mask])) * 100
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
    df_train = pq.ParquetFile(TRAIN_FILE).read_row_group(0).to_pandas()
    df_val = pq.ParquetFile(VAL_FILE).read_row_group(0).to_pandas()
    df_test = pq.ParquetFile(TEST_FILE).read_row_group(0).to_pandas() # El examen final

    # 2. JUNTAR TRAIN Y VAL (Maximizando el conocimiento)
    print("Combinando Train y Validación para el reentrenamiento...")
    df_total_train = pd.concat([df_train, df_val], ignore_index=True)

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

    print(f"Entrenando al Campeón con {len(X_train_final):,} registros. Evaluando sobre {len(X_test):,} registros.")

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
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), columnas_categoricas)
        ],
        remainder='passthrough'
    )

    pipeline_xgb_final = Pipeline(steps=[
        ('preprocesamiento', preprocesador),
        ('modelo', XGBRegressor(
            n_estimators=200,      
            learning_rate=0.1,     
            max_depth=7,           
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
    
    with open(os.path.join(RESULTS_DIR, 'final_test_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)
if __name__ == "__main__":
    evaluacion_final()