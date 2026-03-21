import pandas as pd
import numpy as np
import os
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

TRAIN_FILE = os.path.join(DATA_DIR, 'train.parquet')
VAL_FILE = os.path.join(DATA_DIR, 'val.parquet')
TEST_FILE = os.path.join(DATA_DIR, 'test.parquet') 

def evaluacion_final():
    print("INICIANDO EVALUACIÓN FINAL DEL PROYECTO (XGBOOST)...")

    # 1. LEER TODOS LOS DATOS (El primer bloque de cada uno)
    print("Leyendo todos los datos...")
    df_train = pq.ParquetFile(TRAIN_FILE).read_row_group(0).to_pandas()
    df_val = pq.ParquetFile(VAL_FILE).read_row_group(0).to_pandas()
    df_test = pq.ParquetFile(TEST_FILE).read_row_group(0).to_pandas() # El examen final

    # 2. JUNTAR TRAIN Y VAL (Maximizando el conocimiento)
    print("Combinando Train y Validación para el reentrenamiento...")
    df_entrenamiento_total = pd.concat([df_train, df_val], ignore_index=True)

    # 3. SEPARAR X e y
    columnas_a_ignorar = ['propina', 'origen_id', 'destino_id']

    X_train_total = df_entrenamiento_total.drop(columns=[col for col in columnas_a_ignorar if col in df_entrenamiento_total.columns])
    y_train_total = df_entrenamiento_total['propina']
    
    X_test = df_test.drop(columns=[col for col in columnas_a_ignorar if col in df_test.columns])
    y_test = df_test['propina']

    print(f"Entrenando al Campeón con {len(X_train_total):,} registros. Evaluando sobre {len(X_test):,} registros.")

    # 4. DEFINIR COLUMNAS
    columnas_categoricas = [
        'tipo_vehiculo', 'origen_zona', 'origen_barrio', 
        'destino_zona', 'destino_barrio', 'evento_tipo', 'franja_horaria'
    ]
    columnas_numericas = [col for col in X_train_total.columns if col not in columnas_categoricas]

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
    pipeline_xgb_final.fit(X_train_total, y_train_total)

    # 7. EL EXAMEN FINAL (TEST)
    print("\nEnfrentando el modelo al conjunto de Test...")
    y_pred_test = pipeline_xgb_final.predict(X_test)
    
    # 8. CÁLCULO DE MÉTRICAS OFICIALES
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)
    mean_error = np.mean(y_pred_test - y_test) 
    
    print("\n" + "="*50)
    print("RESULTADOS FINALES (SOBRE TEST)")
    print("="*50)
    print(f"MAE:  ${mae:.4f} (Error absoluto medio)")
    print(f"RMSE: ${rmse:.4f} (Raíz del error cuadrático)")
    print(f"R2:    {r2:.4f} (Varianza explicada)")
    print(f"BIAS: ${mean_error:.4f} ({'Al alza' if mean_error > 0 else 'A la baja'})")
    print("="*50)

if __name__ == "__main__":
    evaluacion_final()