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

def entrenar_xgboost():
    print("Iniciando entrenamiento del modelo (XGBoost)...")

    # 1. LEER DATOS (Modo eficiente)
    print("Leyendo datos...")
    df_train = pq.ParquetFile(TRAIN_FILE).read_row_group(0).to_pandas()
    df_val = pq.ParquetFile(VAL_FILE).read_row_group(0).to_pandas()

    # 2. SEPARAR X e y
    columnas_a_ignorar = ['propina', 'origen_id', 'destino_id']

    X_train = df_train.drop(columns=[col for col in columnas_a_ignorar if col in df_train.columns])
    y_train = df_train['propina']
    
    X_val = df_val.drop(columns=[col for col in columnas_a_ignorar if col in df_val.columns])
    y_val = df_val['propina']

    print(f"Entrenando con {len(X_train):,} registros. Validando con {len(X_val):,}.")

    # 3. DEFINIR COLUMNAS POR TIPO
    columnas_categoricas = [
        'tipo_vehiculo', 'origen_zona', 'origen_barrio', 
        'destino_zona', 'destino_barrio', 'evento_tipo', 'franja_horaria'
    ]
    columnas_numericas = [col for col in X_train.columns if col not in columnas_categoricas]

    # 4. CREAR EL PREPROCESADOR
    preprocesador = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), columnas_numericas),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), columnas_categoricas)
        ],
        remainder='passthrough'
    )

    # 5. CREAR EL PIPELINE CON XGBOOST
    # Configuramos hiperparámetros robustos para evitar que memorice a los "millonarios"
    pipeline_xgb = Pipeline(steps=[
        ('preprocesamiento', preprocesador),
        ('modelo', XGBRegressor(
            n_estimators=200,      # Número de árboles
            learning_rate=0.1,     # Ritmo de aprendizaje (suave)
            max_depth=7,           # Profundidad de cada árbol (para capturar interacciones)
            subsample=0.8,         # Usa el 80% de datos por árbol (previene sobreajuste)
            colsample_bytree=0.8,  # Usa el 80% de columnas por árbol
            random_state=42,
            n_jobs=-1              # Usa todos los núcleos de tu procesador
        ))
    ])

    # 6. ENTRENAMIENTO
    print("Ajustando el Pipeline y entrenando los árboles de decisión...")
    pipeline_xgb.fit(X_train, y_train)

    # 7. PREDICCIÓN SOBRE VALIDACIÓN
    print("\n Realizando predicciones sobre Validación...")
    y_pred = pipeline_xgb.predict(X_val)
    
    # 8. CÁLCULO DE MÉTRICAS
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    mean_error = np.mean(y_pred - y_val) 
    epsilon = 1e-8
    mape = np.mean(np.abs((y_val - y_pred) / (y_val + epsilon))) * 100
    
    print("\n" + "="*45)
    print("REPORTE XGBOOST")
    print("="*45)
    print(f"MAE:  ${mae:.4f} (Error absoluto medio)")
    print(f"RMSE: ${rmse:.4f} (Raíz del error cuadrático)")
    print(f"R2:    {r2:.4f} (Varianza explicada)")
    print(f"BIAS: ${mean_error:.4f} ({'Al alza' if mean_error > 0 else 'A la baja'})")
    print(f"MAPE:  {mape:.2f}% (Error porcentual)")
    print("="*45)

if __name__ == "__main__":
    entrenar_xgboost()