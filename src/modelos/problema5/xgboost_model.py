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

# Crear directorios
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

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

def leer_datos_optimizados(ruta, sample_size):
    """Lee el archivo con muestreo aleatorio para no colapsar la RAM."""
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
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    
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

def extraer_importancia_xgb(pipeline, col_num, col_cat):
    """Extrae la importancia de las variables para XGBoost."""
    modelo = pipeline.named_steps['modelo']
    prepro = pipeline.named_steps['preprocesamiento']
    
    nombres_cat = prepro.named_transformers_['cat'].get_feature_names_out(col_cat)
    nombres_total = np.concatenate([col_num, nombres_cat])
    
    importancia = pd.DataFrame({
        'feature': nombres_total,
        'importance': modelo.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    return importancia

def entrenar_xgboost():
    print("Iniciando entrenamiento del modelo (XGBoost)...")

    # 1. LEER DATOS (Modo eficiente)
    print("Leyendo datos...")
    df_train = leer_datos_optimizados(TRAIN_FILE, sample_size=28000000)
    df_val = leer_datos_optimizados(VAL_FILE, sample_size=6000000)
    df_test = leer_datos_optimizados(TEST_FILE, sample_size=6000000)

    # 2. SEPARAR X e y
    columnas_a_ignorar = ['propina', 'origen_id', 'destino_id', 'destino_zona', 'destino_barrio']

    def prepare_xy(df):
        X = df.drop(columns=[col for col in columnas_a_ignorar if col in df.columns])
        y = df['propina']
        return X, y

    X_train, y_train = prepare_xy(df_train)
    X_val, y_val = prepare_xy(df_val)
    X_test, y_test = prepare_xy(df_test)

    del df_train, df_val, df_test
    gc.collect()

    # 3. DEFINIR COLUMNAS POR TIPO
    columnas_categoricas = [
        'tipo_vehiculo',   # "Yellow Taxi" o "VTC"
        'origen_zona',     # Texto de la zona
        'origen_barrio',   # Texto del barrio (Manhattan, Queens...)
        # 'destino_zona',    # Texto de la zona
        # 'destino_barrio',  # Texto del barrio
        'evento_tipo',     # "No hay", "Concierto", etc.
        'franja_horaria'   # "Madrugada", "Noche", etc.
    ]
    columnas_numericas = [col for col in X_train.columns if col not in columnas_categoricas]

    # 4. CREAR EL PREPROCESADOR
    preprocesador = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), columnas_numericas),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), columnas_categoricas)
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
            n_jobs=-1,              # Usa todos los núcleos de tu procesador
            tree_method='hist'      # Método eficiente para grandes datasets
        ))
    ])

    # 6. ENTRENAMIENTO
    print("Ajustando el Pipeline y entrenando los árboles de decisión...")
    pipeline_xgb.fit(X_train, y_train)

    # 7. IMPORTANCIA DE VARIABLES
    df_importancia = extraer_importancia_xgb(pipeline_xgb, columnas_numericas, columnas_categoricas)
    df_importancia.to_csv(REPORTS_DIR / 'xgboost_features.csv', index=False)
    print("\nTOP 5 Variables para XGBoost:")
    print(df_importancia.head(5).to_string(index=False))

    # 8. PREDICCIÓN SOBRE VALIDACIÓN
    print("\n Realizando predicciones sobre Validación...")
    metrics_val = calcular_metricas(y_val, pipeline_xgb.predict(X_val), "val")
    metrics_test = calcular_metricas(y_test, pipeline_xgb.predict(X_test), "test")

    resultados = {
        "model_name": "XGBoost",
        "data_info": {
            "train_samples": len(y_train),
            "val_samples": len(y_val),
            "test_samples": len(y_test)
        },
        **metrics_val,
        **metrics_test
    }

    # 8. GUARDAR RESULTADOS
    res_path = os.path.join(REPORTS_DIR, 'xgboost_results.json')
    with open(res_path, 'w') as f:
        json.dump(resultados, f, indent=4)

    joblib.dump(pipeline_xgb, os.path.join(MODELS_DIR, 'xgboost_model.joblib'))

    print("\n" + "="*45)
    print("REPORTE XGBOOST")
    print(f"  MAE:  ${resultados['test_mae']:.4f}")
    print(f"  RMSE: ${resultados['test_rmse']:.4f}")
    print(f"  R2:   {resultados['test_r2']:.4f}")
    print(f"  BIAS: ${resultados['test_bias']:.4f}")
    print(f"  Resultados guardados en: {res_path}")

if __name__ == "__main__":
    entrenar_xgboost()