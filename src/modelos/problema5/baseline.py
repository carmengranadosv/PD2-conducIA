import pandas as pd
import numpy as np
import os
import json
import joblib
from pathlib import Path
import pyarrow.parquet as pq

# Scikit-Learn Pipeline y Preprocesamiento
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parents[3] 
DATA_DIR = BASE_DIR / 'data' / 'processed' / 'tlc_clean' / 'problema5'
MODELS_DIR = BASE_DIR / 'models' / 'problema5'
REPORTS_DIR = BASE_DIR / 'reports' / 'problema5'

# Crear directorios si no existen
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / 'train.parquet'
VAL_FILE = DATA_DIR / 'val.parquet'
TEST_FILE = DATA_DIR / 'test.parquet'

def optimizar_tipos(df):
    """Reduce el uso de memoria convirtiendo tipos de datos."""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

def leer_datos_optimizados(ruta, sample_size=2000000):
    """Lee el archivo por fragmentos o aplica un muestreo para no saturar la RAM."""
    print(f"Leyendo {ruta.name}...")
    # Leemos metadatos para saber el total
    parquet_file = pq.ParquetFile(ruta)
    total_rows = parquet_file.metadata.num_rows
    
    # Si es excesivo, tomamos una muestra aleatoria del 10% o un tope fijo
    # Esto garantiza que el modelo vea variedad sin explotar la RAM
    if total_rows > sample_size:
        print(f"  Archivo muy grande ({total_rows:,} filas). Muestreando {sample_size:,}...")
        df = pd.read_parquet(ruta).sample(n=sample_size, random_state=42)
    else:
        df = pd.read_parquet(ruta)
    
    return optimizar_tipos(df)

# Función para calcular la importancia de las variables
def extraer_importancia_lr(pipeline, col_num, col_cat):
    """Extrae los coeficientes del modelo."""
    modelo = pipeline.named_steps['modelo']
    prepro = pipeline.named_steps['preprocesamiento']
    nombres_cat = prepro.named_transformers_['cat'].get_feature_names_out(col_cat)
    nombres_total = np.concatenate([col_num, nombres_cat])
    
    importancia = pd.DataFrame({
        'feature': nombres_total,
        'coeficiente': modelo.coef_,
        'abs_impacto': np.abs(modelo.coef_)
    }).sort_values(by='abs_impacto', ascending=False)
    return importancia

# Función para calcular las métricas del modelo
def calcular_metricas(y_true, y_pred, nombre_set):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    return {
        f"{nombre_set}_mae": float(mae),
        f"{nombre_set}_rmse": float(rmse),
        f"{nombre_set}_r2": float(r2),
        f"{nombre_set}_bias": float(bias),
        f"{nombre_set}_mape": float(mape)
    }

def entrenar_baseline():
    print("Iniciando entrenamiento del Baseline (Regresión Lineal)...")

    # CARGA DE DATOS (Muestreo para Baseline)
    df_train = leer_datos_optimizados(TRAIN_FILE, sample_size=28000000)
    df_val = leer_datos_optimizados(VAL_FILE, sample_size=6000000)
    df_test = leer_datos_optimizados(TEST_FILE, sample_size=6000000)

    # SEPARAR X e y (Variable Objetivo: propina)
    columnas_a_ignorar = ['propina', 'origen_id', 'destino_id', 'destino_zona', 'destino_barrio']

    # Para no repetir el código de preparar los df
    def prepare_xy(df):
        X = df.drop(columns=[col for col in columnas_a_ignorar if col in df.columns])
        y = df['propina']
        return X, y

    X_train, y_train = prepare_xy(df_train)
    X_val, y_val = prepare_xy(df_val)
    X_test, y_test = prepare_xy(df_test)

    print(f"Datos: Train({len(X_train)}), Val({len(X_val)}), Test({len(X_test)})")


    # DEFINIR COLUMNAS POR TIPO
    columnas_categoricas = [
        'tipo_vehiculo',   # "Yellow Taxi" o "VTC"
        'origen_zona',     # Texto de la zona
        'origen_barrio',   # Texto del barrio (Manhattan, Queens...)
        # 'destino_zona',    # Texto de la zona
        # 'destino_barrio',  # Texto del barrio
        'evento_tipo',     # "No hay", "Concierto", etc.
        'franja_horaria'   # "Madrugada", "Noche", etc.
    ]

    # El resto son numéricas (incluyendo las binarias como es_fin_semana, nieve, trafico_denso)
    columnas_numericas = [col for col in X_train.columns if col not in columnas_categoricas]

    # CREAR EL PREPROCESADOR (ColumnTransformer)
    preprocesador = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), columnas_numericas),
            ('cat', OneHotEncoder(handle_unknown='ignore'), columnas_categoricas)
        ],
        remainder='passthrough'
    )

    # CREAR EL PIPELINE FINAL
    pipeline_lr = Pipeline(steps=[
        ('preprocesamiento', preprocesador),
        ('modelo', Ridge(alpha=1.0))
    ])

    # ENTRENAMIENTO
    print("Ajustando el Pipeline y entrenando el modelo...")
    pipeline_lr.fit(X_train, y_train)

    # IMPORTANCIA
    df_importancia = extraer_importancia_lr(pipeline_lr, columnas_numericas, columnas_categoricas)
    df_importancia.to_csv(os.path.join(REPORTS_DIR, 'baseline_features.csv'), index=False)

    # EVALUACIÓN SOBRE VALIDACIÓN Y TEST
    print("Calculando las métricas en validación y test...")
    metrics_val = calcular_metricas(y_val, pipeline_lr.predict(X_val), "val")
    metrics_test = calcular_metricas(y_test, pipeline_lr.predict(X_test), "test")

    # Unificar resultados
    resultados = {
        "model_name": "Baseline_Ridge",
        "data_info": {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test)
        },
        **metrics_val,
        **metrics_test
    }

    # GUARDAR RESULTADOS EN JSON
    res_path = os.path.join(REPORTS_DIR, 'baseline_results.json')
    with open(res_path, 'w') as f:
        json.dump(resultados, f, indent=4)
    
    # GUARDAR MODELO ENTRENADO
    joblib.dump(pipeline_lr, os.path.join(MODELS_DIR, 'baseline_model.joblib'))

    print("\nEvaluación en TEST finalizada:")
    print(f"  MAE:  ${resultados['test_mae']:.4f}")
    print(f"  RMSE: ${resultados['test_rmse']:.4f}")
    print(f"  R2:   {resultados['test_r2']:.4f}")
    print(f"  BIAS: ${resultados['test_bias']:.4f}")
    print(f"  Resultados guardados en: {res_path}")

if __name__ == "__main__":
    entrenar_baseline()

