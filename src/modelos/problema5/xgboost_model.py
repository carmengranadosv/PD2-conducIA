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
    df_train = pq.ParquetFile(TRAIN_FILE).read_row_group(0).to_pandas()
    df_val = pq.ParquetFile(VAL_FILE).read_row_group(0).to_pandas()
    df_test = pq.ParquetFile(TEST_FILE).read_row_group(0).to_pandas()

    # 2. SEPARAR X e y
    columnas_a_ignorar = ['propina', 'origen_id', 'destino_id']

    def prepare_xy(df):
        X = df.drop(columns=[col for col in columnas_a_ignorar if col in df.columns])
        y = df['propina']
        return X, y

    X_train, y_train = prepare_xy(df_train)
    X_val, y_val = prepare_xy(df_val)
    X_test, y_test = prepare_xy(df_test)

    print(f"Datos: Train({len(X_train)}), Val({len(X_val)}), Test({len(X_test)})")

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
            n_jobs=-1,              # Usa todos los núcleos de tu procesador
            tree_method='hist'      # Método eficiente para grandes datasets
        ))
    ])

    # 6. ENTRENAMIENTO
    print("Ajustando el Pipeline y entrenando los árboles de decisión...")
    pipeline_xgb.fit(X_train, y_train)

    # 7. IMPORTANCIA DE VARIABLES
    df_importancia = extraer_importancia_xgb(pipeline_xgb, columnas_numericas, columnas_categoricas)
    df_importancia.to_csv(os.path.join(RESULTS_DIR, 'xgboost_features.csv'), index=False)
    print("\nTOP 5 Variables para XGBoost:")
    print(df_importancia.head(5).to_string(index=False))

    # 8. PREDICCIÓN SOBRE VALIDACIÓN
    print("\n Realizando predicciones sobre Validación...")
    metrics_val = calcular_metricas(y_val, pipeline_xgb.predict(X_val), "val")
    metrics_test = calcular_metricas(y_test, pipeline_xgb.predict(X_test), "test")

    resultados = {
        "model_name": "XGBoost_Regressor",
        **metrics_val,
        **metrics_test
    }

    # 8. GUARDAR RESULTADOS
    res_path = os.path.join(RESULTS_DIR, 'xgboost_results.json')
    with open(res_path, 'w') as f:
        json.dump(resultados, f, indent=4)

    joblib.dump(pipeline_xgb, os.path.join(RESULTS_DIR, 'xgboost_model.joblib'))
    
    print("\n" + "="*45)
    print("REPORTE XGBOOST")
    print(f"  MAE:  ${resultados['test_mae']:.4f}")
    print(f"  RMSE: ${resultados['test_rmse']:.4f}")
    print(f"  R2:   {resultados['test_r2']:.4f}")
    print(f"  BIAS: ${resultados['test_bias']:.4f}")
    print(f"  Resultados guardados en: {res_path}")

if __name__ == "__main__":
    entrenar_xgboost()