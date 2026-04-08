"""
DESCRIPCIÓN:
    Este script implementa el modelo BASELINE para el Problema 4 (Eficiencia y Clima).
    Utiliza una Regresión Ridge (Regularización L2) cargando los datos ya procesados.

OBJETIVO:
    Establecer el benchmark (punto de referencia) de error para comparar con los
    modelos complejos (ST-GCN y Red Neuronal).

FLUJO DE TRABAJO:
    1. CARGA: Lee los archivos train_p4, val_p4 y test_p4 desde data/processed.
    2. DIVISIÓN: Separa cada set en X (entradas) e y (velocidad_mph).
    3. ENCODING: Aplica One-Hot Encoding a variables categóricas.
    4. ENTRENAMIENTO: Ajusta la Regresión Ridge usando Train.
    5. VALIDACIÓN: Comprueba el rendimiento inicial con el set de Val.
    6. EVALUACIÓN FINAL: Reporta métricas definitivas (MAE, RMSE, R2, BIAS, MAPE) 
       con el set de Test.
    7. PERSISTENCIA: Guarda el modelo entrenado (.joblib), las métricas (.json) 
       y la importancia de variables (.csv) en la carpeta 'results'.
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'tlc_clean', 'problema4')
# Creamos la carpeta de resultados para guardar métricas, etc
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- FUNCIONES DE APOYO ---
def calcular_metricas(y_true, y_pred, nombre_set):
    """Calcula el pack completo de métricas igual que tu compañera."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    
    # MAPE (con epsilon para evitar división por cero si la velocidad es 0)
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    return {
        f"{nombre_set}_mae": float(mae),
        f"{nombre_set}_rmse": float(rmse),
        f"{nombre_set}_r2": float(r2),
        f"{nombre_set}_bias": float(bias),
        f"{nombre_set}_mape": float(mape)
    }

def cargar_y_preparar_datos():
    print("Cargando datasets...")
    train = pd.read_parquet(os.path.join(DATA_DIR, 'train_p4.parquet'))
    val = pd.read_parquet(os.path.join(DATA_DIR, 'val_p4.parquet'))
    test = pd.read_parquet(os.path.join(DATA_DIR, 'test_p4.parquet'))

    cols_categoricas = ['tipo_vehiculo', 'franja_horaria']
    cols_numericas = [
        'temp_c', 'precipitation', 'viento_kmh', 'lluvia', 'nieve', 
        'hay_lluvia', 'hay_nieve', 'es_festivo', 'num_eventos', 
        'dia_semana', 'es_fin_semana', 'hora_sen', 'hora_cos'
    ]
    
    def transform_x_y(df):
        # get_dummies genera el OneHotEncoding básico
        X = pd.get_dummies(df[cols_categoricas + cols_numericas], 
                          columns=cols_categoricas, drop_first=True)
        y = df['velocidad_mph']
        return X, y

    X_train, y_train = transform_x_y(train)
    X_val, y_val = transform_x_y(val)
    X_test, y_test = transform_x_y(test)

    # Alinear columnas
    X_train, X_val = X_train.align(X_val, join='left', axis=1, fill_value=0)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    return X_train, y_train, X_val, y_val, X_test, y_test

def ejecutar_baseline():
    print("Iniciando entrenamiento del Baseline (Ridge)...")
    X_train, y_train, X_val, y_val, X_test, y_test = cargar_y_preparar_datos()
    
    # 1. Entrenamiento
    modelo = Ridge(alpha=1.0)
    modelo.fit(X_train, y_train)
    
    # 2. Evaluación (Validación y Test)
    metrics_val = calcular_metricas(y_val, modelo.predict(X_val), "val")
    metrics_test = calcular_metricas(y_test, modelo.predict(X_test), "test")
    
    # 3. Unificar resultados en un JSON
    resultados = {
        "model_name": "Baseline_RidgeRegression",
        **metrics_val,
        **metrics_test
    }
    
    # 4. Guardar Importancia de Variables en CSV
    coefs = pd.Series(modelo.coef_, index=X_train.columns).sort_values()
    df_importancia = pd.DataFrame({
        'feature': coefs.index,
        'coeficiente': coefs.values,
        'abs_impacto': np.abs(coefs.values)
    }).sort_values(by='abs_impacto', ascending=False)
    
    df_importancia.to_csv(os.path.join(RESULTS_DIR, 'baseline_features.csv'), index=False)

    # 5. Guardar Métricas en JSON
    res_path = os.path.join(RESULTS_DIR, 'baseline_results.json')
    with open(res_path, 'w') as f:
        json.dump(resultados, f, indent=4)
    
    # 6. Guardar Modelo en Joblib
    joblib.dump(modelo, os.path.join(RESULTS_DIR, 'baseline_model.joblib'))
    
    print(f"\n✅ Proceso finalizado. Resultados guardados en {RESULTS_DIR}")
    print(f"MAE en Test: {resultados['test_mae']:.2f} mph")

if __name__ == "__main__":
    ejecutar_baseline()