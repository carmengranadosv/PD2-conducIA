"""
PROYECTO: ConducIA - PD2 (GIDIA)
MÓDULO: modelo_baseline.py
========================================================================================
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
    6. EVALUACIÓN FINAL: Reporta métricas definitivas con el set de Test.
========================================================================================
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'tlc_clean', 'problema4')

def cargar_y_preparar_datos():
    """
    Carga los parquets y separa en X (features) e y (target).
    """
    print("Cargando datasets desde la carpeta procesada...")
    
    train = pd.read_parquet(os.path.join(DATA_DIR, 'train_p4.parquet'))
    val = pd.read_parquet(os.path.join(DATA_DIR, 'val_p4.parquet'))
    test = pd.read_parquet(os.path.join(DATA_DIR, 'test_p4.parquet'))

    # Definimos columnas que usará el modelo
    # Nota: No usamos IDs de zona en el Baseline simple para evitar miles de columnas,
    # pero sí el resto de variables climáticas y temporales.
    cols_categoricas = ['tipo_vehiculo', 'franja_horaria']
    cols_numericas = [
        'temp_c', 'precipitation', 'viento_kmh', 'lluvia', 'nieve', 
        'hay_lluvia', 'hay_nieve', 'es_festivo', 'num_eventos', 
        'dia_semana', 'es_fin_semana', 'hora_sen', 'hora_cos'
    ]
    
    def transform_x_y(df):
        # One-Hot Encoding
        X = pd.get_dummies(df[cols_categoricas + cols_numericas], 
                          columns=cols_categoricas, drop_first=True)
        y = df['velocidad_mph']
        return X, y

    X_train, y_train = transform_x_y(train)
    X_val, y_val = transform_x_y(val)
    X_test, y_test = transform_x_y(test)

    # Alinear columnas para asegurar que todos los sets tengan las mismas categorías
    # (Por si alguna franja horaria no apareciera en algún set)
    X_train, X_val = X_train.align(X_val, join='left', axis=1, fill_value=0)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    return X_train, y_train, X_val, y_val, X_test, y_test

def ejecutar_baseline():
    # 1. Obtener los 6 objetos (X, y para cada set)
    X_train, y_train, X_val, y_val, X_test, y_test = cargar_y_preparar_datos()
    
    # 2. Entrenamiento
    print(f"\nEntrenando Regresión Ridge con {len(X_train):,} filas...")
    modelo = Ridge(alpha=1.0)
    modelo.fit(X_train, y_train)
    
    # 3. Evaluación en Validación (para control interno)
    y_val_pred = modelo.predict(X_val)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    print(f"Rendimiento en Validación (Mes 9): MAE = {mae_val:.2f} mph")
    
    # 4. Evaluación en Test (Examen final)
    print("\nEvaluando en el set de Test (Meses 10 y 11)...")
    y_pred = modelo.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # --- RESULTADOS ---
    print("\n" + "="*40)
    print("      RESULTADOS FINALES BASELINE")
    print("="*40)
    print(f"MAE  (Error Medio Absoluto): {mae:.2f} mph")
    print(f"RMSE (Error Cuadrático):     {rmse:.2f} mph")
    print(f"R2   (Varianza explicada):   {r2:.4f}")
    print("="*40)
    
    # 5. Factores determinantes
    coefs = pd.Series(modelo.coef_, index=X_train.columns).sort_values()
    print("\nFactores que más reducen la velocidad (según el modelo):")
    print(coefs.head(3))
    
    return modelo

if __name__ == "__main__":
    try:
        ejecutar_baseline()
    except Exception as e:
        print(f"Error durante el Baseline: {e}")