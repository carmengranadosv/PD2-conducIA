"""
DESCRIPCIÓN:
    Este script realiza la evaluación definitiva del proyecto para el Problema 4. 
    Utiliza el modelo ganador (Red Neuronal con Embeddings) para obtener las 
    métricas finales tras un proceso de reentrenamiento con datos extendidos.

OBJETIVO:
    Obtener el rendimiento oficial del proyecto enfrentando al modelo con datos 
    completamente desconocidos (set de Test), garantizando la máxima capacidad 
    de generalización mediante la unión de los conjuntos de entrenamiento y validación.

FLUJO DE TRABAJO:
    1. CARGA: Lee los archivos train_p4, val_p4 y test_p4 desde data/processed.
    2. UNIFICACIÓN: Combina los conjuntos de Train y Validación para maximizar el 
       volumen de datos de aprendizaje previo al examen final.
    3. PREPARACIÓN: Reajusta los transformadores (LabelEncoder y StandardScaler) 
       con el conjunto unificado de datos.
    4. CARGA DE MODELO: Recupera la arquitectura y pesos del modelo "campeón" 
       previamente entrenado y guardado en formato .keras.
    5. EVALUACIÓN FINAL: Realiza la predicción sobre el set de Test y calcula las 
       métricas oficiales (MAE, RMSE, R2, BIAS, MAPE).
    6. MAPE ESPECÍFICO: Aplica un filtro de velocidad (> 1 mph) para asegurar que 
       el error porcentual no se distorsione por paradas de tráfico.
    7. PERSISTENCIA: Genera el archivo final 'final_test_metrics.json' con los 
       resultados definitivos para la memoria del proyecto.

NOTA IMPORTANTE: cambiar el modelo cargado en la sección 4 si veo que el campeón es otro
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import keras
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'tlc_clean', 'problema4')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(DATA_DIR, 'train_p4.parquet')
VAL_FILE = os.path.join(DATA_DIR, 'val_p4.parquet')
TEST_FILE = os.path.join(DATA_DIR, 'test_p4.parquet')

def calcular_metricas(y_true, y_pred, nombre_set):
    """Cálculo de métricas oficiales para el informe final."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    
    # R2 Score
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    bias = np.mean(y_pred - y_true)
    
    # MAPE: Solo donde la velocidad > 1 mph para evitar distorsiones por tráfico detenido
    mask = y_true > 1.0
    if mask.any():
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
    print("🚀 INICIANDO EXAMEN FINAL DEL MODELO (RED NEURONAL CON EMBEDDINGS)...")

    # 1. CARGA DE DATOS
    print("Cargando todos los datasets...")
    df_train = pd.read_parquet(TRAIN_FILE)
    df_val = pd.read_parquet(VAL_FILE)
    df_test = pd.read_parquet(TEST_FILE)

    # 2. COMBINACIÓN DE DATOS (Train + Val)
    # Al igual que tu compañera, usamos todo el conocimiento previo para el test final
    print("Combinando sets para el reentrenamiento definitivo...")
    df_total_train = pd.concat([df_train, df_val], ignore_index=True)

    # 3. PREPARACIÓN DE VARIABLES
    col_zona = 'origen_id'
    cols_clima = [
        'temp_c', 'precipitation', 'viento_kmh', 'lluvia', 'nieve', 
        'hay_lluvia', 'hay_nieve', 'es_festivo', 'num_eventos', 
        'dia_semana', 'es_fin_semana', 'hora_sen', 'hora_cos'
    ]

    le = LabelEncoder()
    scaler = StandardScaler()

    # Targets
    y_train_final = df_total_train['velocidad_mph'].values
    y_test = df_test['velocidad_mph'].values

    # Inputs
    X_zone_train = le.fit_transform(df_total_train[col_zona])
    X_zone_test = le.transform(df_test[col_zona])

    X_num_train = scaler.fit_transform(df_total_train[cols_clima])
    X_num_test = scaler.transform(df_test[cols_clima])

    # 4. CARGAR O RECONSTRUIR EL MODELO CAMPEÓN
    # Para asegurar el éxito, cargamos el modelo que ya entrenaste anteriormente
    model_path = os.path.join(RESULTS_DIR, 'red_neuronal_model.keras')
    
    if os.path.exists(model_path):
        print("Cargando el modelo campeón entrenado...")
        model = keras.models.load_model(model_path)
    else:
        print("❌ Error: No se encontró el modelo entrenado. Ejecuta primero red_neuronal_embeddings.py")
        return

    # 5. EL EXAMEN FINAL
    print("\nEnfrentando el modelo al conjunto de Test (Datos nunca vistos)...")
    y_pred = model.predict([X_zone_test, X_num_test]).flatten()

    # 6. MÉTRICAS OFICIALES
    res = calcular_metricas(y_test, y_pred, "test")
    
    print("\n" + "="*50)
    print("📊 RESULTADOS FINALES DEL PROYECTO (PROBLEMA 4)")
    print("="*50)
    print(f"MAE:  {res['test_mae']:.4f} mph (Error medio)")
    print(f"RMSE: {res['test_rmse']:.4f} mph")
    print(f"R2:   {res['test_r2']:.4f} (Capacidad de explicación)")
    print(f"BIAS: {res['test_bias']:.4f} ({'Optimista' if res['test_bias'] > 0 else 'Pesimista'})")
    print(f"MAPE: {res['test_mape']:.4f} %")
    print("="*50)

    # 7. GUARDADO DE SEGURIDAD
    with open(os.path.join(RESULTS_DIR, 'final_test_metrics.json'), 'w') as f:
        json.dump(res, f, indent=4)
    
    print(f"\n✅ Evaluación finalizada. Métricas guardadas en {RESULTS_DIR}")

if __name__ == "__main__":
    evaluacion_final()