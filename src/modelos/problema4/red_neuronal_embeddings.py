"""
PROYECTO: ConducIA - PD2 (GIDIA)
MÓDULO: red_neuronal_embeddings.py
========================================================================================
DESCRIPCIÓN:
    Implementación de una Red Neuronal Profunda (MLP) con capas de Embedding 
    para las Zonas de NY y capas Densas para las variables climáticas.
========================================================================================
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import keras
from keras import layers, models, ops
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'tlc_clean', 'problema4')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def calcular_metricas(y_true, y_pred, nombre_set):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    
    # R2 Score manual para evitar dependencias extra si fallan
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    bias = np.mean(y_pred - y_true)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return {
        f"{nombre_set}_mae": float(mae),
        f"{nombre_set}_rmse": float(rmse),
        f"{nombre_set}_r2": float(r2),
        f"{nombre_set}_bias": float(bias),
        f"{nombre_set}_mape": float(mape)
    }

def entrenar_red_embeddings():
    print("Iniciando entrenamiento: Red Neuronal con Embeddings...")

    # 1. CARGAR DATOS
    train = pd.read_parquet(os.path.join(DATA_DIR, 'train_p4.parquet'))
    val = pd.read_parquet(os.path.join(DATA_DIR, 'val_p4.parquet'))
    test = pd.read_parquet(os.path.join(DATA_DIR, 'test_p4.parquet'))

    # 2. DEFINIR COLUMNAS
    # La zona la trataremos por separado para el Embedding
    col_zona = 'origen_id' 
    cols_clima_temp = [
        'temp_c', 'precipitation', 'viento_kmh', 'lluvia', 'nieve', 
        'hay_lluvia', 'hay_nieve', 'es_festivo', 'num_eventos', 
        'dia_semana', 'es_fin_semana', 'hora_sen', 'hora_cos'
    ]

    # 3. PREPARAR INPUTS (X) y TARGET (y)
    # Importante: Las IDs de zona deben empezar en 0 y ser continuas para el Embedding
    # Usamos el mapa de zonas que generamos antes o un LabelEncoder simple
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    
    le = LabelEncoder()
    scaler = StandardScaler()

    # Ajustamos con train y transformamos todos
    y_train = train['velocidad_mph'].values
    y_val = val['velocidad_mph'].values
    y_test = test['velocidad_mph'].values

    # Input 1: La Zona (ID numérico)
    X_zone_train = le.fit_transform(train[col_zona])
    X_zone_val = le.transform(val[col_zona])
    X_zone_test = le.transform(test[col_zona])

    # Input 2: Clima y Tiempo (Numérico escalado)
    X_num_train = scaler.fit_transform(train[cols_clima_temp])
    X_num_val = scaler.transform(val[cols_clima_temp])
    X_num_test = scaler.transform(test[cols_clima_temp])

    num_unique_zones = len(le.classes_)

    # 4. CONSTRUIR ARQUITECTURA (Keras Functional API)
    # Rama de la Zona (Embedding)
    input_zone = layers.Input(shape=(1,), name="input_zona")
    embed_zone = layers.Embedding(input_dim=num_unique_zones, output_dim=10)(input_zone)
    embed_zone = layers.Flatten()(embed_zone)

    # Rama del Clima (Densa)
    input_num = layers.Input(shape=(len(cols_clima_temp),), name="input_clima")
    
    # Combinar ambas ramas
    combined = layers.Concatenate()([embed_zone, input_num])

    # Capas profundas (Siguiendo el estilo de tu compañera)
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    
    output = layers.Dense(1, name="prediccion_velocidad")(x)

    model = models.Model(inputs=[input_zone, input_num], outputs=output)
    
    model.compile(optimizer='adam', loss='mae', metrics=['mse'])

    # 5. ENTRENAMIENTO
    print("Entrenando modelo...")
    early_stop = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    history = model.fit(
        [X_zone_train, X_num_train], y_train,
        validation_data=([X_zone_val, X_num_val], y_val),
        epochs=30,
        batch_size=1024,
        callbacks=[early_stop],
        verbose=1
    )

    # 6. EVALUACIÓN Y GUARDADO (Igual que tu compañera)
    print("\nEvaluando en Test...")
    y_pred_test = model.predict([X_zone_test, X_num_test]).flatten()
    y_pred_val = model.predict([X_zone_val, X_num_val]).flatten()

    metrics_val = calcular_metricas(y_val, y_pred_val, "val")
    metrics_test = calcular_metricas(y_test, y_pred_test, "test")

    resultados = {
        "model_name": "NN_With_Embeddings",
        **metrics_val,
        **metrics_test
    }

    # Guardar todo en /results
    with open(os.path.join(RESULTS_DIR, 'red_neuronal_results.json'), 'w') as f:
        json.dump(resultados, f, indent=4)
    
    model.save(os.path.join(RESULTS_DIR, 'red_neuronal_model.keras'))
    joblib.dump(le, os.path.join(RESULTS_DIR, 'label_encoder_zonas.joblib'))
    joblib.dump(scaler, os.path.join(RESULTS_DIR, 'scaler_clima.joblib'))

    print(f"✅ Proceso completado. MAE en Test: {resultados['test_mae']:.4f}")

if __name__ == "__main__":
    entrenar_red_embeddings()