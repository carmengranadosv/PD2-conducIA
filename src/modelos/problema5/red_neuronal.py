import pandas as pd
import numpy as np
import os
import json
import joblib
from pathlib import Path
import pyarrow.parquet as pq

# Preprocesamiento de Scikit-Learn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Deep Learning (Keras/TensorFlow)
from tensorflow.keras import models, layers, callbacks, Input, optimizers
from tensorflow.keras.layers import BatchNormalization

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
    """Lee el archivo con muestreo para no saturar la RAM."""
    print(f"Leyendo {ruta.name}...")
    parquet_file = pq.ParquetFile(ruta)
    total_rows = parquet_file.metadata.num_rows
    
    if total_rows > sample_size:
        print(f"   Archivo muy grande ({total_rows:,} filas). Muestreando {sample_size:,}...")
        df = pd.read_parquet(ruta).sample(n=sample_size, random_state=42)
    else:
        df = pd.read_parquet(ruta)
    
    return optimizar_tipos(df)

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

def entrenar_red_neuronal():
    print("Iniciando entrenamiento de la Red Neuronal (MLP) ...")

    # CARGAR DATOS
    df_train = leer_datos_optimizados(TRAIN_FILE, sample_size=28000000)
    df_val = leer_datos_optimizados(VAL_FILE, sample_size=6000000)
    df_test = leer_datos_optimizados(TEST_FILE, sample_size=6000000)

    # Definir variables
    target = 'propina'
    columnas_a_ignorar = [target, 'origen_id', 'destino_id', 'destino_zona', 'destino_barrio']
    columnas_embeddings = ['tipo_vehiculo', 'origen_zona', 'origen_barrio', 'evento_tipo', 'franja_horaria']

    columnas_numericas = [c for c in df_train.columns if c not in columnas_embeddings + columnas_a_ignorar]

    # 3. PREPROCESAMIENTO
    encoders = {}
    X_emb_train, X_emb_val, X_emb_test = [], [], []

    for col in columnas_embeddings:
        le = LabelEncoder()
        # Asegurar que todos sean string y manejar valores desconocidos
        df_train[col] = df_train[col].astype(str)
        df_val[col] = df_val[col].astype(str)
        df_test[col] = df_test[col].astype(str)

        X_emb_train.append(le.fit_transform(df_train[col]))
        # Mapeo de seguridad para evitar errores de categorías nuevas en val/test
        X_emb_val.append(le.transform(df_val[col]))
        X_emb_test.append(le.transform(df_test[col]))
        encoders[col] = le

    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(df_train[columnas_numericas])
    X_num_val = scaler.transform(df_val[columnas_numericas])
    X_num_test = scaler.transform(df_test[columnas_numericas])

    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    # Liberar X original para ganar RAM
    del df_train, df_val, df_test

    # CONSTRUIR LA RED NEURONAL (Arquitectura MLP)
    print("Construyendo la arquitectura de la red...")
    emb_inputs = []
    emb_layers = []

    for i, col in enumerate(columnas_embeddings):
        num_classes = len(encoders[col].classes_)
        dim = min(50, (num_classes + 1) // 2) 
        
        inp = Input(shape=(1,), name=f"input_{col}")
        # input_dim es num_classes + 1 para manejar posibles índices fuera de rango
        emb = layers.Embedding(input_dim=num_classes + 1, output_dim=dim)(inp)
        emb = layers.Flatten()(emb)
        
        emb_inputs.append(inp)
        emb_layers.append(emb)

    input_num = Input(shape=(len(columnas_numericas),), name="input_num")
    combined = layers.Concatenate()(emb_layers + [input_num])

    x = layers.Dense(128, activation='relu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1)(x)

    model = models.Model(inputs=emb_inputs + [input_num], outputs=output)
    model.compile(optimizer=optimizers.Adam(0.001), loss='mae')

    # ENTRENAMIENTO CON EARLY STOPPING
    print("Entrenando la red neuronal ...")
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', # Vigila el error en validación
        patience=5,         # Si en X vueltas no mejora, para
        restore_best_weights=True # El mejor modelo, no con el último
    )

    history = model.fit(
        X_emb_train + [X_num_train], y_train,
        epochs=30,          # Máximo de vueltas
        batch_size=2048,     # Le pasamos los viajes de 512 en 512 para que aprenda rápido
        validation_data=(X_emb_val + [X_num_val], y_val), # Examen en cada vuelta
        callbacks=[early_stopping],
        verbose=1           # Muestra la barra de progreso
    )

    # EVALUACIÓN Y MÉTRICAS
    print("\nEvaluando en Test...")
    y_pred_val = model.predict(X_emb_val + [X_num_val], batch_size=2048).flatten()
    y_pred_test = model.predict(X_emb_test + [X_num_test], batch_size=2048).flatten() # flatten() lo aplana para poder restarlo con y_val
    
    metrics_val = calcular_metricas(y_val, y_pred_val, "val")
    metrics_test = calcular_metricas(y_test, y_pred_test, "test")

    resultados = {
        "model_name": "DeepLearning_MLP_28M",
        "data_info": {
            "train_samples": len(y_train),
            "val_samples": len(y_val),
            "test_samples": len(y_test)
        },
        **metrics_val,
        **metrics_test
    }

    # GUARDAMOS LOS RESULTADOS
    res_path = REPORTS_DIR / 'red_neuronal_results.json'
    with open(res_path, 'w') as f:
        json.dump(resultados, f, indent=4)
    
    model.save(MODELS_DIR / 'red_neuronal_model.keras')
    joblib.dump(encoders, MODELS_DIR / 'red_neuronal_encoders.joblib')
    joblib.dump(scaler, MODELS_DIR / 'red_neuronal_scaler.joblib')

    print("\nEvaluación en TEST finalizada:")
    print(f"  MAE:  ${resultados['test_mae']:.4f}")
    print(f"  RMSE: ${resultados['test_rmse']:.4f}")
    print(f"  R2:   {resultados['test_r2']:.4f}")
    print(f"  BIAS: ${resultados['test_bias']:.4f}")
    print(f"  Resultados guardados en: {res_path}")

if __name__ == "__main__":
    entrenar_red_neuronal()