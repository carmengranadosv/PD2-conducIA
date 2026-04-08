import pandas as pd
import numpy as np
import os
import json
import joblib
from pathlib import Path
import pyarrow.parquet as pq

# Preprocesamiento de Scikit-Learn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Deep Learning (Keras/TensorFlow)
from tensorflow.keras import models, layers, callbacks, Input, optimizers
from tensorflow.keras.layers import BatchNormalization

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parents[3] 
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'tlc_clean', 'problema5')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(DATA_DIR, 'train.parquet')
VAL_FILE = os.path.join(DATA_DIR, 'val.parquet')
TEST_FILE = os.path.join(DATA_DIR, 'test.parquet')

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
    df_train = pq.ParquetFile(TRAIN_FILE).read_row_group(0).to_pandas()
    df_val = pq.ParquetFile(VAL_FILE).read_row_group(0).to_pandas()
    df_test = pq.ParquetFile(TEST_FILE).read_row_group(0).to_pandas()

    # SEPARAR X e y
    columnas_a_ignorar = ['propina', 'origen_id', 'destino_id', 'destino_zona', 'destino_barrio']

    def prepare_xy(df):
        X = df.drop(columns=[col for col in columnas_a_ignorar if col in df.columns])
        y = df['propina'].values
        return X, y

    X_train, y_train = prepare_xy(df_train)
    X_val, y_val = prepare_xy(df_val)
    X_test, y_test = prepare_xy(df_test)

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
    columnas_numericas = [col for col in X_train.columns if col not in columnas_categoricas]

    # PREPROCESAMIENTO
    preprocesador = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), columnas_numericas),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), columnas_categoricas)
        ],
        remainder='passthrough'
    )

    # Entrenamos el transformador con Train y lo aplicamos a Train, Val y Test
    X_train_prep = preprocesador.fit_transform(X_train)
    X_val_prep = preprocesador.transform(X_val)
    X_test_prep = preprocesador.transform(X_test)
    
    # CONSTRUIR LA RED NEURONAL (Arquitectura MLP)
    print("Construyendo la arquitectura de la red...")
    input_dim = X_train_prep.shape[1] # Cuántas columnas nos han quedado tras el OneHotEncoder
    
    model = models.Sequential([
        Input(shape=(input_dim,)), # Nueva forma recomendada por Keras para evitar warnings
        layers.Dense(128, activation='relu'),
        BatchNormalization(),
        layers.Dropout(0.2), # Apagamos el 20% de neuronas para que no se memorice los barrios

        layers.Dense(64, activation='relu'),
        BatchNormalization(),
        layers.Dropout(0.1),

        layers.Dense(32, activation='relu'),
        layers.Dense(1) # Salida lineal
    ])

    optimizador = optimizers.Adam(learning_rate=0.001)

    # Compilación
    model.compile(optimizer='adam', loss='mae', metrics=['mse'])

    # ENTRENAMIENTO CON EARLY STOPPING
    print("Entrenando la red neuronal ...")
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', # Vigila el error en validación
        patience=5,         # Si en X vueltas no mejora, para
        restore_best_weights=True # El mejor modelo, no con el último
    )

    history = model.fit(
        X_train_prep, y_train,
        epochs=30,          # Máximo de vueltas
        batch_size=1024,     # Le pasamos los viajes de 512 en 512 para que aprenda rápido
        validation_data=(X_val_prep, y_val), # Examen en cada vuelta
        callbacks=[early_stopping],
        verbose=1           # Muestra la barra de progreso
    )

    # EVALUACIÓN Y MÉTRICAS
    print("\nEvaluando en Test...")
    y_pred_val = model.predict(X_val_prep, batch_size=1024).flatten()
    y_pred_test = model.predict(X_test_prep, batch_size=1024).flatten() # flatten() lo aplana para poder restarlo con y_val
    
    metrics_val = calcular_metricas(y_val, y_pred_val, "val")
    metrics_test = calcular_metricas(y_test, y_pred_test, "test")

    resultados = {
        "model_name": "DeepLearning_MLP",
        **metrics_val,
        **metrics_test
    }

    # GUARDAMOS LOS RESULTADOS
    res_path = os.path.join(RESULTS_DIR, 'red_neuronal_results.json')
    with open(res_path, 'w') as f:
        json.dump(resultados, f, indent=4)
    
    model.save(os.path.join(RESULTS_DIR, 'red_neuronal_model.keras'))
    joblib.dump(preprocesador, os.path.join(RESULTS_DIR, 'red_neuronal_prepro.joblib'))

    print("\nEvaluación en TEST finalizada:")
    print(f"  MAE:  ${resultados['test_mae']:.4f}")
    print(f"  RMSE: ${resultados['test_rmse']:.4f}")
    print(f"  R2:   {resultados['test_r2']:.4f}")
    print(f"  BIAS: ${resultados['test_bias']:.4f}")
    print(f"  Resultados guardados en: {res_path}")

if __name__ == "__main__":
    entrenar_red_neuronal()