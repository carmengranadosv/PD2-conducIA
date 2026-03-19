import pandas as pd
import numpy as np
import os
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

TRAIN_FILE = os.path.join(DATA_DIR, 'train.parquet')
VAL_FILE = os.path.join(DATA_DIR, 'val.parquet')

def entrenar_red_neuronal():
    print(" Iniciando entrenamiento de la Red Neuronal ...")

    # LEER DATOS 
    print(" Leyendo datos ...")
    df_train = pq.ParquetFile(TRAIN_FILE).read_row_group(0).to_pandas()
    df_val = pq.ParquetFile(VAL_FILE).read_row_group(0).to_pandas()

    # SEPARAR X e y
    columnas_a_ignorar = ['propina', 'origen_id', 'destino_id']
    X_train = df_train.drop(columns=[col for col in columnas_a_ignorar if col in df_train.columns])
    y_train = df_train['propina'].values # .values lo convierte a formato NumPy para Keras
    
    X_val = df_val.drop(columns=[col for col in columnas_a_ignorar if col in df_val.columns])
    y_val = df_val['propina'].values

    # DEFINIR COLUMNAS
    columnas_categoricas = [
        'tipo_vehiculo', 'origen_zona', 'origen_barrio', 
        'destino_zona', 'destino_barrio', 'evento_tipo', 'franja_horaria'
    ]
    columnas_numericas = [col for col in X_train.columns if col not in columnas_categoricas]

    # PREPROCESAMIENTO EXPLÍCITO
    print(" Estandarizando números y codificando categorías...")
    preprocesador = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), columnas_numericas),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), columnas_categoricas)
        ],
        remainder='passthrough'
    )

    # Entrenamos el transformador con Train y lo aplicamos a Train y Val
    X_train_prep = preprocesador.fit_transform(X_train)
    X_val_prep = preprocesador.transform(X_val)
    
    # 5. CONSTRUIR LA RED NEURONAL (Arquitectura MLP)
    print(" Construyendo la arquitectura de la red...")
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
    model.compile(optimizer='rmsprop', loss='mae', metrics=['mse'])

    # ENTRENAMIENTO CON EARLY STOPPING
    print(" Entrenando la red neuronal ...")
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', # Vigila el error en validación
        patience=8,         # Si en X vueltas no mejora, para
        restore_best_weights=True # El mejor modelo, no con el último
    )

    history = model.fit(
        X_train_prep, y_train,
        epochs=50,          # Máximo de vueltas
        batch_size=512,     # Le pasamos los viajes de 512 en 512 para que aprenda rápido
        validation_data=(X_val_prep, y_val), # Examen en cada vuelta
        callbacks=[early_stopping],
        verbose=1           # Muestra la barra de progreso
    )

    # EVALUACIÓN Y MÉTRICAS
    print("\n Realizando predicciones finales sobre Validación...")
    y_pred = model.predict(X_val_prep).flatten() # flatten() lo aplana para poder restarlo con y_val
    
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    mean_error = np.mean(y_pred - y_val) 
    mape = np.mean(np.abs((y_val - y_pred) / (y_val + 1e-8))) * 100
    
    print("\n" + "="*45)
    print(" REPORTE RED NEURONAL PROFUNDA (Keras)")
    print("="*45)
    print(f"MAE:  ${mae:.4f} (Error absoluto medio)")
    print(f"RMSE: ${rmse:.4f} (Raíz del error cuadrático)")
    print(f"R2:    {r2:.4f} (Varianza explicada)")
    print(f"BIAS: ${mean_error:.4f} ({'Al alza' if mean_error > 0 else 'A la baja'})")
    print(f"MAPE:  {mape:.2f}% (Error porcentual)")
    print("="*45)

if __name__ == "__main__":
    entrenar_red_neuronal()