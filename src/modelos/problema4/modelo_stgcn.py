"""
DESCRIPCIÓN:
    Implementación de una Red de Grafos Espacio-Temporal (ST-GCN) usando Keras 3.
    Este modelo utiliza una Matriz de Adyacencia para aprender relaciones espaciales
    entre las zonas de Nueva York y predecir velocidades simultáneas.

OBJETIVO:
    Capturar la interdependencia del tráfico urbano mediante convoluciones de grafo,
    permitiendo que la información de una zona fluya hacia sus vecinas.

FLUJO DE TRABAJO:
    1. CARGA: Lee la matriz de adyacencia (.npy) y los datasets procesados.
    2. NORMALIZACIÓN: Pre-calcula la matriz de adyacencia normalizada (D^-1 * A).
    3. ARQUITECTURA: 
       - Capas GCN: Realizan la propagación de mensajes entre nodos vecinos.
       - Bloque Denso: Procesa las características extraídas para la predicción final.
    4. EVALUACIÓN: Calcula métricas (MAE, RMSE, R2, BIAS, MAPE) siguiendo el 
       estándar del proyecto para asegurar la comparabilidad.
    5. PERSISTENCIA: Guarda el modelo (.keras) y los resultados (.json) en 'results'.
"""

import os
import numpy as np
import keras
import json
import joblib
from keras import layers, ops
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'tlc_clean', 'problema4')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# 1. CAPA PERSONALIZADA DE GRAFOS (GCN)
# ==========================================
@keras.saving.register_keras_serializable()
class GraphConvLayer(layers.Layer):
    def __init__(self, num_outputs, adj_matrix, **kwargs):
        super().__init__(**kwargs)
        self.num_outputs = num_outputs
        
        # Normalización: D^-1 * A
        row_sum = np.array(adj_matrix.sum(1))
        d_inv = np.power(row_sum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        adj_norm = np.diag(d_inv).dot(adj_matrix).astype("float32")
        
        self.adj = ops.convert_to_tensor(adj_norm)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], self.num_outputs),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        # Propagación Espacial: A * X
        x = ops.matmul(self.adj, inputs)
        # Transformación: (A*X) * W
        return ops.matmul(x, self.kernel)

# ==========================================
# 2. FUNCIONES DE APOYO Y MÉTRICAS
# ==========================================
def calcular_metricas(y_true, y_pred, nombre_set):
    """Mismas métricas que Baseline y Red Neuronal para comparación justa."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    
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

# ==========================================
# 3. CONSTRUCCIÓN DEL MODELO ST-GCN
# ==========================================
def build_stgcn(num_nodes, num_features, adj_matrix):
    inputs = layers.Input(shape=(num_nodes, num_features), name="input_grafos")
    
    # Bloques GCN
    x = GraphConvLayer(64, adj_matrix, name="gcn_1")(inputs)
    x = layers.Activation("relu")(x)
    x = layers.BatchNormalization()(x)
    
    x = GraphConvLayer(32, adj_matrix, name="gcn_2")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)
    
    # Reducción y Predicción
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    
    outputs = layers.Dense(num_nodes, name="prediccion_velocidad")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mae', metrics=['mse'])
    
    return model

# ==========================================
# 4. LÓGICA DE EJECUCIÓN
# ==========================================
if __name__ == "__main__":
    ADJ_PATH = os.path.join(DATA_DIR, 'adj_matrix.npy')
    
    if os.path.exists(ADJ_PATH):
        adj = np.load(ADJ_PATH)
        n_nodos = adj.shape[0]
        n_features = 13 # El mismo número de variables que el Baseline/Red Neuronal
        
        print(f"--- Arquitectura ST-GCN lista (Backend: {keras.backend.backend()}) ---")
        model = build_stgcn(n_nodos, n_features, adj)
        model.summary()
        
        # Nota: Aquí falta la lógica del generador de datos para entrenamiento real
        print("\n✅ Estructura preparada y lista para recibir el generador de datos.")
    else:
        print(f"❌ Error: Matriz no encontrada en {ADJ_PATH}")