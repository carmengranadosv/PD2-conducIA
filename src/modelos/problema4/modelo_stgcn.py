"""
MÓDULO: modelo_stgcn.py
DESCRIPCIÓN: Implementación de Red de Grafos Espacio-Temporal (ST-GCN) usando Keras 3.
"""

import os
import numpy as np
import keras
from keras import layers, ops
from pathlib import Path

# ==========================================
# 1. CAPA PERSONALIZADA DE GRAFOS (GCN)
# ==========================================
@keras.saving.register_keras_serializable()
class GraphConvLayer(layers.Layer):
    """
    Implementa la operación espacial: A * X * W
    Keras 3 utiliza 'ops' para que funcione en TF, PyTorch o JAX.
    """
    def __init__(self, num_outputs, adj_matrix, **kwargs):
        super().__init__(**kwargs)
        self.num_outputs = num_outputs
        
        # --- Normalización de la Matriz de Adyacencia (Pre-procesado) ---
        row_sum = np.array(adj_matrix.sum(1))
        d_inv = np.power(row_sum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = np.diag(d_inv)
        adj_norm = d_mat_inv.dot(adj_matrix).astype("float32")
        
        # En Keras 3, las constantes se manejan mejor como pesos no entrenables o buffers
        self.adj = ops.convert_to_tensor(adj_norm)

    def build(self, input_shape):
        # input_shape: (Batch, Nodos, Features)
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], self.num_outputs),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        # 1. Propagación Espacial: A * X
        # Keras 3 'ops.matmul' maneja el broadcasting de forma eficiente
        x = ops.matmul(self.adj, inputs)
        
        # 2. Transformación de características: (A*X) * W
        return ops.matmul(x, self.kernel)

# ==========================================
# 2. CONSTRUCCIÓN DEL MODELO ST-GCN
# ==========================================
def build_stgcn(num_nodes, num_features, adj_matrix):
    """
    Crea el modelo ST-GCN con la API funcional de Keras 3.
    """
    inputs = layers.Input(shape=(num_nodes, num_features), name="input_grafos")
    
    # Capa 1 de Grafo
    x = GraphConvLayer(64, adj_matrix, name="gcn_1")(inputs)
    x = layers.Activation("relu")(x)
    x = layers.BatchNormalization()(x)
    
    # Capa 2 de Grafo
    x = GraphConvLayer(32, adj_matrix, name="gcn_2")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)
    
    # Bloque de salida
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    
    # Salida: Predecimos velocidad para cada una de las 262 zonas
    outputs = layers.Dense(num_nodes, name="prediccion_velocidad")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mae",
        metrics=["mse"]
    )
    
    return model

# ==========================================
# 3. LÓGICA DE PRUEBA
# ==========================================
if __name__ == "__main__":
    # Intentamos cargar la matriz para una prueba rápida
    # Ajustamos la ruta según tu estructura: src/modelos/problema4/
    BASE_DIR = Path(__file__).resolve().parents[3]
    DATA_PATH = BASE_DIR / 'data' / 'processed' / 'tlc_clean' / 'problema4' / 'adj_matrix.npy'
    
    if DATA_PATH.exists():
        adj = np.load(DATA_PATH)
        n_nodos = adj.shape[0]
        n_features = 10 # Features por nodo (ej. velocidad_t-1, temp, hora...)
        
        print(f"--- Iniciando Keras 3 (Backend: {keras.backend.backend()}) ---")
        print(f"--- Creando Modelo para {n_nodos} zonas ---")
        
        model = build_stgcn(n_nodos, n_features, adj)
        model.summary()
        print("\n✅ Modelo Keras 3 listo para entrenar.")
    else:
        print(f"❌ Matriz no encontrada en: {DATA_PATH}")