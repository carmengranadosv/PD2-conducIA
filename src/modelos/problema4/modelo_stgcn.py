"""
MÓDULO: modelo_stgcn.py
DESCRIPCIÓN: Implementación de Red de Grafos Espacio-Temporal (ST-GCN).
Este modelo utiliza una Matriz de Adyacencia para aprender relaciones espaciales.
"""
#USAR KERAS
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

# ==========================================
# 1. CAPA PERSONALIZADA DE GRAFOS (GCN)
# ==========================================
class GraphConvLayer(layers.Layer):
    """
    Implementa la operación: A * X * W
    Donde A es la matriz de adyacencia normalizada.
    """
    def __init__(self, num_outputs, adj_matrix, **kwargs):
        super(GraphConvLayer, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        
        # --- Normalización de la Matriz de Adyacencia ---
        # Calculamos D^-1 * A (Normalización por filas)
        row_sum = np.array(adj_matrix.sum(1))
        d_inv = np.power(row_sum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = np.diag(d_inv)
        adj_norm = d_mat_inv.dot(adj_matrix)
        
        # Convertimos a constante de TensorFlow para que no se entrene
        self.adj = tf.constant(adj_norm, dtype=tf.float32)

    def build(self, input_shape):
        # input_shape: (Batch, Nodos, Features)
        self.kernel = self.add_weight(
            "kernel", 
            shape=[int(input_shape[-1]), self.num_outputs],
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, inputs):
        # 1. Propagación Espacial (Mensajes entre vecinos)
        # Usamos einsum para aplicar la matriz de adyacencia a cada batch
        # 'nn' (matriz adj) x 'bnf' (batch, nodos, features) -> 'bnf'
        x = tf.einsum('nn,bnf->bnf', self.adj, inputs)
        
        # 2. Transformación de características (Pesos W)
        return tf.matmul(x, self.kernel)

# ==========================================
# 2. CONSTRUCCIÓN DEL MODELO ST-GCN
# ==========================================
def build_stgcn(num_nodes, num_features, adj_matrix):
    """
    Crea el modelo combinando capas de grafos y densas.
    """
    # Entrada: (Batch, 262 zonas, N características)
    inputs = layers.Input(shape=(num_nodes, num_features), name="input_grafos")
    
    # Bloque 1: Convolución de Grafo + Activación
    x = GraphConvLayer(64, adj_matrix, name="gcn_1")(inputs)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Bloque 2: Segunda capa de Grafo para captar relaciones indirectas
    x = GraphConvLayer(32, adj_matrix, name="gcn_2")(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Bloque 3: Reducción y Predicción
    # Aplanamos para conectar con la capa densa final
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    
    # Salida: Predecimos la velocidad de cada zona (262 valores)
    # Si quieres predecir solo una zona a la vez, cambia esto a 1
    outputs = layers.Dense(num_nodes, name="prediccion_velocidad")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compilación con métricas estándar de regresión
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mae',
        metrics=['mse']
    )
    
    return model

# ==========================================
# 3. LÓGICA DE PRUEBA (SOLO PARA VERIFICAR)
# ==========================================
if __name__ == "__main__":
    from pathlib import Path
    
    # Intentamos cargar la matriz para una prueba rápida
    BASE_DIR = Path(__file__).resolve().parents[3]
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'tlc_clean', 'problema4', 'adj_matrix.npy')
    
    if os.path.exists(DATA_PATH):
        adj = np.load(DATA_PATH)
        n_nodos = adj.shape[0]
        n_features = 10 # Ejemplo: hora, día, clima, etc.
        
        print(f"--- Creando Modelo ST-GCN para {n_nodos} nodos ---")
        stgcn_model = build_stgcn(n_nodos, n_features, adj)
        stgcn_model.summary()
        print("\n✅ Modelo construido y compilado correctamente.")
    else:
        print(f"❌ No se encontró la matriz en {DATA_PATH}. Ejecuta generar_grafo.py primero.")