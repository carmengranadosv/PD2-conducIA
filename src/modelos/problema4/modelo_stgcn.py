import os
import numpy as np
import pandas as pd
import keras
import json
import joblib
from keras import layers, ops
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'tlc_clean', 'problema4')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# 1. CAPA PERSONALIZADA GCN (A * X * W)
# ==========================================
@keras.saving.register_keras_serializable()
class GraphConvLayer(layers.Layer):
    def __init__(self, num_outputs, adj_matrix, **kwargs):
        super().__init__(**kwargs)
        self.num_outputs = num_outputs
        self.adj_matrix = adj_matrix 
        
        # Normalización de la matriz de adyacencia
        row_sum = np.array(adj_matrix.sum(1))
        d_inv = np.power(row_sum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        adj_norm = np.diag(d_inv).dot(adj_matrix).astype("float32")
        self.adj = ops.convert_to_tensor(adj_norm)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel", shape=(input_shape[-1], self.num_outputs),
            initializer="glorot_uniform", trainable=True
        )

    def call(self, inputs):
        x = ops.matmul(self.adj, inputs)
        return ops.matmul(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_outputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_outputs": self.num_outputs,
            "adj_matrix": self.adj_matrix,
        })
        return config

# ==========================================
# 2. FUNCIÓN DE MÉTRICAS (IGUAL QUE EN LOS OTROS MODELOS)
# ==========================================
def calcular_metricas(y_true, y_pred, nombre_set):
    """Calcula el pack de métricas estándar del proyecto."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    
    # R2 Score
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    bias = np.mean(y_pred - y_true)
    
    # MAPE con protección contra división por cero
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return {
        f"{nombre_set}_mae": float(mae),
        f"{nombre_set}_rmse": float(rmse),
        f"{nombre_set}_r2": float(r2),
        f"{nombre_set}_bias": float(bias),
        f"{nombre_set}_mape": float(mape)
    }

# ==========================================
# 3. PREPARACIÓN DE DATOS
# ==========================================
def preparar_datos_grafo(df, scaler=None):
    df_data = df.copy()
    num_nodos_esperados = 262 
    
    features_cols = [
        'temp_c', 'precipitation', 'viento_kmh', 'lluvia', 'nieve', 
        'hay_lluvia', 'hay_nieve', 'es_festivo', 'num_eventos', 
        'dia_semana', 'es_fin_semana', 'hora_sen', 'hora_cos'
    ]
    
    if scaler is None:
        scaler = StandardScaler()
        df_data[features_cols] = scaler.fit_transform(df_data[features_cols])
    else:
        df_data[features_cols] = scaler.transform(df_data[features_cols])

    X_raw = df_data[features_cols].values
    y_raw = df_data['velocidad_mph'].values
    
    num_muestras = len(X_raw) // num_nodos_esperados
    X = X_raw[:num_muestras * num_nodos_esperados].reshape(num_muestras, num_nodos_esperados, -1)
    y = y_raw[:num_muestras * num_nodos_esperados].reshape(num_muestras, num_nodos_esperados)
    
    return X, y, scaler

# ==========================================
# 4. EJECUCIÓN PRINCIPAL
# ==========================================
def ejecutar_stgcn():
    print("🚀 Iniciando entrenamiento y evaluación completa ST-GCN...")
    
    # 1. CARGA DE DATOS
    # Cargamos Train, Val y Test para tener la comparativa completa
    train_df = pd.read_parquet(os.path.join(DATA_DIR, 'train_p4.parquet'))
    val_df = pd.read_parquet(os.path.join(DATA_DIR, 'val_p4.parquet'))
    test_df = pd.read_parquet(os.path.join(DATA_DIR, 'test_p4.parquet'))
    adj = np.load(os.path.join(DATA_DIR, 'adj_matrix.npy'))
    
    # 2. PREPARACIÓN (Cubo de datos: Muestras, 262, 13)
    # Importante: El scaler se ajusta con Train y se aplica a Val y Test
    X_train, y_train, scaler = preparar_datos_grafo(train_df)
    X_val, y_val, _ = preparar_datos_grafo(val_df, scaler)
    X_test, y_test, _ = preparar_datos_grafo(test_df, scaler)

    # 3. CONSTRUCCIÓN DEL MODELO
    inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = GraphConvLayer(64, adj)(inputs)
    x = layers.Activation("relu")(x)
    x = layers.BatchNormalization()(x)
    x = GraphConvLayer(32, adj)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(X_train.shape[1])(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mae')

    # 4. ENTRENAMIENTO
    # Ya no necesitamos validation_split porque evaluaremos con el set de Val real
    print("Entrenando modelo...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    # 5. PREDICCIONES PARA EVALUACIÓN
    print("\nCalculando métricas finales para Validación y Test...")
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    # 6. APLANAR DATOS (Necesario para las fórmulas de métricas globales)
    # Validación
    y_val_true_flat = y_val.flatten()
    y_val_pred_flat = y_pred_val.flatten()
    # Test
    y_test_true_flat = y_test.flatten()
    y_test_pred_flat = y_pred_test.flatten()

    # 7. CÁLCULO DE MÉTRICAS OFICIALES
    metrics_val = calcular_metricas(y_val_true_flat, y_val_pred_flat, "val")
    metrics_test = calcular_metricas(y_test_true_flat, y_test_pred_flat, "test")
    
    # 8. ESTRUCTURA DEL JSON FINAL
    resultados_finales = {
        "model_name": "ST-GCN_Spatial_Temporal",
        **metrics_val,
        **metrics_test
    }
    
    # 9. GUARDADO DE RESULTADOS Y MODELO
    with open(os.path.join(RESULTS_DIR, 'stgcn_results.json'), 'w') as f:
        json.dump(resultados_finales, f, indent=4)
        
    model.save(os.path.join(RESULTS_DIR, 'stgcn_model.keras'))
    
    print(f"\n✅ Proceso completado con éxito.")
    print(f"Resultados guardados en: {os.path.join(RESULTS_DIR, 'stgcn_results.json')}")

if __name__ == "__main__":
    ejecutar_stgcn()