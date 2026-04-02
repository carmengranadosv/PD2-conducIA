import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import joblib
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

print("="*80)
print(" MLP CASCADING - OPTIMIZACIÓN DE ZONA PARA EL CONDUCTOR")
print("="*80)

# ============================================================
# 1. CONFIGURACIÓN Y CASCADING INPUT
# ============================================================
CONFIG = {
    'ruta_train': 'data/processed/tlc_clean/problema2/train.parquet',
    'ruta_demanda_p1': 'models/problema1/predicciones_demanda.csv', # El "Cascade"
    'output_dir': 'models/problema2',
    'filas_max': 200000
}

# Simulamos la entrada del Problema 1 si no existe el archivo aún
# (En el cluster esto leería la salida real del modelo LSTM de tus amigos)
def get_demanda_p1(df):
    return df['num_eventos'] * 1.2 # Placeholder del cascading input

# ============================================================
# 2. PREPARACIÓN DE DATOS (MULTICLASE)
# ============================================
cols = ['origen_id', 'oferta_inferida', 'temp_c', 'lluvia', 'num_eventos', 'espera_min', 'hora_sen']
df = pq.read_table(CONFIG['ruta_train'], columns=cols).to_pandas().head(CONFIG['filas_max'])

# CASCADING: Integramos la demanda del Problema 1
df['demanda_p1'] = get_demanda_p1(df)

# TARGET: ¿Éxito en < 10 min?
df['exito'] = (df['espera_min'] <= 10).astype(int)

# Filtramos solo los éxitos para que el modelo aprenda qué zonas son las "ganadoras"
df_ganadoras = df[df['exito'] == 1].copy()

# Encoder para las zonas (La "n" de tu arquitectura)
le = LabelEncoder()
df_ganadoras['zona_target'] = le.fit_transform(df_ganadoras['origen_id'])
n_zonas = len(le.classes_)

print(f" Arquitectura: Capa final Softmax para {n_zonas} zonas.")

# ============================================================
# 3. ENTRENAMIENTO MLP (SOFTMAX)
# ============================================================
X = df_ganadoras[['demanda_p1', 'oferta_inferida', 'temp_c', 'lluvia', 'hora_sen']]
y = df_ganadoras['zona_target']

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64), # Un poco más grande para manejar n zonas
    activation='relu',
    solver='adam',
    max_iter=50,
    verbose=True,
    random_state=42
)

print("\n Entrenando modelo de recomendación de zonas...")
mlp.fit(X_sc, y)

# ============================================================
# 4. GUARDADO Y EXPLICACIÓN
# ============================================================
joblib.dump(mlp, Path(CONFIG['output_dir']) / 'mlp_cascading.pkl')
joblib.dump(le, Path(CONFIG['output_dir']) / 'zona_encoder.pkl')

print(f"\n{'='*80}")
print(f" MODELO LISTO PARA EL CONDUCTOR")
print(f" Al recibir el contexto actual, el modelo devuelve una")
print(f" distribución Softmax sobre las {n_zonas} zonas.")
print(f"{'='*80}")