# src/modelos/problema2/gnn.py
"""
GNN simplificada sin PyTorch Geometric.
Construye el grafo de zonas por co-ocurrencia de viajes (origen→destino frecuentes).
La propagación de mensajes se implementa con multiplicación matricial (NumPy/PyTorch).
"""
import os, json
os.environ['KERAS_BACKEND'] = 'jax'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, warnings
warnings.filterwarnings('ignore')

import keras
from keras import layers, callbacks
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from pathlib import Path

np.random.seed(42)
keras.utils.set_random_seed(42)

FEATURES_DIR = Path('data/processed/tlc_clean/problema2/features')
RAW_P1_DIR   = Path('data/processed/tlc_clean/problema1/raw')
PARQUET_PATH = 'data/processed/tlc_clean/datos_final.parquet'
MODELS_DIR   = Path('models/problema2');    MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR    = Path('reports/problema2/plots');      PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR  = Path('reports/problema2/resultados'); RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    'hidden_dim': 64,
    'gnn_layers': 2,       # pasos de propagación de mensajes
    'dropout': 0.3,
    'epochs': 100,
    'batch_size': 512,
    'learning_rate': 0.001,
    'patience': 15,
    'top_k_vecinos': 10,   # cada zona conectada con sus 10 destinos más frecuentes
}

# ── 1. Cargar features ────────────────────────────────────────────────────────
print("Cargando features...")
df_train = pd.read_parquet(FEATURES_DIR / 'train.parquet')
df_val   = pd.read_parquet(FEATURES_DIR / 'val.parquet')
df_test  = pd.read_parquet(FEATURES_DIR / 'test.parquet')

with open(FEATURES_DIR / 'metadata.json') as f:
    meta = json.load(f)

FEATURES_NUM = [f for f in meta['feature_cols'] if f != 'origen_id']
TARGET = 'target'

# ── 2. Encoding zonas ─────────────────────────────────────────────────────────
le = joblib.load(MODELS_DIR / 'zona_encoder.pkl')
for df in [df_train, df_val, df_test]:
    df['zona_enc'] = df['origen_id'].apply(
        lambda z: le.transform([z])[0] if z in le.classes_ else -1)

df_train = df_train[df_train['zona_enc'] >= 0].copy()
df_val   = df_val[df_val['zona_enc'] >= 0].copy()
df_test  = df_test[df_test['zona_enc'] >= 0].copy()

N_ZONAS = len(le.classes_)
print(f"  Zonas en grafo: {N_ZONAS}")

# ── 3. Construir matriz de adyacencia ─────────────────────────────────────────
# A[i,j] = frecuencia de viajes origen_i → destino_j (normalizada)
print("Construyendo grafo de zonas (co-ocurrencia origen→destino)...")

import pyarrow.parquet as pq

A = np.zeros((N_ZONAS, N_ZONAS), dtype=np.float32)

pf = pq.ParquetFile(PARQUET_PATH)
for i in range(pf.num_row_groups):
    batch = pf.read_row_group(i, columns=['origen_id', 'destino_id',
                                           'fecha_inicio'])
    df_g = batch.to_pandas()
    # Solo datos de train (antes del corte)
    corte = pd.Timestamp(json.load(
        open('data/processed/tlc_clean/problema2/raw/metadata.json')
    )['cortes']['corte_train'])
    df_g = df_g[pd.to_datetime(df_g['fecha_inicio']) < corte]

    for _, row in df_g[['origen_id', 'destino_id']].iterrows():
        o = row['origen_id']
        d = row['destino_id']
        if o in le.classes_ and d in le.classes_:
            oi = le.transform([o])[0]
            di = le.transform([d])[0]
            A[oi, di] += 1
    print(f"\r   Row group {i+1}/{pf.num_row_groups}...", end='', flush=True)

print()

# Quedarse solo con top-K vecinos por zona (sparse)
K = CONFIG['top_k_vecinos']
for i in range(N_ZONAS):
    row = A[i].copy()
    if row.sum() == 0:
        continue
    threshold = np.sort(row)[-K] if row.sum() > 0 else 0
    A[i, row < threshold] = 0

# Normalizar filas (D^-1 * A) — propagación de mensajes estándar
row_sums = A.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
A_norm = A / row_sums
A_norm = A_norm.astype(np.float32)

np.save(MODELS_DIR / 'gnn_adj_matrix.npy', A_norm)
print(f"  Matriz de adyacencia: {A_norm.shape}, densidad: {(A_norm > 0).mean():.4f}")

# ── 4. Preparar features ──────────────────────────────────────────────────────
FEATURES_FINAL = FEATURES_NUM + ['zona_enc']

scaler = StandardScaler()
X_train_raw = df_train[FEATURES_FINAL].fillna(0).values.astype(np.float32)
X_val_raw   = df_val[FEATURES_FINAL].fillna(0).values.astype(np.float32)
X_test_raw  = df_test[FEATURES_FINAL].fillna(0).values.astype(np.float32)

X_train_sc = scaler.fit_transform(X_train_raw)
X_val_sc   = scaler.transform(X_val_raw)
X_test_sc  = scaler.transform(X_test_raw)
joblib.dump(scaler, MODELS_DIR / 'gnn_scaler.pkl')

y_train = df_train[TARGET].values.astype(np.float32)
y_val   = df_val[TARGET].values.astype(np.float32)
y_test  = df_test[TARGET].values.astype(np.float32)

zona_train = df_train['zona_enc'].values
zona_val   = df_val['zona_enc'].values
zona_test  = df_test['zona_enc'].values

# ── 5. Propagación de mensajes ────────────────────────────────────────────────
def gnn_propagate(X_sc, zona_enc, A_norm, n_layers):
    """
    Para cada muestra, añade features agregadas de sus zonas vecinas.
    Propagación: h_v = mean(h_u para u en vecinos(v))
    Se hace de forma matricial: primero construimos features por zona,
    luego propagamos con A_norm, luego extraemos por muestra.
    """
    n_features = X_sc.shape[1]
    n_zonas    = A_norm.shape[0]

    # Agregar features medias por zona (usando datos del split actual)
    H = np.zeros((n_zonas, n_features), dtype=np.float32)
    counts = np.zeros(n_zonas, dtype=np.float32)
    for i, z in enumerate(zona_enc):
        H[z] += X_sc[i]
        counts[z] += 1
    counts[counts == 0] = 1
    H = H / counts[:, None]

    # Propagación: H_new = A_norm @ H (l veces)
    H_prop = H.copy()
    for _ in range(n_layers):
        H_prop = A_norm @ H_prop

    # Extraer features propagadas por muestra
    return H_prop[zona_enc]

print(f"Propagando mensajes ({CONFIG['gnn_layers']} capas)...")
X_train_gnn = gnn_propagate(X_train_sc, zona_train, A_norm, CONFIG['gnn_layers'])
X_val_gnn   = gnn_propagate(X_val_sc,   zona_val,   A_norm, CONFIG['gnn_layers'])
X_test_gnn  = gnn_propagate(X_test_sc,  zona_test,  A_norm, CONFIG['gnn_layers'])

# Concatenar features originales + propagadas
X_train_final = np.concatenate([X_train_sc, X_train_gnn], axis=1).astype(np.float32)
X_val_final   = np.concatenate([X_val_sc,   X_val_gnn],   axis=1).astype(np.float32)
X_test_final  = np.concatenate([X_test_sc,  X_test_gnn],  axis=1).astype(np.float32)

print(f"  Dimensión final features: {X_train_final.shape[1]} "
      f"(original {X_train_sc.shape[1]} + propagadas {X_train_gnn.shape[1]})")

# ── 6. Modelo clasificador sobre features GNN ─────────────────────────────────
pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

inputs = keras.Input(shape=(X_train_final.shape[1],))
x = layers.Dense(CONFIG['hidden_dim'], activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(CONFIG['dropout'])(x)
x = layers.Dense(CONFIG['hidden_dim'] // 2, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(CONFIG['dropout'])(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, output, name='GNN_P2')
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
    loss='binary_crossentropy',
    metrics=['AUC'],
)
model.summary()

# ── 7. Entrenar ───────────────────────────────────────────────────────────────
cbs = [
    callbacks.EarlyStopping(monitor='val_auc', patience=CONFIG['patience'],
                            restore_best_weights=True, verbose=1, mode='max'),
    callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5,
                                min_lr=1e-6, verbose=1, mode='max'),
]

history = model.fit(
    X_train_final, y_train,
    validation_data=(X_val_final, y_val),
    epochs=CONFIG['epochs'],
    batch_size=CONFIG['batch_size'],
    class_weight={0: pos_weight, 1: 1.0},
    callbacks=cbs, verbose=1,
)

# ── 8. Evaluar ────────────────────────────────────────────────────────────────
def metrics(y_true, y_prob, label=""):
    y_pred = (y_prob >= 0.5).astype(int)
    auc = float(roc_auc_score(y_true, y_prob))
    ap  = float(average_precision_score(y_true, y_prob))
    rep = classification_report(y_true, y_pred, output_dict=True)
    f1  = float(rep['weighted avg']['f1-score'])
    if label:
        print(f"  {label}: AUC={auc:.4f}  AP={ap:.4f}  F1={f1:.4f}")
    return {'auc': auc, 'ap': ap, 'f1': f1}

prob_val  = model.predict(X_val_final,  batch_size=1024, verbose=0).flatten()
prob_test = model.predict(X_test_final, batch_size=1024, verbose=0).flatten()

print("\nRESULTADOS:")
m_val  = metrics(y_val,  prob_val,  "Val ")
m_test = metrics(y_test, prob_test, "Test")

print("\nClassification Report (Test):")
print(classification_report(y_test, (prob_test >= 0.5).astype(int)))

# ── 9. Guardar ────────────────────────────────────────────────────────────────
model.save(MODELS_DIR / 'gnn_model.keras')

results = {
    'modelo': 'GNN (Graph Neural Network simplificada)',
    'nota': f'Grafo zona→zona por co-ocurrencia. {CONFIG["gnn_layers"]} capas de '
            f'propagación. Top-{CONFIG["top_k_vecinos"]} vecinos por zona.',
    'config': CONFIG,
    'val':  m_val,
    'test': m_test,
}
with open(RESULTS_DIR / 'gnn_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ── 10. Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history.history['auc'],     label='Train', color='#2c3e50')
axes[0].plot(history.history['val_auc'], label='Val',   color='#e74c3c')
axes[0].set_title('AUC por época', fontweight='bold')
axes[0].set_xlabel('Época'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train', color='#2c3e50')
axes[1].plot(history.history['val_loss'], label='Val',   color='#e74c3c')
axes[1].set_title('Loss (BCE)', fontweight='bold')
axes[1].set_xlabel('Época'); axes[1].legend(); axes[1].grid(alpha=0.3)

from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, prob_test)
axes[2].plot(fpr, tpr, color='#9b59b6', lw=2, label=f'AUC={m_test["auc"]:.3f}')
axes[2].plot([0,1],[0,1],'k--', alpha=0.4)
axes[2].set_xlabel('FPR'); axes[2].set_ylabel('TPR')
axes[2].set_title('Curva ROC — Test', fontweight='bold')
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.suptitle('GNN — Optimización de Zona (Problema 2)', fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'gnn_resultados.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nPlot:       {PLOTS_DIR / 'gnn_resultados.png'}")
print(f"Resultados: {RESULTS_DIR / 'gnn_results.json'}")