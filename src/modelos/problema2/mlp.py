# src/modelos/problema2/mlp.py
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
MODELS_DIR   = Path('models/problema2');    MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR    = Path('reports/problema2/plots');      PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR  = Path('reports/problema2/resultados'); RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    'hidden_layers': [256, 128, 64],
    'dropout': 0.2, # bajamos un poco el dropout para que aprenda más
    'epochs': 10, # ponemos más épocas para darle margen pero el Early Stopping parará si no es necesario
    'batch_size': 1024, # doble de bathc para mayor estabilidad en JAX
    'learning_rate': 0.001,
    'patience': 3,
}

# ── 1. Cargar ─────────────────────────────────────────────────────────────────
print("Cargando features...")
df_train = pd.read_parquet(FEATURES_DIR / 'train.parquet')
df_val   = pd.read_parquet(FEATURES_DIR / 'val.parquet')
df_test  = pd.read_parquet(FEATURES_DIR / 'test.parquet')

with open(FEATURES_DIR / 'metadata.json') as f:
    meta = json.load(f)

FEATURES_NUM = [f for f in meta['feature_cols'] if f != 'origen_id']
TARGET = 'target'

print(f"  Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

# ── 2. Encoding zona ──────────────────────────────────────────────────────────
le = joblib.load(MODELS_DIR / 'zona_encoder.pkl')
for df in [df_train, df_val, df_test]:
    df['zona_enc'] = df['origen_id'].apply(
        lambda z: le.transform([z])[0] if z in le.classes_ else -1)

FEATURES_FINAL = FEATURES_NUM + ['zona_enc']

X_train = df_train[FEATURES_FINAL].fillna(0).values.astype(np.float32)
X_val   = df_val[FEATURES_FINAL].fillna(0).values.astype(np.float32)
X_test  = df_test[FEATURES_FINAL].fillna(0).values.astype(np.float32)
y_train = df_train[TARGET].values.astype(np.float32)
y_val   = df_val[TARGET].values.astype(np.float32)
y_test  = df_test[TARGET].values.astype(np.float32)

# ── 3. Normalizar ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
joblib.dump(scaler, MODELS_DIR / 'mlp_scaler.pkl')

# Balance de clases
pos_weight = 0.5  # Esto hará que el modelo sea un poco más exigente para dar un "1"
# pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
print(f"  Peso clase positiva: {pos_weight:.2f}")

# ── 4. Modelo MLP ─────────────────────────────────────────────────────────────
inputs = keras.Input(shape=(X_train.shape[1],))
x = inputs
for units in CONFIG['hidden_layers']:
    x = layers.Dense(units, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG['dropout'])(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, output, name='MLP_P2')
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
    loss='binary_crossentropy',
    metrics=['AUC'],
)
model.summary()

# ── 5. Entrenar ───────────────────────────────────────────────────────────────
cbs = [
    callbacks.EarlyStopping(monitor='val_AUC', patience=CONFIG['patience'], min_delta=0.001,
                            restore_best_weights=True, verbose=1, mode='max'),
    callbacks.ReduceLROnPlateau(monitor='val_AUC', factor=0.5, patience=5,
                                min_lr=1e-6, verbose=1, mode='max'),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=CONFIG['epochs'],
    batch_size=CONFIG['batch_size'],
    class_weight={0: pos_weight, 1: 1.0},
    callbacks=cbs, verbose=1,
)

# ── 6. Evaluar ────────────────────────────────────────────────────────────────
def metrics(y_true, y_prob, label=""):
    y_pred = (y_prob >= 0.5).astype(int)
    auc = float(roc_auc_score(y_true, y_prob))
    ap  = float(average_precision_score(y_true, y_prob))
    rep = classification_report(y_true, y_pred, output_dict=True)
    f1  = float(rep['weighted avg']['f1-score'])
    if label:
        print(f"  {label}: AUC={auc:.4f}  AP={ap:.4f}  F1={f1:.4f}")
    return {'auc': auc, 'ap': ap, 'f1': f1}

prob_val  = model.predict(X_val,  batch_size=1024, verbose=0).flatten()
prob_test = model.predict(X_test, batch_size=1024, verbose=0).flatten()

print("\nRESULTADOS:")
m_val  = metrics(y_val,  prob_val,  "Val ")
m_test = metrics(y_test, prob_test, "Test")

print("\nClassification Report (Test):")
print(classification_report(y_test, (prob_test >= 0.5).astype(int)))

# ── 7. Guardar ────────────────────────────────────────────────────────────────
model.save(MODELS_DIR / 'mlp_model.keras')

results = {
    'modelo': 'MLP (Red Neuronal Densa)',
    'nota': 'Incluye demanda_p1 como cascading input del Problema 1',
    'config': CONFIG,
    'features': FEATURES_FINAL,
    'val':  m_val,
    'test': m_test,
}
with open(RESULTS_DIR / 'mlp_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ── 8. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history.history['AUC'],     label='Train', color='#2c3e50')
axes[0].plot(history.history['val_AUC'], label='Val',   color='#e74c3c')
axes[0].set_title('AUC por época', fontweight='bold')
axes[0].set_xlabel('Época'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train', color='#2c3e50')
axes[1].plot(history.history['val_loss'], label='Val',   color='#e74c3c')
axes[1].set_title('Loss (BCE)', fontweight='bold')
axes[1].set_xlabel('Época'); axes[1].legend(); axes[1].grid(alpha=0.3)

from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, prob_test)
axes[2].plot(fpr, tpr, color='#3498db', lw=2, label=f'AUC={m_test["auc"]:.3f}')
axes[2].plot([0,1],[0,1],'k--', alpha=0.4)
axes[2].set_xlabel('FPR'); axes[2].set_ylabel('TPR')
axes[2].set_title('Curva ROC — Test', fontweight='bold')
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.suptitle('MLP — Optimización de Zona (Problema 2)', fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'mlp_resultados.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nPlot:       {PLOTS_DIR / 'mlp_resultados.png'}")
print(f"Resultados: {RESULTS_DIR / 'mlp_results.json'}")