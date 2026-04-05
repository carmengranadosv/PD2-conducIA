# src/modelos/problema1/lstm.py
import os, json
os.environ['KERAS_BACKEND'] = 'jax'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, warnings
warnings.filterwarnings('ignore')

import keras
from keras import layers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

np.random.seed(42)
keras.utils.set_random_seed(42)

# ── Directorios ───────────────────────────────────────────────────────────────
FEATURES_DIR = Path('data/processed/tlc_clean/problema1/features')
MODELS_DIR   = Path('models/problema1');   MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR  = Path('reports/problema1');  REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR    = REPORTS_DIR / 'plots';      PLOTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR  = REPORTS_DIR / 'resultados'; RESULTS_DIR.mkdir(exist_ok=True)

CONFIG = {
    'lookback': 12, 'lstm_units_1': 64, 'lstm_units_2': 32,
    'dense_units': 16, 'dropout': 0.2,
    'epochs': 50, 'batch_size': 64, 'learning_rate': 0.001, 'patience': 10,
}

# ── 1. Cargar datos ───────────────────────────────────────────────────────────
print("Cargando features...")
df_train = pd.read_parquet(FEATURES_DIR / 'train.parquet')
df_val   = pd.read_parquet(FEATURES_DIR / 'val.parquet')
df_test  = pd.read_parquet(FEATURES_DIR / 'test.parquet')

assert len(df_val) > 0, "Val vacío: revisar agregacion.py o division_datos.py"

print(f"  Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

FEATURES = [
    'hora', 'dia_semana', 'dia_mes', 'mes_num', 'es_finde', 'demanda',
    'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h',
    'roll_mean_3h', 'roll_std_3h', 'roll_mean_24h', 'roll_std_24h',
    'media_hist', 'temp_c', 'precipitation', 'viento_kmh', 'velocidad_mph',
    'lluvia', 'nieve', 'es_festivo', 'num_eventos',
]
TARGET = 'target'

# ── 2. Encoding zonas ─────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df_train['origen_id'])
for df in [df_train, df_val, df_test]:
    # Zonas en val/test que no están en train → asignar -1 y excluir
    df['zona_enc'] = df['origen_id'].apply(
        lambda z: le.transform([z])[0] if z in le.classes_ else -1
    )

df_train = df_train[df_train['zona_enc'] >= 0].copy()
df_val   = df_val[df_val['zona_enc'] >= 0].copy()
df_test  = df_test[df_test['zona_enc'] >= 0].copy()

joblib.dump(le, MODELS_DIR / 'lstm_zona_encoder.pkl')

# ── 3. Normalización features Y TARGET ───────────────────────────────────────
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_sc = scaler_X.fit_transform(df_train[FEATURES])
X_val_sc   = scaler_X.transform(df_val[FEATURES])
X_test_sc  = scaler_X.transform(df_test[FEATURES])

y_train_sc = scaler_y.fit_transform(df_train[[TARGET]]).flatten()
y_val_sc   = scaler_y.transform(df_val[[TARGET]]).flatten()
# Test: guardamos el target original para métricas reales
y_test_real = df_test[TARGET].values

joblib.dump(scaler_X, MODELS_DIR / 'lstm_scaler_X.pkl')
joblib.dump(scaler_y, MODELS_DIR / 'lstm_scaler_y.pkl')

# ── 4. Crear secuencias ───────────────────────────────────────────────────────
def create_sequences(X_sc, y_sc, zona_enc_col, timestamps, lookback):
    """Crea secuencias por zona, preservando orden temporal."""
    X_list, y_list = [], []
    zonas = np.unique(zona_enc_col)
    for z in zonas:
        mask = zona_enc_col == z
        idx = np.where(mask)[0]
        # Ordenar por timestamp dentro de la zona
        order = np.argsort(timestamps[idx])
        idx = idx[order]
        X_z = X_sc[idx]
        y_z = y_sc[idx]
        if len(X_z) < lookback + 1:
            continue
        for i in range(len(X_z) - lookback):
            X_list.append(X_z[i:i+lookback])
            y_list.append(y_z[i+lookback])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

LB = CONFIG['lookback']
print(f"Creando secuencias (lookback={LB}h)...")

X_tr, y_tr = create_sequences(X_train_sc, y_train_sc,
    df_train['zona_enc'].values, df_train['timestamp_hora'].values, LB)
X_v, y_v = create_sequences(X_val_sc, y_val_sc,
    df_val['zona_enc'].values, df_val['timestamp_hora'].values, LB)
X_te, _ = create_sequences(X_test_sc, np.zeros(len(X_test_sc)),
    df_test['zona_enc'].values, df_test['timestamp_hora'].values, LB)

# Target real correspondiente a las secuencias de test
_, y_te_real = create_sequences(
    X_test_sc, y_test_real,
    df_test['zona_enc'].values, df_test['timestamp_hora'].values, LB)

print(f"  Train: {X_tr.shape} | Val: {X_v.shape} | Test: {X_te.shape}")

# ── 5. Modelo ─────────────────────────────────────────────────────────────────
model = keras.Sequential([
    layers.Input(shape=(LB, len(FEATURES))),
    layers.LSTM(CONFIG['lstm_units_1'], return_sequences=True, dropout=CONFIG['dropout']),
    layers.BatchNormalization(),
    layers.LSTM(CONFIG['lstm_units_2'], dropout=CONFIG['dropout']),
    layers.BatchNormalization(),
    layers.Dense(CONFIG['dense_units'], activation='relu'),
    layers.Dropout(CONFIG['dropout']),
    layers.Dense(1),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
    loss='mse', metrics=['mae']
)
model.summary()

# ── 6. Entrenar ───────────────────────────────────────────────────────────────
cbs = [
    callbacks.EarlyStopping(monitor='val_loss', patience=CONFIG['patience'],
                            restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience=5, min_lr=1e-6, verbose=1),
]

history = model.fit(
    X_tr, y_tr,
    validation_data=(X_v, y_v),
    epochs=CONFIG['epochs'],
    batch_size=CONFIG['batch_size'],
    callbacks=cbs, verbose=1,
)

# ── 7. Evaluar — desnormalizar predicciones ───────────────────────────────────
def metrics(y_true, y_pred, label=""):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100)
    if label:
        print(f"  {label}: RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.4f}  MAPE={mape:.1f}%")
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

# Val
y_v_pred_sc = model.predict(X_v, batch_size=256, verbose=0).flatten()
y_v_pred    = np.clip(scaler_y.inverse_transform(y_v_pred_sc.reshape(-1,1)).flatten(), 0, None)
y_v_real    = scaler_y.inverse_transform(y_v.reshape(-1,1)).flatten()

# Test
y_te_pred_sc = model.predict(X_te, batch_size=256, verbose=0).flatten()
y_te_pred    = np.clip(scaler_y.inverse_transform(y_te_pred_sc.reshape(-1,1)).flatten(), 0, None)

print("\nRESULTADOS:")
m_val  = metrics(y_v_real,   y_v_pred,  "Val ")
m_test = metrics(y_te_real,  y_te_pred, "Test")

# ── 8. Guardar modelo ─────────────────────────────────────────────────────────
model.save(MODELS_DIR / 'lstm_model.keras')

# Schema unificado de resultados
results = {
    'modelo': 'LSTM',
    'config': CONFIG,
    'features': FEATURES,
    'val':  m_val,
    'test': m_test,
}
with open(RESULTS_DIR / 'lstm_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ── 9. Plots — en reports/problema1/plots/ ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Training curves
axes[0].plot(history.history['loss'],     label='Train', color='#2c3e50')
axes[0].plot(history.history['val_loss'], label='Val',   color='#e74c3c')
axes[0].set_title('Loss (MSE)', fontweight='bold')
axes[0].set_xlabel('Época'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history.history['mae'],     label='Train', color='#2c3e50')
axes[1].plot(history.history['val_mae'], label='Val',   color='#e74c3c')
axes[1].set_title('MAE', fontweight='bold')
axes[1].set_xlabel('Época'); axes[1].legend(); axes[1].grid(alpha=0.3)

# Predicho vs Real (primeras 200 muestras de test)
n = min(200, len(y_te_real))
axes[2].plot(y_te_real[:n],  label='Real',       color='#2c3e50', lw=2)
axes[2].plot(y_te_pred[:n],  label='LSTM pred',  color='#27ae60', lw=1.5, alpha=0.8)
axes[2].set_title('Predicho vs Real — Test', fontweight='bold')
axes[2].set_xlabel('Timestep'); axes[2].set_ylabel('Viajes/hora')
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.suptitle('LSTM — Predicción Demanda (Problema 1)', fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'lstm_resultados.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nPlot guardado: {PLOTS_DIR / 'lstm_resultados.png'}")
print(f"Resultados: {RESULTS_DIR / 'lstm_results.json'}")