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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

np.random.seed(42)
keras.utils.set_random_seed(42)

FEATURES_DIR = Path('data/processed/tlc_clean/problema1/features')
MODELS_DIR   = Path('models/problema1');    MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR    = Path('reports/problema1/plots');      PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR  = Path('reports/problema1/resultados'); RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    'lookback': 24,
    'lstm_units_1': 128,
    'lstm_units_2': 64,
    'dense_units': 32,
    'dropout': 0.2,
    'epochs': 100,
    'batch_size': 256,
    'learning_rate': 0.001,
    'patience': 15,
    'max_train_seq': 500_000,
    'max_val_seq': 150_000,
    'top_n_zonas': 50,
}

# ── 1. Cargar ─────────────────────────────────────────────────────────────────
print("Cargando features...")
df_train = pd.read_parquet(FEATURES_DIR / 'train.parquet')
df_val   = pd.read_parquet(FEATURES_DIR / 'val.parquet')
df_test  = pd.read_parquet(FEATURES_DIR / 'test.parquet')
assert len(df_val) > 0, "Val vacío: revisar features.py"

# ── Filtrar top-N zonas por demanda total (calculado solo sobre train) ────────
TOP_N = CONFIG['top_n_zonas']
zonas_top = (
    df_train.groupby('origen_id')['demanda']
    .sum()
    .nlargest(TOP_N)
    .index
)
df_train = df_train[df_train['origen_id'].isin(zonas_top)].copy()
df_val   = df_val[df_val['origen_id'].isin(zonas_top)].copy()
df_test  = df_test[df_test['origen_id'].isin(zonas_top)].copy()

print(f"  Top {TOP_N} zonas → train={len(df_train):,} | val={len(df_val):,} | test={len(df_test):,}")

FEATURES = [
    'hora', 'dia_semana', 'dia_mes', 'mes_num', 'es_finde', 'demanda',
    'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h',
    'roll_mean_3h', 'roll_std_3h', 'roll_mean_24h', 'roll_std_24h',
    'media_hist', 'temp_c', 'precipitation', 'viento_kmh', 'velocidad_mph',
    'lluvia', 'nieve', 'es_festivo', 'num_eventos',
]
TARGET = 'target'

# ── 2. Encoding zonas ─────────────────────────────────────────────────────────
le = LabelEncoder()
le.fit(df_train['origen_id'])
for df in [df_train, df_val, df_test]:
    df['zona_enc'] = df['origen_id'].apply(
        lambda z: le.transform([z])[0] if z in le.classes_ else -1
    )
df_train = df_train[df_train['zona_enc'] >= 0].copy()
df_val   = df_val[df_val['zona_enc'] >= 0].copy()
df_test  = df_test[df_test['zona_enc'] >= 0].copy()
joblib.dump(le, MODELS_DIR / 'lstm_zona_encoder.pkl')
print(f"  Zonas codificadas: {len(le.classes_)}")

# ── 3. Log1p en el target ─────────────────────────────────────────────────────
print("Aplicando log1p al target...")
for df in [df_train, df_val, df_test]:
    df['target_log'] = np.log1p(df[TARGET])

# ── 4. Normalización ─────────────────────────────────────────────────────────
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_sc = scaler_X.fit_transform(df_train[FEATURES])
X_val_sc   = scaler_X.transform(df_val[FEATURES])
X_test_sc  = scaler_X.transform(df_test[FEATURES])

y_train_sc = scaler_y.fit_transform(df_train[['target_log']]).flatten()
y_val_sc   = scaler_y.transform(df_val[['target_log']]).flatten()

y_test_real = df_test[TARGET].values

joblib.dump(scaler_X, MODELS_DIR / 'lstm_scaler_X.pkl')
joblib.dump(scaler_y, MODELS_DIR / 'lstm_scaler_y.pkl')

# ── 5. Crear secuencias por zona ──────────────────────────────────────────────
def create_sequences(X_sc, y_sc, zona_enc, timestamps, lookback):
    X_list, y_list = [], []
    for z in np.unique(zona_enc):
        mask  = zona_enc == z
        idx   = np.where(mask)[0]
        order = np.argsort(timestamps[idx])
        idx   = idx[order]
        X_z, y_z = X_sc[idx], y_sc[idx]
        if len(X_z) < lookback + 1:
            continue
        for i in range(len(X_z) - lookback):
            X_list.append(X_z[i:i+lookback])
            y_list.append(y_z[i+lookback])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

LB = CONFIG['lookback']
print(f"Creando secuencias (lookback={LB}h)...")

X_tr, y_tr = create_sequences(
    X_train_sc, y_train_sc,
    df_train['zona_enc'].values, df_train['timestamp_hora'].values, LB)

X_v, y_v = create_sequences(
    X_val_sc, y_val_sc,
    df_val['zona_enc'].values, df_val['timestamp_hora'].values, LB)

y_test_log_sc = scaler_y.transform(
    np.log1p(y_test_real).reshape(-1, 1)).flatten()
X_te, y_te_sc = create_sequences(
    X_test_sc, y_test_log_sc,
    df_test['zona_enc'].values, df_test['timestamp_hora'].values, LB)

y_te_real_seq = np.expm1(
    scaler_y.inverse_transform(y_te_sc.reshape(-1, 1)).flatten()
).clip(0)

print(f"  Train: {X_tr.shape} | Val: {X_v.shape} | Test: {X_te.shape}")

# ── 6. Subsampleo si es necesario ────────────────────────────────────────────
rng = np.random.default_rng(42)

MAX_TR = CONFIG['max_train_seq']
if len(X_tr) > MAX_TR:
    idx = np.sort(rng.choice(len(X_tr), MAX_TR, replace=False))
    X_tr, y_tr = X_tr[idx], y_tr[idx]
    print(f"  Subsampling train → {MAX_TR:,}")

MAX_V = CONFIG['max_val_seq']
if len(X_v) > MAX_V:
    idx_v = np.sort(rng.choice(len(X_v), MAX_V, replace=False))
    X_v, y_v = X_v[idx_v], y_v[idx_v]
    print(f"  Subsampling val → {MAX_V:,}")

# ── 7. Modelo ─────────────────────────────────────────────────────────────────
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

# ── 8. Entrenar ───────────────────────────────────────────────────────────────
cbs = [
    callbacks.EarlyStopping(
        monitor='val_loss', patience=CONFIG['patience'],
        restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=5, min_lr=1e-6, verbose=1),
]

history = model.fit(
    X_tr, y_tr,
    validation_data=(X_v, y_v),
    epochs=CONFIG['epochs'],
    batch_size=CONFIG['batch_size'],
    callbacks=cbs, verbose=1,
)

# ── 9. Evaluar ────────────────────────────────────────────────────────────────
def predict_real(X):
    pred_sc  = model.predict(X, batch_size=512, verbose=0).flatten()
    pred_log = scaler_y.inverse_transform(pred_sc.reshape(-1, 1)).flatten()
    return np.clip(np.expm1(pred_log), 0, None)

def metrics(y_true, y_pred, label=""):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100)
    if label:
        print(f"  {label}: RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.4f}  MAPE={mape:.1f}%")
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

y_v_real = np.expm1(
    scaler_y.inverse_transform(y_v.reshape(-1, 1)).flatten()).clip(0)
y_v_pred  = predict_real(X_v)
y_te_pred = predict_real(X_te)

print("\nRESULTADOS:")
m_val  = metrics(y_v_real,      y_v_pred,  "Val ")
m_test = metrics(y_te_real_seq, y_te_pred, "Test")

# ── 10. Guardar ───────────────────────────────────────────────────────────────
model.save(MODELS_DIR / 'lstm_model.keras')

results = {
    'modelo': 'LSTM',
    'nota': f'Entrenado sobre top {TOP_N} zonas por demanda total',
    'config': {k: v for k, v in CONFIG.items()},
    'features': FEATURES,
    'zonas_top': list(zonas_top.astype(str)),
    'val':  m_val,
    'test': m_test,
}
with open(RESULTS_DIR / 'lstm_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ── 11. Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(history.history['loss'],     label='Train', color='#2c3e50')
axes[0].plot(history.history['val_loss'], label='Val',   color='#e74c3c')
axes[0].set_title('Loss (MSE — espacio log)', fontweight='bold')
axes[0].set_xlabel('Época'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history.history['mae'],     label='Train', color='#2c3e50')
axes[1].plot(history.history['val_mae'], label='Val',   color='#e74c3c')
axes[1].set_title('MAE (espacio log normalizado)', fontweight='bold')
axes[1].set_xlabel('Época'); axes[1].legend(); axes[1].grid(alpha=0.3)

n = min(300, len(y_te_real_seq))
axes[2].plot(y_te_real_seq[:n], label='Real',      color='#2c3e50', lw=2)
axes[2].plot(y_te_pred[:n],     label='LSTM pred', color='#27ae60', lw=1.5, alpha=0.8)
axes[2].set_title('Predicho vs Real — Test (top zonas)', fontweight='bold')
axes[2].set_xlabel('Timestep'); axes[2].set_ylabel('Viajes/hora')
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.suptitle(f'LSTM — Top {TOP_N} zonas (Problema 1)', fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'lstm_resultados.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nPlot:       {PLOTS_DIR / 'lstm_resultados.png'}")
print(f"Resultados: {RESULTS_DIR / 'lstm_results.json'}")