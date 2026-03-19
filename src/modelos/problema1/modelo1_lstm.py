# src/modelos/problema1/modelo1_lstm.py

"""
Modelo LSTM - Problema 1: Predicción de Demanda por Zona
Compatible con Python 3.14 usando Keras 3
"""

import os
os.environ['KERAS_BACKEND'] = 'jax'  # Backend JAX para Python 3.14

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Keras 3
import keras
from keras import layers, callbacks

# ML utilities
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Seeds
np.random.seed(42)
keras.utils.set_random_seed(42)

print("="*80)
print(" LSTM - PREDICCIÓN DE DEMANDA POR ZONA (Problema 1)")
print("="*80)

# ============================================================
# CONFIGURACIÓN
# ============================================================

CONFIG = {
    # Datos
    'features_dir': 'data/processed/tlc_clean/problema1/features',
    
    # LSTM
    'lookback': 12,  # 12 horas de historia
    
    # Modelo
    'lstm_units_1': 64,
    'lstm_units_2': 32,
    'dense_units': 16,
    'dropout': 0.2,
    
    # Entrenamiento
    'epochs': 50,
    'batch_size': 64,
    'learning_rate': 0.001,
    'patience': 10,
    
    # Output
    'output_dir': 'models/problema1',
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

print(f"\n Config:")
print(f"   Lookback: {CONFIG['lookback']} horas")
print(f"   LSTM: {CONFIG['lstm_units_1']}→{CONFIG['lstm_units_2']}")
print(f"   Epochs: {CONFIG['epochs']}, Batch: {CONFIG['batch_size']}")

# ============================================================
# PASO 1: CARGAR DATOS
# ============================================================

print(f"\n{'='*80}")
print(" PASO 1: CARGAR DATOS")
print("="*80)

features_dir = Path(CONFIG['features_dir'])

df_train = pd.read_parquet(features_dir / 'train.parquet')
df_val = pd.read_parquet(features_dir / 'val.parquet')
df_test = pd.read_parquet(features_dir / 'test.parquet')

# Si val está vacío, usar parte de test
if len(df_val) == 0:
    print("  Val vacío, usando 50% de test como val")
    df_full_test = df_test.copy()
    df_val = df_full_test.sample(frac=0.5, random_state=42)
    df_test = df_full_test.drop(df_val.index)

print(f" Datos cargados:")
print(f"   Train: {len(df_train):,} registros")
print(f"   Val:   {len(df_val):,} registros")
print(f"   Test:  {len(df_test):,} registros")

# ============================================================
# PASO 2: DEFINIR FEATURES
# ============================================================

print(f"\n{'='*80}")
print(" PASO 2: FEATURES")
print("="*80)

FEATURES = [
    'hora', 'dia_semana', 'dia_mes', 'mes_num', 'es_finde',
    'demanda',
    'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h',
    'roll_mean_3h', 'roll_std_3h', 'roll_mean_24h', 'roll_std_24h',
    'media_hist',
    'temp_c', 'precipitation', 'viento_kmh', 'velocidad_mph',
    'lluvia', 'nieve', 'es_festivo', 'num_eventos',
]

TARGET = 'target'

print(f"   Features: {len(FEATURES)}")
print(f"   Target: {TARGET}")

# ============================================================
# PASO 3: ENCODING ZONAS
# ============================================================

print(f"\n{'='*80}")
print(" PASO 3: ENCODING ZONAS")
print("="*80)

le = LabelEncoder()
le.fit(df_train['origen_id'])

df_train['zona_enc'] = le.transform(df_train['origen_id'])
df_val['zona_enc'] = le.transform(df_val['origen_id'])
df_test['zona_enc'] = le.transform(df_test['origen_id'])

encoder_path = Path(CONFIG['output_dir']) / 'zona_encoder.pkl'
joblib.dump(le, encoder_path)

print(f"   Zonas: {len(le.classes_)}")
print(f"    Encoder guardado: {encoder_path}")

# ============================================================
# PASO 4: NORMALIZACIÓN
# ============================================================

print(f"\n{'='*80}")
print(" PASO 4: NORMALIZACIÓN")
print("="*80)

scaler = StandardScaler()
scaler.fit(df_train[FEATURES])

df_train_sc = df_train.copy()
df_val_sc = df_val.copy()
df_test_sc = df_test.copy()

df_train_sc[FEATURES] = scaler.transform(df_train[FEATURES])
df_val_sc[FEATURES] = scaler.transform(df_val[FEATURES])
df_test_sc[FEATURES] = scaler.transform(df_test[FEATURES])

scaler_path = Path(CONFIG['output_dir']) / 'scaler.pkl'
joblib.dump(scaler, scaler_path)

print(f"    Scaler guardado: {scaler_path}")

# ============================================================
# PASO 5: CREAR SECUENCIAS
# ============================================================

print(f"\n{'='*80}")
print(" PASO 5: CREAR SECUENCIAS LSTM")
print("="*80)

def create_sequences(df, lookback, features, target):
    """Crea secuencias por zona"""
    X_list, y_list, z_list = [], [], []
    
    for zona in df['zona_enc'].unique():
        df_z = df[df['zona_enc'] == zona].sort_values('timestamp_hora')
        
        if len(df_z) < lookback + 1:
            continue
        
        X_z = df_z[features].values
        y_z = df_z[target].values
        
        for i in range(len(X_z) - lookback):
            X_list.append(X_z[i:i+lookback])
            y_list.append(y_z[i+lookback])
            z_list.append(zona)
    
    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(z_list, dtype=np.int32)
    )

lookback = CONFIG['lookback']

print(f"   Creando secuencias (lookback={lookback})...")

X_train, y_train, z_train = create_sequences(df_train_sc, lookback, FEATURES, TARGET)
X_val, y_val, z_val = create_sequences(df_val_sc, lookback, FEATURES, TARGET)
X_test, y_test, z_test = create_sequences(df_test_sc, lookback, FEATURES, TARGET)

print(f"\n Secuencias creadas:")
print(f"   Train: X={X_train.shape}, y={y_train.shape}")
print(f"   Val:   X={X_val.shape}, y={y_val.shape}")
print(f"   Test:  X={X_test.shape}, y={y_test.shape}")

# ============================================================
# PASO 6: CONSTRUIR MODELO LSTM
# ============================================================

print(f"\n{'='*80}")
print("  PASO 6: CONSTRUIR MODELO")
print("="*80)

model = keras.Sequential([
    layers.Input(shape=(lookback, len(FEATURES))),
    
    layers.LSTM(CONFIG['lstm_units_1'], return_sequences=True, dropout=CONFIG['dropout']),
    layers.BatchNormalization(),
    
    layers.LSTM(CONFIG['lstm_units_2'], dropout=CONFIG['dropout']),
    layers.BatchNormalization(),
    
    layers.Dense(CONFIG['dense_units'], activation='relu'),
    layers.Dropout(CONFIG['dropout']),
    
    layers.Dense(1)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
    loss='mse',
    metrics=['mae']
)

print("\n Arquitectura:")
model.summary()

# ============================================================
# PASO 7: ENTRENAR
# ============================================================

print(f"\n{'='*80}")
print(" PASO 7: ENTRENAR")
print("="*80)

cbs = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=CONFIG['patience'],
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
]

print(f"\n Entrenando...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=CONFIG['epochs'],
    batch_size=CONFIG['batch_size'],
    callbacks=cbs,
    verbose=1
)

print(f"\n Entrenamiento completado")

# ============================================================
# PASO 8: EVALUAR
# ============================================================

print(f"\n{'='*80}")
print(" PASO 8: EVALUAR")
print("="*80)

y_pred_test = model.predict(X_test, batch_size=256, verbose=0).flatten()
y_pred_test = np.clip(y_pred_test, 0, None)

rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f"\n TEST RESULTS:")
print(f"   RMSE: {rmse:.4f}")
print(f"   MAE:  {mae:.4f}")
print(f"   R²:   {r2:.4f}")

# ============================================================
# PASO 9: GUARDAR
# ============================================================

print(f"\n{'='*80}")
print(" PASO 9: GUARDAR")
print("="*80)

model_path = Path(CONFIG['output_dir']) / 'lstm_model.keras'
model.save(model_path)
print(f"    Modelo: {model_path}")

# Config
config_path = Path(CONFIG['output_dir']) / 'config.json'
with open(config_path, 'w') as f:
    json.dump({
        **CONFIG,
        'features': FEATURES,
        'metrics': {'test_rmse': float(rmse), 'test_mae': float(mae), 'test_r2': float(r2)}
    }, f, indent=2)
print(f"    Config: {config_path}")

# Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(history.history['loss'], label='Train')
ax[0].plot(history.history['val_loss'], label='Val')
ax[0].set_title('Loss')
ax[0].legend()
ax[0].grid(True, alpha=0.3)

ax[1].plot(history.history['mae'], label='Train')
ax[1].plot(history.history['val_mae'], label='Val')
ax[1].set_title('MAE')
ax[1].legend()
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = Path(CONFIG['output_dir']) / 'training.png'
plt.savefig(plot_path, dpi=150)
print(f"    Plot: {plot_path}")

print(f"\n{'='*80}")
print(" COMPLETADO")
print("="*80)