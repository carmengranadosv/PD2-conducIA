# src/modelos/problema2/analizar_predicciones.py

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, precision_recall_curve

print("=" * 90)
print("ANALISIS DETALLADO DE PREDICCIONES - MLP (PROBLEMA 2)")
print("=" * 90)

# Configuración de rutas
FEATURES_DIR = Path('data/processed/tlc_clean/problema2/features')
MODEL_DIR    = Path('models/problema2')
REPORTS_DIR  = Path('reports/problema2/plots')
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. CARGAR DATOS Y MODELO
print("Cargando datos y modelos...")
df_test = pd.read_parquet(FEATURES_DIR / 'test.parquet')
with open(FEATURES_DIR / 'metadata.json') as f:
    meta = json.load(f)

scaler = joblib.load(MODEL_DIR / 'mlp_scaler.pkl')
le = joblib.load(MODEL_DIR / 'zona_encoder.pkl')

import keras
model = keras.models.load_model(MODEL_DIR / 'mlp_model.keras')

# 2. PREPARAR FEATURES
# Identificar columnas según el entrenamiento (excluyendo origen_id que se codifica aparte)
FEATURES_NUM = [f for f in meta['feature_cols'] if f != 'origen_id']
df_test['zona_enc'] = df_test['origen_id'].apply(lambda z: le.transform([z])[0] if z in le.classes_ else -1)
FEATURES_FINAL = FEATURES_NUM + ['zona_enc']

X_test_raw = df_test[FEATURES_FINAL].fillna(0)
X_test_sc = scaler.transform(X_test_raw)
y_test = df_test['target'].values

# 3. GENERAR PREDICCIONES
print("Generando predicciones con la Red Neuronal...")
probs = model.predict(X_test_sc, batch_size=2048, verbose=0).flatten()
preds = (probs >= 0.5).astype(int)

df_test['prob_mlp'] = probs
df_test['pred_mlp'] = preds

# 4. ANÁLISIS POR ZONA
print("\nTOP 10 ZONAS CON MAYOR TASA DE ÉXITO REAL (Y PREDICCIÓN):")
top_zonas = df_test.groupby('origen_id').agg({
    'target': ['count', 'mean'],
    'prob_mlp': 'mean'
}).sort_values(('target', 'mean'), ascending=False).head(10)
print(top_zonas.round(3))

# 5. VISUALIZACIONES
print("\nGenerando visualizaciones de rendimiento...")
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# A. Matriz de Confusión
ax1 = fig.add_subplot(gs[0, 0])
cm = confusion_matrix(y_test, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax1)
ax1.set_title('Matriz de Confusión', fontweight='bold')
ax1.set_xlabel('Predicho (0=Normal, 1=Top)'); ax1.set_ylabel('Real')

# B. Distribución de Probabilidades por Clase
ax2 = fig.add_subplot(gs[0, 1])
sns.histplot(data=df_test, x='prob_mlp', hue='target', bins=50, alpha=0.6, ax=ax2, palette=['#e74c3c', '#2ecc71'])
ax2.axvline(0.5, color='black', linestyle='--')
ax2.set_title('Distribución de Probabilidades', fontweight='bold')

# C. Curva Precision-Recall
ax3 = fig.add_subplot(gs[0, 2])
precision, recall, _ = precision_recall_curve(y_test, probs)
ax3.plot(recall, precision, color='#3498db', lw=3)
ax3.set_title('Curva Precision-Recall', fontweight='bold')
ax3.set_xlabel('Recall (Cobertura)'); ax3.set_ylabel('Precision (Calidad)')
ax3.grid(alpha=0.3)

# D. Evolución Temporal Real (USANDO ventana_inicio)
ax4 = fig.add_subplot(gs[1, :])

# Elegimos la zona del Top 10 que más registros tenga para que el gráfico sea rico en datos
# (En tus datos, la zona 54 es ideal)
zona_ejemplo = top_zonas[('target', 'count')].idxmax()

# Detectar columna temporal (Priorizamos 'ventana_inicio' que sabemos que existe)
col_temporal = None
for col in ['ventana_inicio', 'timestamp_hora', 'fecha', 'hora', 'pickup_hour']:
    if col in df_test.columns:
        col_temporal = col
        break

try:
    if col_temporal:
        # Convertir a datetime para asegurar el formato en el eje X
        df_test[col_temporal] = pd.to_datetime(df_test[col_temporal])
        
        # Filtrar datos de la zona elegida y ordenar cronológicamente
        df_zona = df_test[df_test['origen_id'] == zona_ejemplo].sort_values(col_temporal).head(120)
        
        ax4.plot(df_zona[col_temporal], df_zona['target'], 'o', label='Real (Exito)', alpha=0.4, color='gray', markersize=4)
        ax4.plot(df_zona[col_temporal], df_zona['prob_mlp'], '-', label='Probabilidad MLP', color='#2ecc71', linewidth=2)
        ax4.fill_between(df_zona[col_temporal], 0.5, df_zona['prob_mlp'], 
                         where=(df_zona['prob_mlp'] >= 0.5), color='#2ecc71', alpha=0.2)
        
        # Formateador de fechas para el eje X
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))
        plt.xticks(rotation=45)
        ax4.set_xlabel(f"Eje Temporal ({col_temporal})")
    else:
        # Fallback por si no encuentra columna temporal
        df_zona = df_test[df_test['origen_id'] == zona_ejemplo].head(120)
        ax4.plot(range(len(df_zona)), df_zona['target'], 'o', label='Real', alpha=0.4)
        ax4.plot(range(len(df_zona)), df_zona['prob_mlp'], '-', label='Probabilidad MLP', color='#2ecc71')
        ax4.set_xlabel("Muestras consecutivas (No se detectó columna temporal)")
except Exception as e:
    print(f"Nota: No se pudo generar el gráfico temporal detallado ({e})")

ax4.axhline(0.5, color='red', linestyle='--', alpha=0.5)
ax4.set_title(f'Análisis Temporal Real: Zona {zona_ejemplo} (Probabilidad de ser zona TOP)', fontweight='bold')
ax4.legend(loc='upper right')

# Título General y Guardado
auc_val = roc_auc_score(y_test, probs)
plt.suptitle(f"Análisis detallado MLP - Problema 2 (AUC-ROC: {auc_val:.4f})", fontsize=16, fontweight='bold')
plot_path = REPORTS_DIR / 'analisis_detallado_mlp.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')

# 6. MÉTRICAS FINALES POR CONSOLA
print("\nREPORT DE CLASIFICACIÓN FINAL:")
print(classification_report(y_test, preds))

print("\n" + "=" * 90)
print(f"ANÁLISIS COMPLETADO. Imagen guardada en: {plot_path}")
print("=" * 90)