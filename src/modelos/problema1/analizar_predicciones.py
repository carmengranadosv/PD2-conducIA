# src/modelos/problema1/analizar_predicciones.py

"""
Análisis detallado de predicciones: ejemplos zona por zona
Compara predicciones vs realidad con visualizaciones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json

print("="*80)
print("📊 ANÁLISIS DETALLADO DE PREDICCIONES")
print("="*80)

# ============================================================
# CONFIGURACIÓN
# ============================================================

# Cargar datos test
DATA_DIR = Path('data/processed/tlc_clean/problema1/features')
MODEL_DIR = Path('models/problema1')

df_test = pd.read_parquet(DATA_DIR / 'test.parquet')

# Cargar modelo Random Forest
rf = joblib.load(MODEL_DIR / 'baseline_random_forest.pkl')

with open(DATA_DIR / 'metadata.json') as f:
    meta = json.load(f)

FEATURE_COLS = meta['feature_cols']
TARGET = meta['target']

# Hacer predicciones
X_test = df_test[FEATURE_COLS]
y_test = df_test[TARGET]
y_pred_rf = np.clip(rf.predict(X_test), 0, None)

df_test['pred_rf'] = y_pred_rf

print(f"\n✅ Datos cargados: {len(df_test):,} registros")
print(f"   Período: {df_test['timestamp_hora'].min()} → {df_test['timestamp_hora'].max()}")

# ============================================================
# ANÁLISIS 1: ZONAS MÁS ACTIVAS
# ============================================================

print("\n" + "="*80)
print("📍 TOP 10 ZONAS MÁS ACTIVAS")
print("="*80)

zonas_top = (
    df_test.groupby('origen_id')
    .agg({
        'demanda': 'sum',
        'target': 'mean',
        'pred_rf': 'mean'
    })
    .sort_values('demanda', ascending=False)
    .head(10)
)

print("\n┌─────────┬──────────────┬─────────────┬─────────────┬─────────┐")
print("│  Zona   │ Total Viajes │ Demanda Avg │ Pred Avg RF │  Error  │")
print("├─────────┼──────────────┼─────────────┼─────────────┼─────────┤")

for zona_id, row in zonas_top.iterrows():
    total = row['demanda']
    real_avg = row['target']
    pred_avg = row['pred_rf']
    error = abs(real_avg - pred_avg)
    print(f"│ {zona_id:>7} │ {total:>12,.0f} │ {real_avg:>11.2f} │ {pred_avg:>11.2f} │ {error:>7.2f} │")

print("└─────────┴──────────────┴─────────────┴─────────────┴─────────┘")

# ============================================================
# ANÁLISIS 2: EJEMPLOS ZONA POR ZONA
# ============================================================

print("\n" + "="*80)
print("🔍 EJEMPLOS DETALLADOS POR ZONA")
print("="*80)

# Seleccionar 3 zonas: alta demanda, media, baja
zona_alta = df_test.groupby('origen_id')['demanda'].sum().idxmax()
zona_media = df_test.groupby('origen_id')['demanda'].sum().quantile(0.5)
zona_media = (df_test.groupby('origen_id')['demanda'].sum() - zona_media).abs().idxmin()
zona_baja = df_test.groupby('origen_id')['demanda'].sum().idxmin()

zonas_ejemplo = {
    'Alta demanda': zona_alta,
    'Demanda media': zona_media,
    'Baja demanda': zona_baja
}

for nombre, zona_id in zonas_ejemplo.items():
    print(f"\n{'─'*80}")
    print(f"📍 {nombre.upper()}: Zona {zona_id}")
    print(f"{'─'*80}")
    
    df_zona = df_test[df_test['origen_id'] == zona_id].sort_values('timestamp_hora').head(24)
    
    print(f"\n{'Hora':<20} {'Real':>8} {'Pred RF':>8} {'Error':>8} {'Lag_1h':>8} {'Media_hist':>11}")
    print("─" * 72)
    
    for _, row in df_zona.iterrows():
        hora = row['timestamp_hora'].strftime('%Y-%m-%d %H:%M')
        real = row['target']
        pred = row['pred_rf']
        error = abs(real - pred)
        lag1 = row['lag_1h']
        mhist = row['media_hist']
        
        print(f"{hora:<20} {real:>8.1f} {pred:>8.1f} {error:>8.1f} {lag1:>8.1f} {mhist:>11.1f}")

# ============================================================
# ANÁLISIS 3: PATRONES TEMPORALES
# ============================================================

print("\n" + "="*80)
print("⏰ DEMANDA POR HORA DEL DÍA")
print("="*80)

demanda_hora = df_test.groupby('hora').agg({
    'target': 'mean',
    'pred_rf': 'mean'
})

print("\n┌──────┬──────────────┬──────────────┬─────────┐")
print("│ Hora │ Demanda Real │  Pred RF     │  Error  │")
print("├──────┼──────────────┼──────────────┼─────────┤")

for hora, row in demanda_hora.iterrows():
    real = row['target']
    pred = row['pred_rf']
    error = abs(real - pred)
    print(f"│ {hora:>4} │ {real:>12.2f} │ {pred:>12.2f} │ {error:>7.2f} │")

print("└──────┴──────────────┴──────────────┴─────────┘")

# ============================================================
# ANÁLISIS 4: DÍA DE LA SEMANA
# ============================================================

print("\n" + "="*80)
print("📅 DEMANDA POR DÍA DE LA SEMANA")
print("="*80)

dias_nombres = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

demanda_dia = df_test.groupby('dia_semana').agg({
    'target': 'mean',
    'pred_rf': 'mean'
})

print("\n┌────────────┬──────────────┬──────────────┬─────────┐")
print("│    Día     │ Demanda Real │  Pred RF     │  Error  │")
print("├────────────┼──────────────┼──────────────┼─────────┤")

for dia, row in demanda_dia.iterrows():
    nombre = dias_nombres[dia]
    real = row['target']
    pred = row['pred_rf']
    error = abs(real - pred)
    print(f"│ {nombre:<10} │ {real:>12.2f} │ {pred:>12.2f} │ {error:>7.2f} │")

print("└────────────┴──────────────┴──────────────┴─────────┘")

# ============================================================
# VISUALIZACIONES
# ============================================================

print("\n" + "="*80)
print("📊 GENERANDO VISUALIZACIONES DETALLADAS")
print("="*80)

fig = plt.figure(figsize=(20, 12))

# 1. Zona de alta demanda - Serie temporal
ax1 = plt.subplot(3, 3, 1)
df_alta = df_test[df_test['origen_id'] == zona_alta].sort_values('timestamp_hora').head(100)
ax1.plot(df_alta['target'].values, label='Real', color='#2c3e50', linewidth=2)
ax1.plot(df_alta['pred_rf'].values, label='RF Pred', color='#2ecc71', linewidth=1.5, alpha=0.8)
ax1.set_title(f'Zona {zona_alta} - Alta Demanda (100h)', fontsize=11, fontweight='bold')
ax1.set_xlabel('Horas')
ax1.set_ylabel('Viajes/hora')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Zona media - Serie temporal
ax2 = plt.subplot(3, 3, 2)
df_media = df_test[df_test['origen_id'] == zona_media].sort_values('timestamp_hora').head(100)
ax2.plot(df_media['target'].values, label='Real', color='#2c3e50', linewidth=2)
ax2.plot(df_media['pred_rf'].values, label='RF Pred', color='#f39c12', linewidth=1.5, alpha=0.8)
ax2.set_title(f'Zona {zona_media} - Demanda Media (100h)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Horas')
ax2.set_ylabel('Viajes/hora')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Zona baja - Serie temporal
ax3 = plt.subplot(3, 3, 3)
df_baja = df_test[df_test['origen_id'] == zona_baja].sort_values('timestamp_hora').head(100)
ax3.plot(df_baja['target'].values, label='Real', color='#2c3e50', linewidth=2)
ax3.plot(df_baja['pred_rf'].values, label='RF Pred', color='#e74c3c', linewidth=1.5, alpha=0.8)
ax3.set_title(f'Zona {zona_baja} - Baja Demanda (100h)', fontsize=11, fontweight='bold')
ax3.set_xlabel('Horas')
ax3.set_ylabel('Viajes/hora')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Scatter: Real vs Predicho
ax4 = plt.subplot(3, 3, 4)
sample_idx = np.random.choice(len(df_test), min(5000, len(df_test)), replace=False)
ax4.scatter(df_test.iloc[sample_idx]['target'], 
           df_test.iloc[sample_idx]['pred_rf'], 
           alpha=0.3, s=10, color='#3498db')
max_val = max(df_test['target'].max(), df_test['pred_rf'].max())
ax4.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfecto')
ax4.set_xlabel('Demanda Real')
ax4.set_ylabel('Demanda Predicha (RF)')
ax4.set_title('Real vs Predicho (5K puntos)', fontsize=11, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Distribución errores
ax5 = plt.subplot(3, 3, 5)
errores = df_test['target'] - df_test['pred_rf']
ax5.hist(errores, bins=50, color='#9b59b6', edgecolor='black', alpha=0.7)
ax5.axvline(0, color='red', linestyle='--', linewidth=2)
ax5.set_xlabel('Error (Real - Predicho)')
ax5.set_ylabel('Frecuencia')
ax5.set_title(f'Distribución Errores (Media: {errores.mean():.2f})', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. Demanda por hora del día
ax6 = plt.subplot(3, 3, 6)
ax6.plot(demanda_hora.index, demanda_hora['target'], 'o-', label='Real', 
         color='#2c3e50', linewidth=2, markersize=6)
ax6.plot(demanda_hora.index, demanda_hora['pred_rf'], 's-', label='RF Pred', 
         color='#2ecc71', linewidth=2, markersize=5)
ax6.set_xlabel('Hora del día')
ax6.set_ylabel('Demanda promedio')
ax6.set_title('Patrón Diario', fontsize=11, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.set_xticks(range(0, 24, 3))

# 7. Demanda por día semana - CORREGIDO
ax7 = plt.subplot(3, 3, 7)

# Crear array completo de 7 días (rellenar con NaN si falta alguno)
dias_completos = pd.DataFrame({
    'dia_semana': range(7),
    'dia_nombre': dias_nombres
})

demanda_dia_completo = dias_completos.merge(
    demanda_dia.reset_index(), 
    on='dia_semana', 
    how='left'
)

# Rellenar NaN con 0
demanda_dia_completo['target'] = demanda_dia_completo['target'].fillna(0)
demanda_dia_completo['pred_rf'] = demanda_dia_completo['pred_rf'].fillna(0)

x = np.arange(7)
width = 0.35
ax7.bar(x - width/2, demanda_dia_completo['target'], width, label='Real', color='#3498db', edgecolor='black')
ax7.bar(x + width/2, demanda_dia_completo['pred_rf'], width, label='RF Pred', color='#2ecc71', edgecolor='black')
ax7.set_xlabel('Día de la semana')
ax7.set_ylabel('Demanda promedio')
ax7.set_title('Patrón Semanal', fontsize=11, fontweight='bold')
ax7.set_xticks(x)
ax7.set_xticklabels(['L', 'M', 'X', 'J', 'V', 'S', 'D'])
ax7.legend()
ax7.grid(True, alpha=0.3, axis='y')

# 8. Errores por zona (top 20)
ax8 = plt.subplot(3, 3, 8)
errores_zona = df_test.groupby('origen_id').apply(
    lambda x: np.sqrt(((x['target'] - x['pred_rf'])**2).mean())
).sort_values(ascending=False).head(20)
ax8.barh(range(len(errores_zona)), errores_zona.values, color='#e74c3c', edgecolor='black')
ax8.set_yticks(range(len(errores_zona)))
ax8.set_yticklabels([f'Zona {z}' for z in errores_zona.index], fontsize=8)
ax8.set_xlabel('RMSE')
ax8.set_title('Top 20 Zonas con Mayor Error', fontsize=11, fontweight='bold')
ax8.grid(True, alpha=0.3, axis='x')

# 9. Boxplot errores por hora
ax9 = plt.subplot(3, 3, 9)
df_test['error'] = df_test['target'] - df_test['pred_rf']
datos_box = [df_test[df_test['hora'] == h]['error'].values for h in range(24)]
bp = ax9.boxplot(datos_box, positions=range(24), widths=0.6, patch_artist=True,
                 boxprops=dict(facecolor='#3498db', alpha=0.7),
                 medianprops=dict(color='red', linewidth=2))
ax9.axhline(0, color='green', linestyle='--', linewidth=1.5)
ax9.set_xlabel('Hora del día')
ax9.set_ylabel('Error (Real - Predicho)')
ax9.set_title('Distribución Error por Hora', fontsize=11, fontweight='bold')
ax9.grid(True, alpha=0.3, axis='y')
ax9.set_xticks(range(0, 24, 3))

plt.suptitle('Análisis Detallado de Predicciones - Random Forest', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

# Guardar
output_dir = Path('reports/problema1')
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'analisis_detallado.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Visualización guardada: {output_dir / 'analisis_detallado.png'}")

plt.show()

print("\n" + "="*80)
print("✅ ANÁLISIS COMPLETADO")
print("="*80)