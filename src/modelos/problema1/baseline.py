# src/modelos/problema1/baseline.py

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# CREAR DIRECTORIOS
# ============================================================

PLOTS_DIR = Path('reports/problema1/plots')
RESULTS_DIR = Path('reports/problema1/resultados')

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

plot_path = PLOTS_DIR / 'baseline_resultados.png'
rf_path = RESULTS_DIR / 'baseline_random_forest.pkl'
results_path = RESULTS_DIR / 'baseline_results.json'
fi_path = RESULTS_DIR / 'feature_importance.csv'

print("=" * 80)
print("BASELINES - PREDICCIÓN DE DEMANDA POR ZONA (Problema 1)")
print("=" * 80)

# ============================================================
# PASO 1: CARGAR FEATURES
# ============================================================

print("\nCargando features...")

DATA_DIR = Path('data/processed/tlc_clean/problema1/features')

df_train = pd.read_parquet(DATA_DIR / 'train.parquet')
df_val   = pd.read_parquet(DATA_DIR / 'val.parquet')
df_test  = pd.read_parquet(DATA_DIR / 'test.parquet')

with open(DATA_DIR / 'metadata.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)

FEATURE_COLS = meta['feature_cols']
TARGET = meta['target']

print(f"   Train: {len(df_train):>8,} filas | {meta['split']['train']['fecha_inicio']} → {meta['split']['train']['fecha_fin']}")
print(f"   Val:   {len(df_val):>8,} filas | {meta['split']['val']['fecha_inicio']} → {meta['split']['val']['fecha_fin']}")
print(f"   Test:  {len(df_test):>8,} filas | {meta['split']['test']['fecha_inicio']} → {meta['split']['test']['fecha_fin']}")
print(f"   Features: {len(FEATURE_COLS)}")

X_train, y_train = df_train[FEATURE_COLS], df_train[TARGET]
X_val, y_val     = df_val[FEATURE_COLS], df_val[TARGET]
X_test, y_test   = df_test[FEATURE_COLS], df_test[TARGET]

print(f"\nEstadísticas Target (demanda siguiente hora):")
print(f"   Media:   {y_test.mean():.2f} viajes/hora")
print(f"   Mediana: {y_test.median():.2f}")
print(f"   Std:     {y_test.std():.2f}")
print(f"   Min:     {y_test.min():.0f}")
print(f"   Max:     {y_test.max():.0f}")
print(f"   P25:     {y_test.quantile(0.25):.2f}")
print(f"   P75:     {y_test.quantile(0.75):.2f}")

# ============================================================
# HELPER: MÉTRICAS
# ============================================================

def calc_metrics(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100

    if label:
        print(f"   {label:<8} RMSE={rmse:6.2f}  MAE={mae:6.2f}  R²={r2:6.3f}  MAPE={mape:5.1f}%")

    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape)
    }

results = {}

# ============================================================
# BASELINE 1: PERSISTENCIA NAIVE
# ============================================================

print(f"\n{'='*80}")
print("BASELINE 1 — PERSISTENCIA: demanda(t+1) = demanda(t)")
print(f"{'='*80}")

results['Naive'] = {
    'val': calc_metrics(y_val, df_val['demanda'].values, "Val"),
    'test': calc_metrics(y_test, df_test['demanda'].values, "Test"),
    'pred_test': df_test['demanda'].values,
}

# ============================================================
# BASELINE 2: MEDIA HISTÓRICA
# ============================================================

print(f"\n{'='*80}")
print("BASELINE 2 — MEDIA HISTÓRICA (zona × hora × día_semana)")
print(f"{'='*80}")

pred_mh_val = df_val['media_hist'].values
pred_mh_test = df_test['media_hist'].values

results['Media_hist'] = {
    'val': calc_metrics(y_val, pred_mh_val, "Val"),
    'test': calc_metrics(y_test, pred_mh_test, "Test"),
    'pred_test': pred_mh_test,
}

# ============================================================
# BASELINE 3: RANDOM FOREST
# ============================================================

print(f"\n{'='*80}")
print("BASELINE 3 — RANDOM FOREST")
print(f"{'='*80}")

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
)

print("Entrenando Random Forest...")
rf.fit(X_train, y_train)
print("Entrenamiento completado")

pred_rf_val = np.clip(rf.predict(X_val), 0, None)
pred_rf_test = np.clip(rf.predict(X_test), 0, None)

results['RandomForest'] = {
    'val': calc_metrics(y_val, pred_rf_val, "Val"),
    'test': calc_metrics(y_test, pred_rf_test, "Test"),
    'pred_test': pred_rf_test,
}

joblib.dump(rf, rf_path)
print(f"   Modelo guardado: {rf_path}")

# ============================================================
# RESUMEN
# ============================================================

print(f"\n{'='*80}")
print("RESUMEN COMPARATIVO — TEST SET")
print(f"{'='*80}")
print(f"{'Modelo':<20} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'MAPE':>8}")
print("-" * 60)

for name, res in results.items():
    m = res['test']
    print(f"{name:<20} {m['rmse']:>8.2f} {m['mae']:>8.2f} {m['r2']:>8.3f} {m['mape']:>7.1f}%")

rf_rmse = results['RandomForest']['test']['rmse']
naive_rmse = results['Naive']['test']['rmse']
mh_rmse = results['Media_hist']['test']['rmse']

print(f"\nMejoras:")
print(f"   RF vs Naive:      {(naive_rmse - rf_rmse) / naive_rmse * 100:+.1f}% RMSE")
print(f"   RF vs Media hist: {(mh_rmse - rf_rmse) / mh_rmse * 100:+.1f}% RMSE")

# ============================================================
# GUARDAR RESULTADOS
# ============================================================

results_summary = {
    'models': {
        nombre: {
            'val': res['val'],
            'test': res['test']
        }
        for nombre, res in results.items()
    },
    'data_info': {
        'train_samples': len(df_train),
        'val_samples': len(df_val),
        'test_samples': len(df_test),
        'num_features': len(FEATURE_COLS),
        'target_mean': float(y_test.mean()),
        'target_std': float(y_test.std()),
    }
}

with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, indent=2)

print(f"\nResultados guardados: {results_path}")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print(f"\nFeature Importance — Top 15:")
fi = (
    pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': rf.feature_importances_
    })
    .sort_values('importance', ascending=False)
)

print(fi.head(15).to_string(index=False))

fi.to_csv(fi_path, index=False)
print(f"   Feature importance guardado: {fi_path}")

# ============================================================
# VISUALIZACIONES
# ============================================================

print("\nGenerando visualizaciones...")

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

# 1. Comparativa RMSE / MAE
ax1 = fig.add_subplot(gs[0, 0])
model_names = list(results.keys())
rmse_vals = [results[m]['test']['rmse'] for m in model_names]
mae_vals = [results[m]['test']['mae'] for m in model_names]
x = np.arange(len(model_names))
w = 0.35

b1 = ax1.bar(x - w/2, rmse_vals, w, label='RMSE', edgecolor='black')
b2 = ax1.bar(x + w/2, mae_vals, w, label='MAE', edgecolor='black')

ax1.set_xticks(x)
ax1.set_xticklabels(model_names, fontsize=10)
ax1.set_ylabel('Viajes/hora', fontsize=11)
ax1.set_title('Error por Modelo — Test Set', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

for bar in list(b1) + list(b2):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height + 0.3,
             f'{height:.1f}', ha='center', va='bottom', fontsize=9)

# 2. R² por modelo
ax2 = fig.add_subplot(gs[0, 1])
r2_vals = [results[m]['test']['r2'] for m in model_names]
bars = ax2.bar(model_names, r2_vals, edgecolor='black', width=0.6)

ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
ax2.set_ylabel('R² Score', fontsize=11)
ax2.set_title('R² por Modelo — Test Set', fontsize=12, fontweight='bold')
ax2.set_ylim(min(min(r2_vals) - 0.1, -0.05), 1.05)
ax2.grid(True, alpha=0.3, axis='y')

for bar, v in zip(bars, r2_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3. Predicho vs Real — zona más activa
ax3 = fig.add_subplot(gs[1, 0])

top_zone = df_test.groupby('origen_id')[TARGET].sum().idxmax()
mask = (df_test['origen_id'] == top_zone).values
sample = min(200, mask.sum())

ax3.plot(y_test.values[mask][:sample], label='Real', lw=2, alpha=0.9)
ax3.plot(pred_rf_test[mask][:sample], label='RF', lw=1.5, alpha=0.8)
ax3.plot(pred_mh_test[mask][:sample], label='Media hist', lw=1.2, alpha=0.7, linestyle='-.')
ax3.plot(df_test['demanda'].values[mask][:sample], label='Naive', lw=1, alpha=0.6, linestyle='--')

ax3.set_xlabel('Timesteps (horas)', fontsize=11)
ax3.set_ylabel('Viajes/hora', fontsize=11)
ax3.set_title(f'Predicho vs Real — Zona {top_zone} (más activa)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9, loc='best')
ax3.grid(True, alpha=0.3)

# 4. Feature Importance top 12
ax4 = fig.add_subplot(gs[1, 1])
fi_top = fi.head(12)
ax4.barh(fi_top['feature'].iloc[::-1], fi_top['importance'].iloc[::-1], edgecolor='black', linewidth=0.5)
ax4.set_xlabel('Importancia', fontsize=11)
ax4.set_title('Feature Importance — Top 12', fontsize=12, fontweight='bold')
ax4.tick_params(axis='y', labelsize=9)
ax4.grid(True, alpha=0.3, axis='x')

plt.suptitle('Baselines - Predicción de Demanda por Zona (Problema 1)',
             fontsize=14, fontweight='bold', y=0.995)

plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Visualización guardada: {plot_path}")

plt.show()

# ============================================================
# RESUMEN FINAL
# ============================================================

print("\n" + "=" * 80)
print("BASELINE COMPLETADO")
print("=" * 80)

print(f"\nArchivos generados:")
print(f"   Modelo:     {rf_path}")
print(f"   Resultados: {results_path}")
print(f"   Features:   {fi_path}")
print(f"   Plot:       {plot_path}")

best_model = min(results.items(), key=lambda x: x[1]['test']['rmse'])

print(f"\nMejores resultados (Test):")
print(f"   Modelo: {best_model[0]}")
print(f"   RMSE:   {best_model[1]['test']['rmse']:.2f}")
print(f"   MAE:    {best_model[1]['test']['mae']:.2f}")
print(f"   R²:     {best_model[1]['test']['r2']:.3f}")

print("\n" + "=" * 80)