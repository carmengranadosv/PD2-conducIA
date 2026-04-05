# src/modelos/problema1/comparar_modelos.py
"""
Lee todos los resultados de modelos del Problema 1 y genera
una tabla + gráfica comparativa unificada.
Ejecutar DESPUÉS de baseline.py, lstm.py (y transformer.py cuando esté).
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

RESULTS_DIR = Path('reports/problema1/resultados')
PLOTS_DIR   = Path('reports/problema1/plots')
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Cargar todos los JSON de resultados ───────────────────────────────────────
result_files = {
    'Naive':       RESULTS_DIR / 'baseline_results.json',
    'Media hist':  RESULTS_DIR / 'baseline_results.json',
    'RandomForest':RESULTS_DIR / 'baseline_results.json',
    'LSTM':        RESULTS_DIR / 'lstm_results.json',
    'Transformer': RESULTS_DIR / 'transformer_results.json',  # cuando exista
}

modelos = {}

# Baseline: tiene formato interno diferente (múltiples modelos en un JSON)
if (RESULTS_DIR / 'baseline_results.json').exists():
    with open(RESULTS_DIR / 'baseline_results.json') as f:
        bl = json.load(f)
    for nombre in ['Naive', 'Media_hist', 'RandomForest']:
        if nombre in bl.get('models', {}):
            modelos[nombre] = bl['models'][nombre]

# LSTM y Transformer: formato unificado
for nombre, path in [('LSTM', RESULTS_DIR / 'lstm_results.json'),
                     ('Transformer', RESULTS_DIR / 'transformer_results.json')]:
    if path.exists():
        with open(path) as f:
            d = json.load(f)
        modelos[nombre] = {'val': d['val'], 'test': d['test']}

if not modelos:
    print("No se encontraron resultados. Ejecuta primero baseline.py y lstm.py")
    exit()

# ── Tabla comparativa ─────────────────────────────────────────────────────────
print("="*70)
print(f"{'Modelo':<16} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'MAPE':>8}  [Test]")
print("-"*70)
for nombre, res in modelos.items():
    m = res['test']
    print(f"{nombre:<16} {m['rmse']:>8.3f} {m['mae']:>8.3f} {m['r2']:>8.4f} {m['mape']:>7.1f}%")
print("="*70)

# ── Gráfica comparativa ───────────────────────────────────────────────────────
nombres = list(modelos.keys())
rmse_vals = [modelos[m]['test']['rmse'] for m in nombres]
mae_vals  = [modelos[m]['test']['mae']  for m in nombres]
r2_vals   = [modelos[m]['test']['r2']   for m in nombres]
mape_vals = [modelos[m]['test']['mape'] for m in nombres]

COLORS = ['#95a5a6', '#bdc3c7', '#f39c12', '#2ecc71', '#3498db'][:len(nombres)]

fig = plt.figure(figsize=(16, 5))
gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

metricas = [
    ('RMSE (viajes/h)',  rmse_vals, True),   # True = menor es mejor
    ('MAE (viajes/h)',   mae_vals,  True),
    ('R²',              r2_vals,   False),   # False = mayor es mejor
    ('MAPE (%)',        mape_vals, True),
]

for i, (titulo, vals, menor_mejor) in enumerate(metricas):
    ax = fig.add_subplot(gs[0, i])
    bars = ax.bar(nombres, vals, color=COLORS, edgecolor='black', width=0.6)
    
    # Destacar el mejor
    mejor_idx = np.argmin(vals) if menor_mejor else np.argmax(vals)
    bars[mejor_idx].set_edgecolor('#2c3e50')
    bars[mejor_idx].set_linewidth(2)
    
    ax.set_title(titulo, fontweight='bold', fontsize=11)
    ax.set_xticklabels(nombres, rotation=30, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Comparativa de Modelos — Problema 1 (Test Set)', 
             fontsize=13, fontweight='bold')
plt.savefig(PLOTS_DIR / 'comparativa_modelos.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nComparativa guardada: {PLOTS_DIR / 'comparativa_modelos.png'}")