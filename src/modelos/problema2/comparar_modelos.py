# src/modelos/problema2/comparar_modelos.py
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

RESULTS_DIR = Path('reports/problema2/resultados')
PLOTS_DIR   = Path('reports/problema2/plots')
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

modelos = {}
for nombre, path in [
    ('Logística',   RESULTS_DIR / 'baseline_results.json'),
    ('MLP',         RESULTS_DIR / 'mlp_results.json'),
    ('GNN',         RESULTS_DIR / 'gnn_results.json'),
]:
    if path.exists():
        with open(path) as f:
            d = json.load(f)
        modelos[nombre] = {'val': d['val'], 'test': d['test']}

print("="*65)
print(f"{'Modelo':<12} {'AUC':>8} {'AP':>8} {'F1':>8}  [Test]")
print("-"*65)
for nombre, res in modelos.items():
    m = res['test']
    print(f"{nombre:<12} {m['auc']:>8.4f} {m['ap']:>8.4f} {m['f1']:>8.4f}")
print("="*65)

nombres   = list(modelos.keys())
auc_vals  = [modelos[m]['test']['auc'] for m in nombres]
ap_vals   = [modelos[m]['test']['ap']  for m in nombres]
f1_vals   = [modelos[m]['test']['f1']  for m in nombres]
COLORS    = ['#2ecc71', '#3498db', '#9b59b6'][:len(nombres)]

fig = plt.figure(figsize=(12, 4))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

for i, (titulo, vals) in enumerate([('AUC-ROC', auc_vals),
                                      ('Average Precision', ap_vals),
                                      ('F1 (weighted)', f1_vals)]):
    ax = fig.add_subplot(gs[0, i])
    bars = ax.bar(nombres, vals, color=COLORS, edgecolor='black', width=0.5)
    mejor = np.argmax(vals)
    bars[mejor].set_edgecolor('#2c3e50')
    bars[mejor].set_linewidth(2.5)
    ax.set_title(titulo, fontweight='bold')
    ax.set_ylim(max(0, min(vals) - 0.05), min(1.0, max(vals) + 0.05))
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(range(len(nombres)))
    ax.set_xticklabels(nombres, rotation=15, ha='right')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f'{v:.4f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Comparativa de Modelos — Problema 2 (Test Set)', fontweight='bold')
plt.savefig(PLOTS_DIR / 'comparativa_modelos.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nComparativa: {PLOTS_DIR / 'comparativa_modelos.png'}")