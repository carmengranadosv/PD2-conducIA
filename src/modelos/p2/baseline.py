# src/modelos/problema2/baseline.py
import pandas as pd
import numpy as np
import json, joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, classification_report,
                              average_precision_score, confusion_matrix)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

FEATURES_DIR = Path('data/processed/tlc_clean/problema2/features')
MODELS_DIR   = Path('models/problema2');    MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR    = Path('reports/problema2/plots');      PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR  = Path('reports/problema2/resultados'); RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print(" BASELINE — REGRESIÓN LOGÍSTICA (Problema 2)")
print("="*70)

# ── 1. Cargar ─────────────────────────────────────────────────────────────────
df_train = pd.read_parquet(FEATURES_DIR / 'train.parquet')
df_val   = pd.read_parquet(FEATURES_DIR / 'val.parquet')
df_test  = pd.read_parquet(FEATURES_DIR / 'test.parquet')

with open(FEATURES_DIR / 'metadata.json') as f:
    meta = json.load(f)

FEATURES = meta['feature_cols']
# Excluir origen_id (string) — lo encodificamos
FEATURES_NUM = [f for f in FEATURES if f != 'origen_id']

TARGET = 'target'

print(f"  Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")
print(f"  Balance target — positivos: {df_train[TARGET].mean():.1%}")

# Encoding zona
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df_train['origen_id'])
for df in [df_train, df_val, df_test]:
    df['zona_enc'] = df['origen_id'].apply(
        lambda z: le.transform([z])[0] if z in le.classes_ else -1)

FEATURES_FINAL = FEATURES_NUM + ['zona_enc']

X_train = df_train[FEATURES_FINAL].fillna(0)
X_val   = df_val[FEATURES_FINAL].fillna(0)
X_test  = df_test[FEATURES_FINAL].fillna(0)
y_train = df_train[TARGET]
y_val   = df_val[TARGET]
y_test  = df_test[TARGET]

# ── 2. Normalizar ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

# ── 3. Entrenar ───────────────────────────────────────────────────────────────
print("\nEntrenando Regresión Logística...")
lr = LogisticRegression(
    class_weight='balanced',
    max_iter=500,
    random_state=42,
    C=1.0,
)
lr.fit(X_train_sc, y_train)

# ── 4. Evaluar ────────────────────────────────────────────────────────────────
def metrics(y_true, y_prob, y_pred, label=""):
    auc  = float(roc_auc_score(y_true, y_prob))
    ap   = float(average_precision_score(y_true, y_prob))
    rep  = classification_report(y_true, y_pred, output_dict=True)
    f1   = float(rep['weighted avg']['f1-score'])
    if label:
        print(f"  {label}: AUC={auc:.4f}  AP={ap:.4f}  F1={f1:.4f}")
    return {'auc': auc, 'ap': ap, 'f1': f1}

prob_val  = lr.predict_proba(X_val_sc)[:,1]
prob_test = lr.predict_proba(X_test_sc)[:,1]
pred_val  = lr.predict(X_val_sc)
pred_test = lr.predict(X_test_sc)

print("\nRESULTADOS:")
m_val  = metrics(y_val,  prob_val,  pred_val,  "Val ")
m_test = metrics(y_test, prob_test, pred_test, "Test")

print("\nClassification Report (Test):")
print(classification_report(y_test, pred_test))

# ── 5. Guardar ────────────────────────────────────────────────────────────────
joblib.dump(lr,     MODELS_DIR / 'baseline_lr.pkl')
joblib.dump(scaler, MODELS_DIR / 'baseline_scaler.pkl')
joblib.dump(le,     MODELS_DIR / 'zona_encoder.pkl')

results = {
    'modelo': 'Regresión Logística',
    'features': FEATURES_FINAL,
    'val':  m_val,
    'test': m_test,
}
with open(RESULTS_DIR / 'baseline_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ── 6. Plots ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 4))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# Curva ROC
from sklearn.metrics import roc_curve
ax1 = fig.add_subplot(gs[0, 0])
fpr, tpr, _ = roc_curve(y_test, prob_test)
ax1.plot(fpr, tpr, color='#2ecc71', lw=2, label=f'AUC={m_test["auc"]:.3f}')
ax1.plot([0,1],[0,1], 'k--', alpha=0.4)
ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')
ax1.set_title('Curva ROC — Test', fontweight='bold')
ax1.legend(); ax1.grid(alpha=0.3)

# Matriz de confusión
ax2 = fig.add_subplot(gs[0, 1])
cm = confusion_matrix(y_test, pred_test)
im = ax2.imshow(cm, cmap='Greens')
ax2.set_xticks([0,1]); ax2.set_yticks([0,1])
ax2.set_xticklabels(['Pred 0','Pred 1'])
ax2.set_yticklabels(['Real 0','Real 1'])
for i in range(2):
    for j in range(2):
        ax2.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                 fontsize=12, fontweight='bold',
                 color='white' if cm[i,j] > cm.max()/2 else 'black')
ax2.set_title('Matriz de Confusión', fontweight='bold')

# Coeficientes top
ax3 = fig.add_subplot(gs[0, 2])
coefs = pd.Series(lr.coef_[0], index=FEATURES_FINAL).abs().nlargest(10)
ax3.barh(coefs.index[::-1], coefs.values[::-1], color='#3498db', edgecolor='black')
ax3.set_xlabel('|Coeficiente|')
ax3.set_title('Top 10 Features', fontweight='bold')
ax3.grid(alpha=0.3, axis='x')

plt.suptitle('Baseline Regresión Logística — Problema 2', fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'baseline_resultados.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nPlot:       {PLOTS_DIR / 'baseline_resultados.png'}")
print(f"Resultados: {RESULTS_DIR / 'baseline_results.json'}")