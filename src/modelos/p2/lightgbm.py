import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    roc_curve,
)
import lightgbm as lgb

np.random.seed(42)

FEATURES_DIR = Path('data/processed/tlc_clean/problema2/features')
MODELS_DIR   = Path('models/problema2'); MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR    = Path('reports/problema2/plots'); PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR  = Path('reports/problema2/resultados'); RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 50,
    'reg_alpha': 0.0,
    'reg_lambda': 0.0,
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 50,
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
        lambda z: le.transform([z])[0] if z in le.classes_ else -1
    )

FEATURES_FINAL = FEATURES_NUM + ['zona_enc']

X_train = df_train[FEATURES_FINAL].fillna(0)
X_val   = df_val[FEATURES_FINAL].fillna(0)
X_test  = df_test[FEATURES_FINAL].fillna(0)

y_train = df_train[TARGET].values.astype(np.int32)
y_val   = df_val[TARGET].values.astype(np.int32)
y_test  = df_test[TARGET].values.astype(np.int32)

# Balance de clases
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = float(neg / pos)
print(f"  scale_pos_weight: {scale_pos_weight:.4f}")

# ── 3. Modelo LightGBM ────────────────────────────────────────────────────────
model = LGBMClassifier(
    objective='binary',
    metric='auc',
    n_estimators=CONFIG['n_estimators'],
    learning_rate=CONFIG['learning_rate'],
    num_leaves=CONFIG['num_leaves'],
    max_depth=CONFIG['max_depth'],
    subsample=CONFIG['subsample'],
    colsample_bytree=CONFIG['colsample_bytree'],
    min_child_samples=CONFIG['min_child_samples'],
    reg_alpha=CONFIG['reg_alpha'],
    reg_lambda=CONFIG['reg_lambda'],
    random_state=CONFIG['random_state'],
    n_jobs=CONFIG['n_jobs'],
    scale_pos_weight=scale_pos_weight,
    verbosity=-1,
)

print("\nEntrenando LightGBM...")
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_names=['train', 'val'],
    eval_metric='auc',
    callbacks=[
        lgb.early_stopping(CONFIG['early_stopping_rounds'], verbose=True),
        lgb.log_evaluation(period=50),
    ]
)

# ── 4. Evaluar ────────────────────────────────────────────────────────────────
def metrics(y_true, y_prob, label=""):
    y_pred = (y_prob >= 0.5).astype(int)
    auc = float(roc_auc_score(y_true, y_prob))
    ap  = float(average_precision_score(y_true, y_prob))
    rep = classification_report(y_true, y_pred, output_dict=True)
    f1  = float(rep['weighted avg']['f1-score'])
    if label:
        print(f"  {label}: AUC={auc:.4f}  AP={ap:.4f}  F1={f1:.4f}")
    return {'auc': auc, 'ap': ap, 'f1': f1}

prob_val  = model.predict_proba(X_val)[:, 1]
prob_test = model.predict_proba(X_test)[:, 1]

print("\nRESULTADOS:")
m_val  = metrics(y_val,  prob_val,  "Val ")
m_test = metrics(y_test, prob_test, "Test")

print("\nClassification Report (Test):")
print(classification_report(y_test, (prob_test >= 0.5).astype(int)))

# ── 5. Guardar ────────────────────────────────────────────────────────────────
joblib.dump(model, MODELS_DIR / 'lightgbm_model.pkl')

results = {
    'modelo': 'LightGBM Classifier',
    'nota': 'Incluye demanda_p1 como cascading input del Problema 1',
    'config': CONFIG,
    'features': FEATURES_FINAL,
    'best_iteration': int(model.best_iteration_) if model.best_iteration_ is not None else None,
    'val':  m_val,
    'test': m_test,
}
with open(RESULTS_DIR / 'lightgbm_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ── 6. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Curva AUC por iteración
evals_result = model.evals_result_
train_auc = evals_result['train']['auc']
val_auc   = evals_result['val']['auc']

axes[0].plot(train_auc, label='Train', color='#2c3e50')
axes[0].plot(val_auc,   label='Val',   color='#e74c3c')
axes[0].set_title('AUC por iteración', fontweight='bold')
axes[0].set_xlabel('Iteración')
axes[0].set_ylabel('AUC')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Importancia de variables
importances = pd.Series(model.feature_importances_, index=FEATURES_FINAL).sort_values(ascending=False).head(10)
axes[1].barh(importances.index[::-1], importances.values[::-1], color='#3498db')
axes[1].set_title('Top 10 features', fontweight='bold')
axes[1].set_xlabel('Importancia')
axes[1].grid(axis='x', alpha=0.3)

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, prob_test)
axes[2].plot(fpr, tpr, color='#27ae60', lw=2, label=f'AUC={m_test["auc"]:.3f}')
axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.4)
axes[2].set_xlabel('FPR')
axes[2].set_ylabel('TPR')
axes[2].set_title('Curva ROC — Test', fontweight='bold')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.suptitle('LightGBM — Optimización de Zona (Problema 2)', fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'lightgbm_resultados.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nModelo:     {MODELS_DIR / 'lightgbm_model.pkl'}")
print(f"Plot:       {PLOTS_DIR / 'lightgbm_resultados.png'}")
print(f"Resultados: {RESULTS_DIR / 'lightgbm_results.json'}")