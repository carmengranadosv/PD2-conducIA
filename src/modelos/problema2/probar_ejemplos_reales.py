# src/modelos/problema2/probar_ejemplos_reales.py

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

print("=" * 88)
print("PRUEBA DE EJEMPLOS REALES - MLP (PROBLEMA 2)")
print("=" * 88)

# 1. CONFIGURACIÓN DE RUTAS
DATA_DIR = Path("data/processed/tlc_clean/problema2/features")
MODEL_DIR = Path("models/problema2")
OUTPUT_DIR = Path("reports/problema2/ejemplos_reales")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target"
# Columnas para visualización en tablas
DISPLAY_COLS = [
    "ventana_inicio",
    "origen_id",
    "target",
    "pred_mlp",
    "prob_mlp",
    "tasa_historica",
    "demanda_p1",
    "espera_media",
    "oferta_inferida",
    "hora",
    "es_festivo"
]

def load_assets():
    """Carga de datos y modelos entrenados."""
    print("Cargando datos de test y modelos...")
    df_test = pd.read_parquet(DATA_DIR / "test.parquet")
    with open(DATA_DIR / "metadata.json", "r") as f:
        meta = json.load(f)
    
    scaler = joblib.load(MODEL_DIR / "mlp_scaler.pkl")
    le = joblib.load(MODEL_DIR / "zona_encoder.pkl")
    
    # Importación local de keras para evitar logs innecesarios al inicio
    import keras
    model = keras.models.load_model(MODEL_DIR / "mlp_model.keras")
    
    return df_test, meta, scaler, le, model

def run_inference(df, meta, scaler, le, model):
    """Ejecuta la predicción y prepara baselines."""
    out = df.copy()
    
    # 1. Preparar features (debe coincidir con el entrenamiento)
    FEATURES_NUM = [f for f in meta['feature_cols'] if f != 'origen_id']
    # Codificar origen_id usando el LabelEncoder cargado
    out['zona_enc'] = out['origen_id'].apply(lambda z: le.transform([z])[0] if z in le.classes_ else -1)
    FEATURES_FINAL = FEATURES_NUM + ['zona_enc']
    
    # 2. Escalar y Predecir
    X_sc = scaler.transform(out[FEATURES_FINAL].fillna(0))
    probs = model.predict(X_sc, batch_size=2048, verbose=0).flatten()
    
    out["prob_mlp"] = probs
    out["pred_mlp"] = (probs >= 0.5).astype(int)
    
    # 3. Baseline de Negocio: Usar la tasa histórica (Umbral TOP = 0.33)
    out["pred_baseline"] = (out["tasa_historica"] >= 0.33).astype(int)
    
    # 4. Flags de Error para análisis
    out["acierto_mlp"] = out["pred_mlp"] == out[TARGET_COL]
    out["tipo_analisis"] = "Acierto"
    out.loc[(out["pred_mlp"] == 1) & (out[TARGET_COL] == 0), "tipo_analisis"] = "Falso Positivo"
    out.loc[(out["pred_mlp"] == 0) & (out[TARGET_COL] == 1), "tipo_analisis"] = "Falso Negativo"
    
    return out

def main():
    # Cargar y Predecir
    df_test, meta, scaler, le, model = load_assets()
    df_res = run_inference(df_test, meta, scaler, le, model)

    # --- MÉTRICAS GLOBALES ---
    print("\n" + "=" * 88)
    print("MÉTRICAS COMPARATIVAS: MLP vs BASELINE HISTÓRICO")
    print("=" * 88)
    
    f1_mlp = f1_score(df_res[TARGET_COL], df_res["pred_mlp"])
    f1_base = f1_score(df_res[TARGET_COL], df_res["pred_baseline"])
    
    print(f"F1-Score MLP:        {f1_mlp:.4f}")
    print(f"F1-Score Histórico:  {f1_base:.4f}")
    print(f"Mejora respecto al histórico: {((f1_mlp - f1_base)/f1_base)*100:.2f}%")
    print("-" * 88)
    print("Informe Detallado MLP:")
    print(classification_report(df_res[TARGET_COL], df_res["pred_mlp"]))

    # --- EXTRACCIÓN DE EJEMPLOS ---
    
    # A. Aciertos con máxima confianza
    best_cases = df_res[df_res["acierto_mlp"]].sort_values("prob_mlp", ascending=False).head(10)
    
    # B. Falsos Negativos (Zonas TOP que el modelo no vió)
    missed_top = df_res[df_res["tipo_analisis"] == "Falso Negativo"].sort_values("prob_mlp", ascending=True).head(10)
    
    # C. Falsos Positivos (El modelo envió al taxi a una zona mediocre)
    false_alarms = df_res[df_res["tipo_analisis"] == "Falso Positivo"].sort_values("prob_mlp", ascending=False).head(10)

    # D. Análisis por Zona (Las 10 más difíciles)
    worst_zones = (
        df_res.groupby("origen_id")
        .agg(
            total_muestras=("target", "count"),
            precision_media=("acierto_mlp", "mean"),
            tasa_real_exito=("target", "mean")
        )
        .sort_values("precision_media")
        .head(10)
        .reset_index()
    )

    # --- IMPRIMIR TABLAS ---
    print("\n" + "=" * 88)
    print("EJEMPLOS: ACIERTOS DE ALTA CONFIANZA (>90% PROB)")
    print("=" * 88)
    print(best_cases[DISPLAY_COLS].to_string(index=False))

    print("\n" + "=" * 88)
    print("EJEMPLOS: FALSOS NEGATIVOS (Oportunidades Perdidas)")
    print("=" * 88)
    print(missed_top[DISPLAY_COLS].to_string(index=False))

    print("\n" + "=" * 88)
    print("ZONAS DONDE EL MODELO TIENE MÁS DIFICULTAD")
    print("=" * 88)
    print(worst_zones.to_string(index=False))

    # --- GUARDAR RESULTADOS ---
    best_cases.to_csv(OUTPUT_DIR / "aciertos_seguros_mlp.csv", index=False)
    missed_top.to_csv(OUTPUT_DIR / "falsos_negativos_mlp.csv", index=False)
    false_alarms.to_csv(OUTPUT_DIR / "falsos_positivos_mlp.csv", index=False)
    worst_zones.to_csv(OUTPUT_DIR / "zonas_dificiles_mlp.csv", index=False)

    print("\n" + "=" * 88)
    print(f"ANÁLISIS COMPLETADO. Tablas guardadas en: {OUTPUT_DIR}")
    print("=" * 88)

if __name__ == "__main__":
    main()