"""
Prueba el Random Forest del Problema 1 con ejemplos reales del test set.

Objetivo:
- Ver metricas globales.
- Comparar el modelo contra los baselines fila a fila.
- Enseñar casos concretos donde acierta y donde falla.
- Guardar tablas CSV para poder revisarlas con calma.

Ejecutar:
    uv run python src/modelos/problema1/probar_random_forest_reales.py
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_DIR = Path("data/processed/tlc_clean/problema1/features")
MODEL_CANDIDATES = [
    Path("models/problema1/baseline_random_forest.pkl"),
    Path("reports/problema1/resultados/baseline_random_forest.pkl"),
]
OUTPUT_DIR = Path("reports/problema1/ejemplos_random_forest")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target"
DISPLAY_COLS = [
    "timestamp_hora",
    "origen_id",
    "target",
    "pred_rf",
    "pred_naive",
    "pred_media_hist",
    "error_rf",
    "error_naive",
    "error_media_hist",
    "demanda",
    "lag_1h",
    "lag_24h",
    "roll_mean_3h",
    "roll_mean_24h",
    "media_hist",
    "hora",
    "dia_semana",
    "es_finde",
    "lluvia",
    "nieve",
    "es_festivo",
    "num_eventos",
]


def pick_model_path() -> Path:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No se encontro el modelo Random Forest. Ejecuta baseline.py primero."
    )


def calc_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mape": float(mape),
    }


def add_prediction_columns(df: pd.DataFrame, feature_cols: list[str], model) -> pd.DataFrame:
    out = df.copy()
    out["pred_rf"] = np.clip(model.predict(out[feature_cols]), 0, None)
    out["pred_naive"] = out["demanda"]
    out["pred_media_hist"] = out["media_hist"]

    out["error_rf"] = (out[TARGET_COL] - out["pred_rf"]).abs()
    out["error_naive"] = (out[TARGET_COL] - out["pred_naive"]).abs()
    out["error_media_hist"] = (out[TARGET_COL] - out["pred_media_hist"]).abs()
    out["mejora_vs_naive"] = out["error_naive"] - out["error_rf"]
    out["mejora_vs_media_hist"] = out["error_media_hist"] - out["error_rf"]
    out["ratio_rf_real"] = out["pred_rf"] / np.clip(out[TARGET_COL], 1, None)

    return out


def print_metric_table(df: pd.DataFrame) -> None:
    metrics = {
        "RandomForest": calc_metrics(df[TARGET_COL], df["pred_rf"]),
        "Naive": calc_metrics(df[TARGET_COL], df["pred_naive"]),
        "Media_hist": calc_metrics(df[TARGET_COL], df["pred_media_hist"]),
    }

    print("\n" + "=" * 88)
    print("RESUMEN GLOBAL EN TEST")
    print("=" * 88)
    print(f"{'Modelo':<16} {'RMSE':>10} {'MAE':>10} {'R2':>10} {'MAPE':>10}")
    print("-" * 88)
    for name, vals in metrics.items():
        print(
            f"{name:<16} {vals['rmse']:>10.2f} {vals['mae']:>10.2f} "
            f"{vals['r2']:>10.3f} {vals['mape']:>9.1f}%"
        )

    rf = metrics["RandomForest"]
    naive = metrics["Naive"]
    mh = metrics["Media_hist"]

    print("\nLectura rapida:")
    print(
        f"- El RF baja el MAE frente a Naive en "
        f"{(naive['mae'] - rf['mae']) / naive['mae'] * 100:.1f}%."
    )
    print(
        f"- El RF baja el MAE frente a Media historica en "
        f"{(mh['mae'] - rf['mae']) / mh['mae'] * 100:.1f}%."
    )


def print_behavior_summary(df: pd.DataFrame) -> None:
    rf_best = (
        (df["error_rf"] < df["error_naive"]) & (df["error_rf"] < df["error_media_hist"])
    ).mean()
    rf_worst = (
        (df["error_rf"] > df["error_naive"]) & (df["error_rf"] > df["error_media_hist"])
    ).mean()
    almost_exact = (df["error_rf"] <= 3).mean()
    hard_spikes = (df[TARGET_COL] >= df[TARGET_COL].quantile(0.95)).mean()

    print("\n" + "=" * 88)
    print("COMPORTAMIENTO DEL MODELO")
    print("=" * 88)
    print(f"- RF es el mejor de los 3 en el {rf_best * 100:.1f}% de los casos.")
    print(f"- RF es el peor de los 3 en el {rf_worst * 100:.1f}% de los casos.")
    print(f"- RF queda a 3 viajes o menos del valor real en el {almost_exact * 100:.1f}% de los casos.")
    print(f"- El {hard_spikes * 100:.1f}% de las filas son picos de demanda (percentil 95 o superior).")


def print_examples(title: str, df: pd.DataFrame, cols: list[str]) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)
    print(df[cols].to_string(index=False))


def save_examples(name: str, df: pd.DataFrame) -> None:
    df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)


def main() -> None:
    print("=" * 88)
    print("PRUEBA REAL DEL RANDOM FOREST - PROBLEMA 1")
    print("=" * 88)

    model_path = pick_model_path()
    with open(DATA_DIR / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    df_test = pd.read_parquet(DATA_DIR / "test.parquet")
    model = joblib.load(model_path)

    df_test = add_prediction_columns(df_test, feature_cols, model)

    print(f"\nModelo cargado: {model_path}")
    print(f"Filas de test: {len(df_test):,}")
    print(
        f"Periodo test: {df_test['timestamp_hora'].min()} -> "
        f"{df_test['timestamp_hora'].max()}"
    )
    print(f"Numero de features: {len(feature_cols)}")

    print_metric_table(df_test)
    print_behavior_summary(df_test)

    best_examples = (
        df_test[
            (df_test["mejora_vs_naive"] > 15) & (df_test["mejora_vs_media_hist"] > 15)
        ]
        .sort_values(["error_rf", "target"], ascending=[True, False])
        .head(10)
    )
    worst_examples = df_test.sort_values("error_rf", ascending=False).head(10)
    peak_examples = (
        df_test[df_test[TARGET_COL] >= df_test[TARGET_COL].quantile(0.99)]
        .sort_values("error_rf")
        .head(10)
    )
    underpred_examples = (
        df_test.assign(sesgo=df_test[TARGET_COL] - df_test["pred_rf"])
        .sort_values("sesgo", ascending=False)
        .head(10)
    )
    overpred_examples = (
        df_test.assign(sesgo=df_test["pred_rf"] - df_test[TARGET_COL])
        .sort_values("sesgo", ascending=False)
        .head(10)
    )
    by_zone = (
        df_test.groupby("origen_id")
        .agg(
            n_casos=("origen_id", "size"),
            demanda_media_real=(TARGET_COL, "mean"),
            pred_media_rf=("pred_rf", "mean"),
            error_medio_rf=("error_rf", "mean"),
        )
        .sort_values("error_medio_rf", ascending=False)
        .head(10)
        .reset_index()
    )

    print_examples(
        "10 CASOS DONDE EL RANDOM FOREST GANA CLARAMENTE A LOS BASELINES",
        best_examples,
        DISPLAY_COLS,
    )
    print_examples(
        "10 CASOS DONDE EL RANDOM FOREST FALLA MAS",
        worst_examples,
        DISPLAY_COLS,
    )
    print_examples(
        "10 PICOS REALES DE DEMANDA (TOP 1%)",
        peak_examples,
        DISPLAY_COLS,
    )
    print_examples(
        "10 CASOS DONDE EL RF SE QUEDA CORTO",
        underpred_examples,
        DISPLAY_COLS + ["sesgo"],
    )
    print_examples(
        "10 CASOS DONDE EL RF SOBREESTIMA MAS",
        overpred_examples,
        DISPLAY_COLS + ["sesgo"],
    )

    print("\n" + "=" * 88)
    print("ZONAS DONDE MAS LE CUESTA AL MODELO")
    print("=" * 88)
    print(by_zone.to_string(index=False))

    feature_importance = (
        pd.DataFrame(
            {"feature": feature_cols, "importance": model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .head(15)
    )

    print("\n" + "=" * 88)
    print("TOP 15 FEATURES MAS IMPORTANTES")
    print("=" * 88)
    print(feature_importance.to_string(index=False))

    save_examples("casos_buenos_rf", best_examples)
    save_examples("casos_malos_rf", worst_examples)
    save_examples("picos_demanda_rf", peak_examples)
    save_examples("infraestimaciones_rf", underpred_examples)
    save_examples("sobreestimaciones_rf", overpred_examples)
    by_zone.to_csv(OUTPUT_DIR / "zonas_mas_dificiles.csv", index=False)
    feature_importance.to_csv(OUTPUT_DIR / "feature_importance_top15.csv", index=False)

    summary = {
        "modelo_path": str(model_path),
        "test_rows": int(len(df_test)),
        "metricas": {
            "RandomForest": calc_metrics(df_test[TARGET_COL], df_test["pred_rf"]),
            "Naive": calc_metrics(df_test[TARGET_COL], df_test["pred_naive"]),
            "Media_hist": calc_metrics(df_test[TARGET_COL], df_test["pred_media_hist"]),
        },
        "porcentajes": {
            "rf_mejor_que_ambos": float(
                (
                    (df_test["error_rf"] < df_test["error_naive"])
                    & (df_test["error_rf"] < df_test["error_media_hist"])
                ).mean()
            ),
            "rf_peor_que_ambos": float(
                (
                    (df_test["error_rf"] > df_test["error_naive"])
                    & (df_test["error_rf"] > df_test["error_media_hist"])
                ).mean()
            ),
            "error_menor_o_igual_3": float((df_test["error_rf"] <= 3).mean()),
        },
        "archivos_generados": [
            "casos_buenos_rf.csv",
            "casos_malos_rf.csv",
            "picos_demanda_rf.csv",
            "infraestimaciones_rf.csv",
            "sobreestimaciones_rf.csv",
            "zonas_mas_dificiles.csv",
            "feature_importance_top15.csv",
        ],
    }

    with open(OUTPUT_DIR / "resumen_random_forest_real.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nArchivos guardados en:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
