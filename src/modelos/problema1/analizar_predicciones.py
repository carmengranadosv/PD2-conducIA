# src/modelos/problema1/analizar_predicciones.py

"""
Analisis detallado de predicciones del Random Forest.

Incluye:
- resumen global y comparativa contra baselines simples,
- ejemplos zona por zona,
- ventanas consecutivas de 24 horas para entender la serie temporal,
- casos buenos y malos con visualizaciones faciles de leer.
"""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("=" * 90)
print("ANALISIS DETALLADO DE PREDICCIONES - RANDOM FOREST")
print("=" * 90)

sns.set_theme(style="whitegrid")
np.random.seed(42)

DATA_DIR = Path("data/processed/tlc_clean/problema1/features")
MODEL_DIR = Path("models/problema1")
OUTPUT_DIR = Path("reports/problema1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df_test = pd.read_parquet(DATA_DIR / "test.parquet")
rf = joblib.load(MODEL_DIR / "baseline_random_forest.pkl")

with open(DATA_DIR / "metadata.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

FEATURE_COLS = meta["feature_cols"]
TARGET = meta["target"]


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


def print_metric_block(metrics: dict[str, dict[str, float]]) -> None:
    print("\n" + "=" * 90)
    print("RESUMEN GLOBAL")
    print("=" * 90)
    print(f"{'Modelo':<16} {'RMSE':>10} {'MAE':>10} {'R2':>10} {'MAPE':>10}")
    print("-" * 90)
    for nombre, vals in metrics.items():
        print(
            f"{nombre:<16} {vals['rmse']:>10.2f} {vals['mae']:>10.2f} "
            f"{vals['r2']:>10.3f} {vals['mape']:>9.1f}%"
        )


def pick_consecutive_window(df_zone: pd.DataFrame, window_size: int = 24) -> pd.DataFrame:
    df_zone = df_zone.sort_values("timestamp_hora").copy()
    diffs = df_zone["timestamp_hora"].diff().eq(pd.Timedelta(hours=1))
    run_id = diffs.ne(True).cumsum()
    run_lengths = df_zone.groupby(run_id).size()
    valid_runs = run_lengths[run_lengths >= window_size].index

    if len(valid_runs) > 0:
        run = valid_runs[0]
        return df_zone[run_id == run].head(window_size).copy()

    return df_zone.head(window_size).copy()


def describe_window(label: str, df_window: pd.DataFrame) -> None:
    print(f"\n{'-' * 90}")
    print(f"VENTANA REAL DE 24H - {label}")
    print(f"{'-' * 90}")
    print(
        f"{'Hora':<20} {'Real':>8} {'RF':>8} {'Naive':>8} {'M.Hist':>8} "
        f"{'Error RF':>10} {'Eventos':>8}"
    )
    print("-" * 90)

    for _, row in df_window.iterrows():
        print(
            f"{row['timestamp_hora'].strftime('%Y-%m-%d %H:%M'):<20} "
            f"{row[TARGET]:>8.1f} {row['pred_rf']:>8.1f} {row['pred_naive']:>8.1f} "
            f"{row['pred_media_hist']:>8.1f} {row['error_rf']:>10.1f} "
            f"{row['num_eventos']:>8.0f}"
        )


X_test = df_test[FEATURE_COLS]
df_test["pred_rf"] = np.clip(rf.predict(X_test), 0, None)
df_test["pred_naive"] = df_test["demanda"]
df_test["pred_media_hist"] = df_test["media_hist"]
df_test["error_rf"] = (df_test[TARGET] - df_test["pred_rf"]).abs()
df_test["error_naive"] = (df_test[TARGET] - df_test["pred_naive"]).abs()
df_test["error_media_hist"] = (df_test[TARGET] - df_test["pred_media_hist"]).abs()
df_test["mejora_rf_vs_naive"] = df_test["error_naive"] - df_test["error_rf"]
df_test["mejora_rf_vs_media_hist"] = df_test["error_media_hist"] - df_test["error_rf"]
df_test["error_signed_rf"] = df_test[TARGET] - df_test["pred_rf"]

y_test = df_test[TARGET]

print(f"\nDatos cargados: {len(df_test):,} registros")
print(f"Periodo: {df_test['timestamp_hora'].min()} -> {df_test['timestamp_hora'].max()}")
print(f"Features: {len(FEATURE_COLS)}")

metrics = {
    "RandomForest": calc_metrics(y_test, df_test["pred_rf"]),
    "Naive": calc_metrics(y_test, df_test["pred_naive"]),
    "Media_hist": calc_metrics(y_test, df_test["pred_media_hist"]),
}
print_metric_block(metrics)

# ============================================================
# ANALISIS 1: ZONAS MAS ACTIVAS
# ============================================================

print("\n" + "=" * 90)
print("TOP 10 ZONAS MAS ACTIVAS")
print("=" * 90)

zonas_top = (
    df_test.groupby("origen_id")
    .agg(
        total_viajes=("demanda", "sum"),
        demanda_promedio_real=(TARGET, "mean"),
        pred_promedio_rf=("pred_rf", "mean"),
        error_medio_rf=("error_rf", "mean"),
    )
    .sort_values("total_viajes", ascending=False)
    .head(10)
)
print(zonas_top.round(2).to_string())

# ============================================================
# ANALISIS 2: EJEMPLOS ZONA POR ZONA
# ============================================================

print("\n" + "=" * 90)
print("EJEMPLOS DETALLADOS POR ZONA")
print("=" * 90)

demanda_por_zona = df_test.groupby("origen_id")["demanda"].sum().sort_values(ascending=False)
zona_alta = int(demanda_por_zona.index[0])
zona_media = int((demanda_por_zona - demanda_por_zona.quantile(0.5)).abs().idxmin())
zona_baja = int(demanda_por_zona.index[-1])

zonas_ejemplo = {
    "Alta demanda": zona_alta,
    "Demanda media": zona_media,
    "Baja demanda": zona_baja,
}

for nombre, zona_id in zonas_ejemplo.items():
    print(f"\n{'-' * 90}")
    print(f"{nombre.upper()} - Zona {zona_id}")
    print(f"{'-' * 90}")
    df_zona = df_test[df_test["origen_id"] == zona_id].sort_values("timestamp_hora").head(24)
    cols_zona = [
        "timestamp_hora",
        TARGET,
        "pred_rf",
        "error_rf",
        "lag_1h",
        "lag_24h",
        "media_hist",
        "num_eventos",
    ]
    df_print = df_zona[cols_zona].copy()
    numeric_cols = [c for c in cols_zona if c != "timestamp_hora"]
    df_print[numeric_cols] = df_print[numeric_cols].round(2)
    print(df_print.to_string(index=False))

# ============================================================
# ANALISIS 3: VENTANAS REALES DE 24 HORAS
# ============================================================

print("\n" + "=" * 90)
print("VENTANAS TEMPORALES DE 24 HORAS")
print("=" * 90)

for nombre, zona_id in zonas_ejemplo.items():
    df_window = pick_consecutive_window(df_test[df_test["origen_id"] == zona_id], 24)
    describe_window(f"{nombre} (Zona {zona_id})", df_window)

# ============================================================
# ANALISIS 4: PATRONES TEMPORALES
# ============================================================

print("\n" + "=" * 90)
print("DEMANDA POR HORA DEL DIA")
print("=" * 90)

demanda_hora = df_test.groupby("hora").agg({TARGET: "mean", "pred_rf": "mean"})
print(demanda_hora.round(2).to_string())

print("\n" + "=" * 90)
print("DEMANDA POR DIA DE LA SEMANA")
print("=" * 90)

dias_nombres = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
demanda_dia = df_test.groupby("dia_semana").agg({TARGET: "mean", "pred_rf": "mean"})
demanda_dia["dia"] = [dias_nombres[i] for i in demanda_dia.index]
print(demanda_dia[["dia", TARGET, "pred_rf"]].round(2).to_string(index=False))

# ============================================================
# ANALISIS 5: CASOS BUENOS Y MALOS
# ============================================================

print("\n" + "=" * 90)
print("CASOS BUENOS Y MALOS")
print("=" * 90)

casos_buenos = (
    df_test[
        (df_test["mejora_rf_vs_naive"] > 20)
        & (df_test["mejora_rf_vs_media_hist"] > 20)
    ]
    .sort_values(["error_rf", TARGET], ascending=[True, False])
    .head(10)
)

casos_malos = df_test.sort_values("error_rf", ascending=False).head(10)

print("\nMejores casos del RF:")
cols_buenos = [
    "timestamp_hora",
    "origen_id",
    TARGET,
    "pred_rf",
    "pred_naive",
    "pred_media_hist",
    "error_rf",
    "mejora_rf_vs_naive",
    "mejora_rf_vs_media_hist",
]
casos_buenos_print = casos_buenos[cols_buenos].copy()
casos_buenos_print[[c for c in cols_buenos if c != "timestamp_hora" and c != "origen_id"]] = (
    casos_buenos_print[[c for c in cols_buenos if c != "timestamp_hora" and c != "origen_id"]].round(2)
)
print(casos_buenos_print.to_string(index=False))

print("\nPeores casos del RF:")
cols_malos = [
    "timestamp_hora",
    "origen_id",
    TARGET,
    "pred_rf",
    "pred_naive",
    "pred_media_hist",
    "error_rf",
    "error_naive",
    "error_media_hist",
]
casos_malos_print = casos_malos[cols_malos].copy()
casos_malos_print[[c for c in cols_malos if c != "timestamp_hora" and c != "origen_id"]] = (
    casos_malos_print[[c for c in cols_malos if c != "timestamp_hora" and c != "origen_id"]].round(2)
)
print(casos_malos_print.to_string(index=False))

# ============================================================
# VISUALIZACION 1: RESUMEN GENERAL
# ============================================================

print("\n" + "=" * 90)
print("GENERANDO VISUALIZACIONES")
print("=" * 90)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.28)

ax1 = fig.add_subplot(gs[0, 0])
df_alta = pick_consecutive_window(df_test[df_test["origen_id"] == zona_alta], 48)
ax1.plot(df_alta["timestamp_hora"], df_alta[TARGET], label="Real", color="#1f2d3d", linewidth=2)
ax1.plot(df_alta["timestamp_hora"], df_alta["pred_rf"], label="RF", color="#2a9d8f", linewidth=1.8)
ax1.set_title(f"Zona {zona_alta} - Alta demanda", fontweight="bold")
ax1.tick_params(axis="x", rotation=45)
ax1.legend()

ax2 = fig.add_subplot(gs[0, 1])
df_media = pick_consecutive_window(df_test[df_test["origen_id"] == zona_media], 48)
ax2.plot(df_media["timestamp_hora"], df_media[TARGET], label="Real", color="#1f2d3d", linewidth=2)
ax2.plot(df_media["timestamp_hora"], df_media["pred_rf"], label="RF", color="#e9c46a", linewidth=1.8)
ax2.set_title(f"Zona {zona_media} - Demanda media", fontweight="bold")
ax2.tick_params(axis="x", rotation=45)
ax2.legend()

ax3 = fig.add_subplot(gs[0, 2])
df_baja = pick_consecutive_window(df_test[df_test["origen_id"] == zona_baja], 48)
ax3.plot(df_baja["timestamp_hora"], df_baja[TARGET], label="Real", color="#1f2d3d", linewidth=2)
ax3.plot(df_baja["timestamp_hora"], df_baja["pred_rf"], label="RF", color="#e76f51", linewidth=1.8)
ax3.set_title(f"Zona {zona_baja} - Baja demanda", fontweight="bold")
ax3.tick_params(axis="x", rotation=45)
ax3.legend()

ax4 = fig.add_subplot(gs[1, 0])
sample_idx = np.random.choice(len(df_test), min(5000, len(df_test)), replace=False)
sample_df = df_test.iloc[sample_idx]
ax4.scatter(sample_df[TARGET], sample_df["pred_rf"], alpha=0.28, s=12, color="#3a86ff")
max_val = max(df_test[TARGET].max(), df_test["pred_rf"].max())
ax4.plot([0, max_val], [0, max_val], "--", color="#d62828", linewidth=2)
ax4.set_xlabel("Demanda real")
ax4.set_ylabel("Prediccion RF")
ax4.set_title("Real vs predicho", fontweight="bold")

ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(df_test["error_signed_rf"], bins=60, color="#6c5ce7", alpha=0.75, edgecolor="black")
ax5.axvline(0, color="#d62828", linestyle="--", linewidth=2)
ax5.set_title("Distribucion del error", fontweight="bold")
ax5.set_xlabel("Real - Predicho")

ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(demanda_hora.index, demanda_hora[TARGET], "o-", label="Real", color="#264653", linewidth=2)
ax6.plot(demanda_hora.index, demanda_hora["pred_rf"], "s-", label="RF", color="#2a9d8f", linewidth=2)
ax6.set_title("Patron diario", fontweight="bold")
ax6.set_xlabel("Hora")
ax6.set_ylabel("Demanda media")
ax6.set_xticks(range(0, 24, 3))
ax6.legend()

ax7 = fig.add_subplot(gs[2, 0])
dias_completos = pd.DataFrame({"dia_semana": range(7), "dia_nombre": dias_nombres})
demanda_dia_completo = dias_completos.merge(demanda_dia.reset_index(), on="dia_semana", how="left").fillna(0)
x = np.arange(7)
width = 0.35
ax7.bar(x - width / 2, demanda_dia_completo[TARGET], width, label="Real", color="#457b9d", edgecolor="black")
ax7.bar(x + width / 2, demanda_dia_completo["pred_rf"], width, label="RF", color="#2a9d8f", edgecolor="black")
ax7.set_xticks(x)
ax7.set_xticklabels(["L", "M", "X", "J", "V", "S", "D"])
ax7.set_title("Patron semanal", fontweight="bold")
ax7.legend()

ax8 = fig.add_subplot(gs[2, 1])
errores_zona = (
    df_test.groupby("origen_id")["error_rf"]
    .mean()
    .sort_values(ascending=False)
    .head(15)
)
ax8.barh(range(len(errores_zona)), errores_zona.values, color="#ef476f", edgecolor="black")
ax8.set_yticks(range(len(errores_zona)))
ax8.set_yticklabels([f"Zona {z}" for z in errores_zona.index], fontsize=8)
ax8.set_title("Zonas con mayor error medio", fontweight="bold")
ax8.set_xlabel("MAE medio")

ax9 = fig.add_subplot(gs[2, 2])
metric_names = list(metrics.keys())
mae_values = [metrics[m]["mae"] for m in metric_names]
bars = ax9.bar(metric_names, mae_values, color=["#2a9d8f", "#f4a261", "#e76f51"], edgecolor="black")
ax9.set_title("MAE por modelo", fontweight="bold")
for bar, val in zip(bars, mae_values):
    ax9.text(bar.get_x() + bar.get_width() / 2, val + 0.2, f"{val:.2f}", ha="center", fontsize=9)

fig.suptitle("Analisis general de predicciones - Random Forest", fontsize=15, fontweight="bold", y=0.995)
fig.subplots_adjust(top=0.92)
summary_plot_path = OUTPUT_DIR / "analisis_detallado.png"
fig.savefig(summary_plot_path, dpi=150, bbox_inches="tight")
print(f"Visualizacion guardada: {summary_plot_path}")
plt.close(fig)

# ============================================================
# VISUALIZACION 2: CASOS BUENOS Y MALOS + 24H
# ============================================================

fig2 = plt.figure(figsize=(20, 12))
gs2 = fig2.add_gridspec(2, 2, hspace=0.30, wspace=0.20)

ax10 = fig2.add_subplot(gs2[0, 0])
best_case = casos_buenos.iloc[0]
best_zone = int(best_case["origen_id"])
best_start = best_case["timestamp_hora"]
best_window = (
    df_test[
        (df_test["origen_id"] == best_zone)
        & (df_test["timestamp_hora"] >= best_start - pd.Timedelta(hours=12))
        & (df_test["timestamp_hora"] <= best_start + pd.Timedelta(hours=11))
    ]
    .sort_values("timestamp_hora")
    .head(24)
)
ax10.plot(best_window["timestamp_hora"], best_window[TARGET], label="Real", color="#1d3557", linewidth=2.2)
ax10.plot(best_window["timestamp_hora"], best_window["pred_rf"], label="RF", color="#2a9d8f", linewidth=2)
ax10.plot(best_window["timestamp_hora"], best_window["pred_naive"], label="Naive", color="#f4a261", linewidth=1.5, alpha=0.9)
ax10.plot(best_window["timestamp_hora"], best_window["pred_media_hist"], label="Media hist", color="#b56576", linewidth=1.5, alpha=0.9)
ax10.axvline(best_case["timestamp_hora"], color="#d62828", linestyle="--", linewidth=2)
ax10.set_title(f"Mejor caso - Zona {best_zone} alrededor del acierto", fontweight="bold")
ax10.tick_params(axis="x", rotation=45)
ax10.legend()

ax11 = fig2.add_subplot(gs2[0, 1])
worst_case = casos_malos.iloc[0]
worst_zone = int(worst_case["origen_id"])
worst_start = worst_case["timestamp_hora"]
worst_window = (
    df_test[
        (df_test["origen_id"] == worst_zone)
        & (df_test["timestamp_hora"] >= worst_start - pd.Timedelta(hours=12))
        & (df_test["timestamp_hora"] <= worst_start + pd.Timedelta(hours=11))
    ]
    .sort_values("timestamp_hora")
    .head(24)
)
ax11.plot(worst_window["timestamp_hora"], worst_window[TARGET], label="Real", color="#1d3557", linewidth=2.2)
ax11.plot(worst_window["timestamp_hora"], worst_window["pred_rf"], label="RF", color="#e63946", linewidth=2)
ax11.plot(worst_window["timestamp_hora"], worst_window["pred_naive"], label="Naive", color="#f4a261", linewidth=1.5, alpha=0.9)
ax11.plot(worst_window["timestamp_hora"], worst_window["pred_media_hist"], label="Media hist", color="#6d597a", linewidth=1.5, alpha=0.9)
ax11.axvline(worst_case["timestamp_hora"], color="#d62828", linestyle="--", linewidth=2)
ax11.set_title(f"Peor caso - Zona {worst_zone} alrededor del fallo", fontweight="bold")
ax11.tick_params(axis="x", rotation=45)
ax11.legend()

ax12 = fig2.add_subplot(gs2[1, 0])
good_plot = casos_buenos.copy()
good_plot["label"] = good_plot["timestamp_hora"].dt.strftime("%m-%d %Hh") + " | Z" + good_plot["origen_id"].astype(str)
ax12.barh(good_plot["label"], good_plot["mejora_rf_vs_naive"], color="#2a9d8f", edgecolor="black")
ax12.set_title("Casos donde RF mejora mucho a Naive", fontweight="bold")
ax12.set_xlabel("Viajes de mejora absoluta")

ax13 = fig2.add_subplot(gs2[1, 1])
bad_plot = casos_malos.copy()
bad_plot["label"] = bad_plot["timestamp_hora"].dt.strftime("%m-%d %Hh") + " | Z" + bad_plot["origen_id"].astype(str)
ax13.barh(bad_plot["label"], bad_plot["error_rf"], color="#e63946", edgecolor="black")
ax13.set_title("Peores errores absolutos del RF", fontweight="bold")
ax13.set_xlabel("Error absoluto")

fig2.suptitle("Casos reales buenos y malos - Random Forest", fontsize=15, fontweight="bold", y=0.995)
fig2.subplots_adjust(top=0.92)
cases_plot_path = OUTPUT_DIR / "analisis_casos_buenos_malos.png"
fig2.savefig(cases_plot_path, dpi=150, bbox_inches="tight")
print(f"Visualizacion guardada: {cases_plot_path}")
plt.close(fig2)

print("\n" + "=" * 90)
print("ANALISIS COMPLETADO")
print("=" * 90)
