import os
os.environ["KERAS_BACKEND"] = "jax"

import json
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd
from keras import layers
from sklearn.preprocessing import LabelEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[2]

FEATURES_DIR = BASE_DIR / "data/processed/tlc_clean/problema2/features"
P1_METADATA_PATH = BASE_DIR / "data/processed/tlc_clean/problema1/features/metadata.json"
RF_P1_PATH = BASE_DIR / "despliegue/modelos_finales/modelo_p1_rf.joblib"
SAVE_DIR = BASE_DIR / "despliegue/modelos_finales"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "hidden_layers": [256, 128, 64],
    "dropout": 0.2,
    "epochs": 10,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "pos_weight": 0.5,
}


def build_p1_features(df: pd.DataFrame, oferta_media: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origen_id": df["origen_id"],
            "hora": df["hora"],
            "dia_semana": df["dia_semana"],
            "dia_mes": df["ventana_inicio"].dt.day,
            "mes_num": df["mes_num"],
            "es_finde": df["es_finde"],
            "demanda": df["oferta_inferida"],
            "lag_1h": df["oferta_inferida"],
            "lag_2h": df["oferta_inferida"],
            "lag_3h": df["oferta_inferida"],
            "lag_6h": df["oferta_inferida"],
            "lag_12h": df["oferta_inferida"],
            "lag_24h": df["tasa_historica"] * oferta_media,
            "roll_mean_3h": df["oferta_inferida"],
            "roll_std_3h": df["oferta_inferida"] * 0.1,
            "roll_mean_24h": df["oferta_inferida"],
            "roll_std_24h": df["oferta_inferida"] * 0.1,
            "media_hist": df["tasa_historica"] * oferta_media,
            "temp_c": df["temp_c"],
            "precipitation": df["precipitation"],
            "viento_kmh": df["viento_kmh"],
            "velocidad_mph": df["viento_kmh"] * 0.621,
            "lluvia": df["lluvia"],
            "nieve": df["nieve"],
            "es_festivo": df["es_festivo"],
            "num_eventos": df["num_eventos"],
        }
    )


def recalcular_demanda_p1(
    df: pd.DataFrame,
    rf_p1,
    p1_feature_cols: list[str],
    batch_size: int = 500_000,
) -> np.ndarray:
    pred = np.empty(len(df), dtype=np.float32)
    oferta_media = float(df["oferta_inferida"].mean())

    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        feats = build_p1_features(df.iloc[start:end], oferta_media)
        pred[start:end] = np.clip(rf_p1.predict(feats[p1_feature_cols]), 0, None)

    return pred


def build_model(input_dim: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for units in CONFIG["hidden_layers"]:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(CONFIG["dropout"])(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, output, name="MLP_P2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["AUC"],
    )
    return model


def reentrenar_p2_completo():
    print("=" * 60)
    print("REENTRENAMIENTO FINAL - PROBLEMA 2 (MLP)")
    print("=" * 60)
    print(f"Buscando datos en: {FEATURES_DIR}")

    required_files = ["train.parquet", "val.parquet", "test.parquet", "metadata.json"]
    missing_files = [name for name in required_files if not (FEATURES_DIR / name).exists()]
    if missing_files:
        raise FileNotFoundError(f"Faltan archivos de features P2: {missing_files}")
    if not RF_P1_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro el RF final de P1 en {RF_P1_PATH}. "
            "Ejecuta primero despliegue/reentrenar/reentrenar_rf_p1.py."
        )
    if not P1_METADATA_PATH.exists():
        raise FileNotFoundError(f"No se encontro metadata de features P1 en {P1_METADATA_PATH}")

    np.random.seed(42)
    keras.utils.set_random_seed(42)

    print("\nCargando todos los conjuntos de datos...")
    df_train = pd.read_parquet(FEATURES_DIR / "train.parquet")
    df_val = pd.read_parquet(FEATURES_DIR / "val.parquet")
    df_test = pd.read_parquet(FEATURES_DIR / "test.parquet")
    for df in [df_train, df_val, df_test]:
        df["ventana_inicio"] = pd.to_datetime(df["ventana_inicio"])

    with open(FEATURES_DIR / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(P1_METADATA_PATH, "r", encoding="utf-8") as f:
        meta_p1 = json.load(f)

    target = meta["target"]
    features_num = [feature for feature in meta["feature_cols"] if feature != "origen_id"]
    features_final = features_num + ["zona_enc"]

    required_p2_cols = [col for col in meta["feature_cols"] if col != "demanda_p1"] + [target]
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        missing_cols = [col for col in required_p2_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Faltan columnas esperadas en P2 {name}: {missing_cols}")

    print(f"Cargando RF final de P1: {RF_P1_PATH}")
    rf_p1 = joblib.load(RF_P1_PATH)
    p1_feature_cols = meta_p1["feature_cols"]

    print("\nRecalculando demanda_p1 con el RF final reentrenado...")
    df_train["demanda_p1"] = recalcular_demanda_p1(df_train, rf_p1, p1_feature_cols)
    df_val["demanda_p1"] = recalcular_demanda_p1(df_val, rf_p1, p1_feature_cols)
    df_test["demanda_p1"] = recalcular_demanda_p1(df_test, rf_p1, p1_feature_cols)
    print(f"Demanda P1 recalculada - media train: {df_train['demanda_p1'].mean():.2f}")

    df_full = pd.concat([df_train, df_val, df_test], ignore_index=True)
    print(f"Dataset unificado con exito: {len(df_full):,} filas.")

    print("\nCodificando zonas...")
    encoder = LabelEncoder()
    df_full["zona_enc"] = encoder.fit_transform(df_full["origen_id"])

    X = df_full[features_final].fillna(0).values.astype(np.float32)
    y = df_full[target].values.astype(np.float32)

    print("Normalizando features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    print("\nEntrenando MLP...")
    model = build_model(X.shape[1])
    model.fit(
        X,
        y,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        class_weight={0: CONFIG["pos_weight"], 1: 1.0},
        verbose=1,
    )

    model_path = SAVE_DIR / "modelo_p2_mlp.keras"
    scaler_path = SAVE_DIR / "modelo_p2_mlp_scaler.pkl"
    encoder_path = SAVE_DIR / "modelo_p2_zona_encoder.pkl"
    metadata_path = SAVE_DIR / "modelo_p2_mlp_metadata.json"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(encoder, encoder_path)

    metadata = {
        "modelo": "MLP_P2",
        "features": features_final,
        "target": target,
        "config": CONFIG,
        "train_samples": int(len(df_full)),
        "input_dim": int(X.shape[1]),
        "cascading_input": {
            "feature": "demanda_p1",
            "rf_p1_path": str(RF_P1_PATH),
            "rf_p1_features": p1_feature_cols,
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nMODELO FINAL GUARDADO EN: {model_path}")
    print(f"SCALER FINAL GUARDADO EN: {scaler_path}")
    print(f"ENCODER FINAL GUARDADO EN: {encoder_path}")
    print(f"METADATA FINAL GUARDADA EN: {metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    reentrenar_p2_completo()
