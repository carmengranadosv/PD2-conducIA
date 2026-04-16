import json
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd
from keras import layers, models
from sklearn.preprocessing import LabelEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data/processed/tlc_clean/problema4"
SAVE_DIR = BASE_DIR / "despliegue/modelos_finales"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILES = {
    "train": DATA_DIR / "train_p4.parquet",
    "val": DATA_DIR / "val_p4.parquet",
    "test": DATA_DIR / "test_p4.parquet",
}

COL_ZONA = "origen_id"
TARGET = "velocidad_mph"
COLS_CLIMA_TEMP = [
    "temp_c",
    "precipitation",
    "viento_kmh",
    "lluvia",
    "nieve",
    "hay_lluvia",
    "hay_nieve",
    "es_festivo",
    "num_eventos",
    "dia_semana",
    "es_fin_semana",
    "hora_sen",
    "hora_cos",
]

CONFIG = {
    "embedding_dim": 10,
    "dense_layers": [128, 64, 32],
    "dropout": 0.2,
    "epochs": 15,
    "batch_size": 4096,
    "optimizer": "adam",
    "loss": "mae",
}


def build_model(num_unique_zones: int, num_features: int) -> keras.Model:
    input_zone = layers.Input(shape=(1,), name="input_zona")
    embed_zone = layers.Embedding(
        input_dim=num_unique_zones,
        output_dim=CONFIG["embedding_dim"],
        name="embedding_zona",
    )(input_zone)
    embed_zone = layers.Flatten()(embed_zone)

    input_num = layers.Input(shape=(num_features,), name="input_clima")
    combined = layers.Concatenate()([embed_zone, input_num])

    x = layers.Dense(CONFIG["dense_layers"][0], activation="relu")(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG["dropout"])(x)

    for units in CONFIG["dense_layers"][1:]:
        x = layers.Dense(units, activation="relu")(x)

    output = layers.Dense(1, name="prediccion_velocidad")(x)

    model = models.Model(inputs=[input_zone, input_num], outputs=output)
    model.compile(
        optimizer=CONFIG["optimizer"],
        loss=CONFIG["loss"],
        metrics=["mse"],
    )
    return model


def reentrenar_p4_completo():
    print("=" * 60)
    print("REENTRENAMIENTO FINAL - PROBLEMA 4 (RED NEURONAL)")
    print("=" * 60)
    print(f"Buscando datos en: {DATA_DIR}")

    missing_files = [str(path) for path in DATA_FILES.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Faltan archivos procesados de P4. Genera primero los datos con "
            "`uv run python src/modelos/problema4/preprocesamiento_base.py`. "
            f"Archivos ausentes: {missing_files}"
        )

    np.random.seed(42)
    keras.utils.set_random_seed(42)

    print("\nCargando train + val + test...")
    df_full = pd.concat(
        [pd.read_parquet(path) for path in DATA_FILES.values()],
        ignore_index=True,
    )
    print(f"Dataset unificado con exito: {len(df_full):,} filas.")

    required_cols = [COL_ZONA, TARGET] + COLS_CLIMA_TEMP
    missing_cols = [col for col in required_cols if col not in df_full.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas esperadas en los datos de P4: {missing_cols}")

    encoder = LabelEncoder()
    scaler = StandardScaler()

    print("\nPreparando entradas...")
    X_zone = encoder.fit_transform(df_full[COL_ZONA]).astype(np.int32)
    X_num = scaler.fit_transform(df_full[COLS_CLIMA_TEMP]).astype(np.float32)
    y = df_full[TARGET].values.astype(np.float32)

    num_unique_zones = len(encoder.classes_)

    print("\nEntrenando red neuronal con embeddings...")
    model = build_model(num_unique_zones=num_unique_zones, num_features=len(COLS_CLIMA_TEMP))
    model.fit(
        [X_zone, X_num],
        y,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        verbose=1,
    )

    model_path = SAVE_DIR / "modelo_p4_red_neuronal.keras"
    encoder_path = SAVE_DIR / "modelo_p4_label_encoder_zonas.joblib"
    scaler_path = SAVE_DIR / "modelo_p4_scaler_clima.joblib"
    metadata_path = SAVE_DIR / "modelo_p4_red_neuronal_metadata.json"

    model.save(model_path)
    joblib.dump(encoder, encoder_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "modelo": "NN_With_Embeddings_P4",
        "target": TARGET,
        "zona_col": COL_ZONA,
        "numeric_features": COLS_CLIMA_TEMP,
        "config": CONFIG,
        "train_samples": int(len(df_full)),
        "num_unique_zones": int(num_unique_zones),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nMODELO FINAL GUARDADO EN: {model_path}")
    print(f"ENCODER FINAL GUARDADO EN: {encoder_path}")
    print(f"SCALER FINAL GUARDADO EN: {scaler_path}")
    print(f"METADATA FINAL GUARDADA EN: {metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    reentrenar_p4_completo()
