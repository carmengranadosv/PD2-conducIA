from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


KEYS = ["origen_id", "mes_num", "dia_semana", "hora"]

P2_NUMERIC_COLS = [
    "n_viajes",
    "espera_media",
    "hora_sen",
    "hora_cos",
    "temp_c",
    "precipitation",
    "viento_kmh",
    "lluvia",
    "nieve",
    "es_festivo",
    "num_eventos",
    "oferta_inferida",
    "tasa_historica",
    "demanda_p1",
]

P5_NUMERIC_COLS = [
    "num_pasajeros",
    "distancia",
    "duracion_min",
    "velocidad_mph",
    "precio_base",
    "precio_total_est",
    "espera_min",
    "temp_c",
    "precipitation",
    "viento_kmh",
    "lluvia",
    "nieve",
    "es_festivo",
    "num_eventos",
    "es_fin_semana",
    "hora_sen",
    "hora_cos",
    "rentabilidad_base_min",
    "trafico_denso",
]

P5_CATEGORICAL_COLS = [
    "tipo_vehiculo",
    "origen_zona",
    "origen_barrio",
    "evento_tipo",
    "franja_horaria",
]


def existing_columns(path: Path, columns: list[str]) -> list[str]:
    schema_cols = set(pq.ParquetFile(path).schema.names)
    return [col for col in columns if col in schema_cols]


def combine_partial_numeric(parts: list[pd.DataFrame], suffix: str) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts)
    combined = combined.groupby(level=list(range(len(KEYS))), sort=False).sum()
    combined.columns = [
        col.removesuffix(suffix)
        for col in combined.columns
    ]
    return combined


def mode_from_counts(parts: list[pd.DataFrame], col: str) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=KEYS + [col])

    counts = pd.concat(parts, ignore_index=True)
    counts = (
        counts.groupby(KEYS + [col], dropna=False, as_index=False)["_count"]
        .sum()
        .sort_values(KEYS + ["_count"], ascending=[True, True, True, True, False])
    )
    return counts.drop_duplicates(KEYS, keep="first")[KEYS + [col]]


def aggregate_context(
    input_path: Path,
    output_path: Path,
    numeric_cols: list[str],
    categorical_cols: list[str] | None = None,
    batch_size: int = 500_000,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el parquet de entrada: {input_path}")

    categorical_cols = categorical_cols or []
    numeric_cols = existing_columns(input_path, numeric_cols)
    categorical_cols = existing_columns(input_path, categorical_cols)
    read_cols = existing_columns(input_path, KEYS + numeric_cols + categorical_cols)

    missing_keys = [col for col in KEYS if col not in read_cols]
    if missing_keys:
        raise ValueError(f"Faltan columnas clave en {input_path}: {missing_keys}")

    sums_parts: list[pd.DataFrame] = []
    counts_parts: list[pd.DataFrame] = []
    row_count_parts: list[pd.DataFrame] = []
    category_count_parts: dict[str, list[pd.DataFrame]] = {
        col: [] for col in categorical_cols
    }

    parquet_file = pq.ParquetFile(input_path)
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size, columns=read_cols), start=1):
        df = batch.to_pandas()
        for col in KEYS:
            df[col] = df[col].astype("int32")

        group = df.groupby(KEYS, dropna=False, sort=False)
        if numeric_cols:
            sums_parts.append(group[numeric_cols].sum().add_suffix("__sum"))
            counts_parts.append(group[numeric_cols].count().add_suffix("__count"))

        row_count_parts.append(group.size().to_frame("_n_contexto"))

        for col in categorical_cols:
            cat_counts = (
                df[KEYS + [col]]
                .dropna(subset=[col])
                .groupby(KEYS + [col], dropna=False, as_index=False)
                .size()
                .rename(columns={"size": "_count"})
            )
            category_count_parts[col].append(cat_counts)

        print(f"{input_path.name}: procesado batch {i}")

    sums = combine_partial_numeric(sums_parts, "__sum")
    counts = combine_partial_numeric(counts_parts, "__count")
    row_counts = pd.concat(row_count_parts).groupby(level=list(range(len(KEYS))), sort=False).sum()

    if numeric_cols:
        context = sums.divide(counts.where(counts != 0))
    else:
        context = pd.DataFrame(index=row_counts.index)

    context = context.join(row_counts).reset_index()

    for col, parts in category_count_parts.items():
        mode_df = mode_from_counts(parts, col)
        context = context.merge(mode_df, on=KEYS, how="left")

    context = context.sort_values(KEYS).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    context.to_parquet(output_path, index=False)
    print(f"Guardado {output_path} con {len(context):,} filas.")
    return context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera contextos historicos agregados para el despliegue web."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Raiz local de datos. En Docker esta carpeta se monta como /app/data.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500_000,
        help="Filas por batch al leer Parquet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    out_dir = data_root / "processed" / "tlc_clean" / "contexto_web"

    aggregate_context(
        input_path=data_root / "processed" / "tlc_clean" / "problema2" / "features" / "train.parquet",
        output_path=out_dir / "contexto_p2.parquet",
        numeric_cols=P2_NUMERIC_COLS,
        batch_size=args.batch_size,
    )

    aggregate_context(
        input_path=data_root / "processed" / "tlc_clean" / "problema5" / "train.parquet",
        output_path=out_dir / "contexto_p5.parquet",
        numeric_cols=P5_NUMERIC_COLS,
        categorical_cols=P5_CATEGORICAL_COLS,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
