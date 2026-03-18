from pathlib import Path
import pandas as pd
import numpy as np


def print_dataset_stats(parquet_path: Path, sample_rows: int = 5) -> None:
    df = pd.read_parquet(parquet_path)

    print("\n" + "=" * 70)
    print(f" DATASET STATS: {parquet_path.name}")
    print("=" * 70)

    # --------------------------------------------------
    # 1️ Dimensiones básicas
    # --------------------------------------------------
    print("\n Dimensiones")
    print(f"   Filas: {len(df):,}")
    print(f"   Columnas: {len(df.columns)}")

    memoria_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"   Memoria: {memoria_mb:.2f} MB")

    # --------------------------------------------------
    # 2️ Tipos de datos
    # --------------------------------------------------
    print("\n Tipos de datos")
    print(df.dtypes.to_string().replace("\n", "\n   "))

    # --------------------------------------------------
    # 3️ Nulos
    # --------------------------------------------------
    print("\n Valores nulos")
    na = df.isna().sum()
    na_pct = (na / len(df) * 100).round(2)

    df_na = pd.DataFrame({
        "Nulos": na,
        "Porcentaje (%)": na_pct
    }).sort_values("Porcentaje (%)", ascending=False)

    df_na = df_na[df_na["Nulos"] > 0]

    if df_na.empty:
        print("    No hay valores nulos")
    else:
        print(df_na.to_string())

    # --------------------------------------------------
    # 4️ Numéricas
    # --------------------------------------------------
    num_cols = df.select_dtypes(include=["number"]).columns

    if len(num_cols) > 0:
        print("\n Estadísticas numéricas")
        desc = df[num_cols].describe().T
        desc["skew"] = df[num_cols].skew()
        desc["kurtosis"] = df[num_cols].kurtosis()
        print(desc.round(3).to_string())

        # Outliers simples (IQR)
        print("\n Outliers aproximados (IQR method)")
        for col in num_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            pct = outliers / len(df) * 100
            print(f"   {col:20s}: {outliers:,} ({pct:.2f}%)")

    # --------------------------------------------------
    # 5️ Categóricas
    # --------------------------------------------------
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    if len(cat_cols) > 0:
        print("\n🔹 Variables categóricas")

        for col in cat_cols:
            nunique = df[col].nunique()
            print(f"\n    {col}")
            print(f"      Categorías únicas: {nunique}")

            if nunique <= 20:
                print(df[col].value_counts().to_string().replace("\n", "\n      "))
            else:
                print("      Top 10 valores:")
                print(df[col].value_counts().head(10).to_string().replace("\n", "\n      "))

    # --------------------------------------------------
    # 6️ Fechas
    # --------------------------------------------------
    date_cols = df.select_dtypes(include=["datetime"]).columns

    if len(date_cols) > 0:
        print("\n Rango temporal")
        for col in date_cols:
            print(f"   {col}: {df[col].min()} -> {df[col].max()}")

    # --------------------------------------------------
    # 7️ Muestra de filas
    # --------------------------------------------------
    print("\n Sample filas")
    print(df.head(sample_rows).to_string())

    print("\n" + "=" * 70)
