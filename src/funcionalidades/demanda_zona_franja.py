"""Clasifica la demanda por combinacion zona-franja horaria.

Uso recomendado desde la raiz del repo:
    uv run python -m src.funcionalidades.demanda_zona_franja
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = [
    PROJECT_ROOT / "data/processed/tlc_clean/problema1/features/train.parquet",
    PROJECT_ROOT / "data/processed/tlc_clean/problema1/features/val.parquet",
    PROJECT_ROOT / "data/processed/tlc_clean/problema1/features/test.parquet",
]
DEFAULT_RAW_INPUT = PROJECT_ROOT / "data/model_ready/dataset.parquet"
DEFAULT_ZONE_LOOKUP = PROJECT_ROOT / "data/external/taxi_zone_lookup.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports/funcionalidades"

FRANJAS_HORARIAS = {
    "madrugada": range(0, 6),
    "manana": range(6, 12),
    "mediodia": range(12, 16),
    "tarde": range(16, 20),
    "noche": range(20, 24),
}


def asignar_franja_horaria(hora: int) -> str:
    """Devuelve la franja horaria asociada a una hora 0-23."""
    hora = int(hora)
    for franja, horas in FRANJAS_HORARIAS.items():
        if hora in horas:
            return franja
    raise ValueError(f"Hora fuera de rango: {hora}")


def cargar_datos_demanda(paths: list[Path] | None = None) -> pd.DataFrame:
    """Carga datos agregados zona-hora o, si no existen, los agrega desde viajes."""
    input_paths = paths or [path for path in DEFAULT_INPUTS if path.exists()]

    if input_paths:
        df = pd.concat([pd.read_parquet(path) for path in input_paths], ignore_index=True)
    elif DEFAULT_RAW_INPUT.exists():
        df = pd.read_parquet(DEFAULT_RAW_INPUT)
    else:
        rutas = ", ".join(str(path) for path in DEFAULT_INPUTS)
        raise FileNotFoundError(
            f"No se encontraron datos agregados ({rutas}) ni {DEFAULT_RAW_INPUT}"
        )

    if {"origen_id", "hora", "demanda"}.issubset(df.columns):
        out = df[["origen_id", "hora", "demanda"]].copy()
    elif {"origen_id", "timestamp_hora", "demanda"}.issubset(df.columns):
        out = df[["origen_id", "timestamp_hora", "demanda"]].copy()
        out["timestamp_hora"] = pd.to_datetime(out["timestamp_hora"])
        out["hora"] = out["timestamp_hora"].dt.hour
        out = out[["origen_id", "hora", "demanda"]]
    elif {"origen_id", "fecha_inicio"}.issubset(df.columns):
        df = df[["origen_id", "fecha_inicio"]].copy()
        df["fecha_inicio"] = pd.to_datetime(df["fecha_inicio"])
        df["timestamp_hora"] = df["fecha_inicio"].dt.floor("h")
        out = (
            df.groupby(["origen_id", "timestamp_hora"], observed=True)
            .size()
            .reset_index(name="demanda")
        )
        out["hora"] = out["timestamp_hora"].dt.hour
        out = out[["origen_id", "hora", "demanda"]]
    else:
        raise ValueError(
            "El dataset necesita columnas ('origen_id', 'hora', 'demanda') "
            "o ('origen_id', 'fecha_inicio')."
        )

    out["origen_id"] = out["origen_id"].astype(int)
    out["hora"] = out["hora"].astype(int)
    out["demanda"] = pd.to_numeric(out["demanda"], errors="coerce").fillna(0)
    return out


def clasificar_niveles_demanda(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Agrega por zona-franja y etiqueta cada combinacion como baja, media o alta."""
    trabajo = df.copy()
    trabajo["franja_horaria"] = trabajo["hora"].apply(asignar_franja_horaria)

    resumen = (
        trabajo.groupby(["origen_id", "franja_horaria"], observed=True)
        .agg(
            demanda_media=("demanda", "mean"),
            demanda_mediana=("demanda", "median"),
            demanda_total=("demanda", "sum"),
            horas_observadas=("demanda", "size"),
        )
        .reset_index()
    )

    hora_pico = (
        trabajo.groupby(["origen_id", "franja_horaria", "hora"], observed=True)["demanda"]
        .mean()
        .reset_index(name="demanda_media_hora")
        .sort_values(["origen_id", "franja_horaria", "demanda_media_hora", "hora"])
        .groupby(["origen_id", "franja_horaria"], observed=True)
        .tail(1)
        .rename(columns={"hora": "hora_pico", "demanda_media_hora": "demanda_hora_pico"})
    )
    resumen = resumen.merge(
        hora_pico[
            ["origen_id", "franja_horaria", "hora_pico", "demanda_hora_pico"]
        ],
        on=["origen_id", "franja_horaria"],
        how="left",
    )

    q_baja = float(resumen["demanda_media"].quantile(0.33))
    q_alta = float(resumen["demanda_media"].quantile(0.66))

    if np.isclose(q_baja, q_alta):
        resumen["nivel_demanda"] = "media"
    else:
        resumen["nivel_demanda"] = np.select(
            [
                resumen["demanda_media"] <= q_baja,
                resumen["demanda_media"] <= q_alta,
            ],
            ["baja", "media"],
            default="alta",
        )

    resumen["score_demanda_0_100"] = (
        resumen["demanda_media"].rank(pct=True, method="average") * 100
    ).round(2)

    metadata = {
        "criterio": "Clasificacion por terciles de demanda_media zona-franja.",
        "umbral_baja_media": round(q_baja, 4),
        "umbral_media_alta": round(q_alta, 4),
        "niveles": {
            "baja": f"demanda_media <= {q_baja:.4f}",
            "media": f"{q_baja:.4f} < demanda_media <= {q_alta:.4f}",
            "alta": f"demanda_media > {q_alta:.4f}",
        },
        "franjas_horarias": {
            franja: f"{min(horas):02d}:00-{max(horas) + 1:02d}:00"
            for franja, horas in FRANJAS_HORARIAS.items()
        },
        "filas_resultado": int(len(resumen)),
    }
    return resumen.sort_values(
        ["nivel_demanda", "demanda_media"], ascending=[True, False]
    ), metadata


def enriquecer_con_zonas(
    resumen: pd.DataFrame, zone_lookup_path: Path = DEFAULT_ZONE_LOOKUP
) -> pd.DataFrame:
    """Anade nombre de zona y borough si existe el lookup de TLC."""
    if not zone_lookup_path.exists():
        return resumen

    zonas = pd.read_csv(zone_lookup_path).rename(
        columns={
            "LocationID": "origen_id",
            "Borough": "origen_barrio",
            "Zone": "origen_zona",
            "service_zone": "origen_service_zone",
        }
    )
    return resumen.merge(
        zonas[["origen_id", "origen_barrio", "origen_zona", "origen_service_zone"]],
        on="origen_id",
        how="left",
    )


def guardar_resultados(resumen: pd.DataFrame, metadata: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    resumen.to_csv(out_dir / "demanda_zona_franja.csv", index=False)
    resumen.to_parquet(out_dir / "demanda_zona_franja.parquet", index=False)

    top_alta = resumen[resumen["nivel_demanda"] == "alta"].nlargest(20, "demanda_media")
    top_alta.to_csv(out_dir / "top20_demanda_alta.csv", index=False)

    with open(out_dir / "demanda_zona_franja_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def normalizar_texto(valor: str) -> str:
    """Normaliza texto para busquedas tolerantes a mayusculas y acentos."""
    texto = unicodedata.normalize("NFKD", str(valor).strip().lower())
    return "".join(letra for letra in texto if not unicodedata.combining(letra))


def normalizar_franja(franja: str) -> str:
    franja_norm = normalizar_texto(franja)
    alias = {
        "manana": "manana",
        "mañana": "manana",
        "medio dia": "mediodia",
        "mediodia": "mediodia",
        "medio-dia": "mediodia",
    }
    return alias.get(franja_norm, franja_norm)


def cargar_resumen_para_consulta(
    out_dir: Path = DEFAULT_REPORT_DIR,
    inputs: list[Path] | None = None,
    zone_lookup_path: Path = DEFAULT_ZONE_LOOKUP,
) -> pd.DataFrame:
    """Carga el informe si existe; si no existe, lo calcula en memoria sin guardarlo."""
    parquet_path = out_dir / "demanda_zona_franja.parquet"
    csv_path = out_dir / "demanda_zona_franja.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    datos = cargar_datos_demanda(inputs)
    resumen, _ = clasificar_niveles_demanda(datos)
    return enriquecer_con_zonas(resumen, zone_lookup_path)


def consultar_demanda(
    resumen: pd.DataFrame,
    zona: str | None = None,
    franja: str | None = None,
    nivel: str | None = None,
    demanda_min: float | None = None,
    demanda_max: float | None = None,
    top: int = 20,
) -> pd.DataFrame:
    """Filtra combinaciones zona-franja por uno o varios criterios."""
    filtros = [zona, franja, nivel, demanda_min, demanda_max]
    if all(valor is None for valor in filtros):
        raise ValueError("Para consultar debes introducir al menos un filtro.")

    resultado = resumen.copy()

    if zona is not None:
        zona_norm = normalizar_texto(zona)
        if zona_norm.isdigit():
            resultado = resultado[resultado["origen_id"].astype(str) == zona_norm]
        else:
            cols_zona = [
                col
                for col in ["origen_zona", "origen_barrio", "origen_service_zone"]
                if col in resultado.columns
            ]
            mask = pd.Series(False, index=resultado.index)
            for col in cols_zona:
                mask |= resultado[col].fillna("").map(normalizar_texto).str.contains(
                    zona_norm, regex=False
                )
            resultado = resultado[mask]

    if franja is not None:
        franja_norm = normalizar_franja(franja)
        resultado = resultado[
            resultado["franja_horaria"].map(normalizar_franja) == franja_norm
        ]

    if nivel is not None:
        nivel_norm = normalizar_texto(nivel)
        resultado = resultado[
            resultado["nivel_demanda"].map(normalizar_texto) == nivel_norm
        ]

    if demanda_min is not None:
        resultado = resultado[resultado["demanda_media"] >= demanda_min]

    if demanda_max is not None:
        resultado = resultado[resultado["demanda_media"] <= demanda_max]

    return resultado.nlargest(top, "demanda_media")


def hay_filtros_consulta(args: argparse.Namespace) -> bool:
    return any(
        valor is not None
        for valor in [
            args.zona,
            args.franja,
            args.nivel,
            args.demanda_min,
            args.demanda_max,
        ]
    )


def imprimir_consulta(resultado: pd.DataFrame) -> None:
    columnas = [
        "origen_id",
        "origen_zona",
        "origen_barrio",
        "franja_horaria",
        "demanda_media",
        "demanda_mediana",
        "nivel_demanda",
        "score_demanda_0_100",
        "hora_pico",
    ]
    columnas = [col for col in columnas if col in resultado.columns]

    if resultado.empty:
        print("No hay resultados para esos filtros.")
        return

    print(f"Resultados encontrados: {len(resultado)}")
    print(resultado[columnas].to_string(index=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clasifica demanda baja/media/alta por zona y franja horaria."
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        help="Parquets de entrada. Por defecto usa los features de problema1.",
    )
    parser.add_argument(
        "--zone-lookup",
        type=Path,
        default=DEFAULT_ZONE_LOOKUP,
        help="CSV taxi_zone_lookup para anadir nombres de zonas.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directorio donde guardar CSV, Parquet y metadata.",
    )
    parser.add_argument(
        "--consultar",
        action="store_true",
        help="Activa el modo buscador. Requiere al menos un filtro.",
    )
    parser.add_argument(
        "--zona",
        help="ID o texto de zona/barrio. Ej: 161, Midtown, Queens.",
    )
    parser.add_argument(
        "--franja",
        help="Franja horaria: madrugada, manana, mediodia, tarde o noche.",
    )
    parser.add_argument(
        "--nivel",
        choices=["baja", "media", "alta"],
        help="Nivel de demanda a buscar.",
    )
    parser.add_argument(
        "--demanda-min",
        type=float,
        help="Demanda media minima.",
    )
    parser.add_argument(
        "--demanda-max",
        type=float,
        help="Demanda media maxima.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Numero maximo de filas a mostrar en modo consulta.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.consultar or hay_filtros_consulta(args):
        if not hay_filtros_consulta(args):
            raise SystemExit("Error: en modo consulta debes indicar al menos un filtro.")
        resumen = cargar_resumen_para_consulta(args.out_dir, args.inputs, args.zone_lookup)
        resultado = consultar_demanda(
            resumen=resumen,
            zona=args.zona,
            franja=args.franja,
            nivel=args.nivel,
            demanda_min=args.demanda_min,
            demanda_max=args.demanda_max,
            top=args.top,
        )
        imprimir_consulta(resultado)
        return

    datos = cargar_datos_demanda(args.inputs)
    resumen, metadata = clasificar_niveles_demanda(datos)
    resumen = enriquecer_con_zonas(resumen, args.zone_lookup)
    guardar_resultados(resumen, metadata, args.out_dir)

    print(f"Resultados guardados en: {args.out_dir}")
    print(resumen["nivel_demanda"].value_counts().to_string())
    print("\nTop 10 combinaciones de demanda alta:")
    columnas = [
        "origen_id",
        "origen_zona",
        "franja_horaria",
        "demanda_media",
        "nivel_demanda",
        "hora_pico",
    ]
    columnas = [col for col in columnas if col in resumen.columns]
    print(resumen.nlargest(10, "demanda_media")[columnas].to_string(index=False))


if __name__ == "__main__":
    main()
