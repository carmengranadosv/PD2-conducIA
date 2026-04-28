from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


CONTEXT_REL = Path("processed/tlc_clean/contexto_web")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Muestra que contexto historico usaria la web para una zona, fecha y hora."
    )
    parser.add_argument("--zona", type=int, default=230, help="ID de zona TLC de origen.")
    parser.add_argument(
        "--fecha",
        type=str,
        default="2026-04-03",
        help="Fecha objetivo en formato YYYY-MM-DD.",
    )
    parser.add_argument("--hora", type=int, default=18, help="Hora objetivo, 0-23.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Raiz local de datos.",
    )
    return parser.parse_args()


def load_context(data_root: Path, name: str) -> pd.DataFrame:
    path = data_root / CONTEXT_REL / name
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Genera los contextos con "
            "`uv run python despliegue/preparar_contexto_web.py --data-root data`."
        )
    return pd.read_parquet(path)


def buscar_contexto(df: pd.DataFrame, zona: int, mes: int, dia_semana: int, hora: int):
    niveles = [
        ("zona + mes + dia_semana + hora", {"origen_id": zona, "mes_num": mes, "dia_semana": dia_semana, "hora": hora}),
        ("zona + mes + hora", {"origen_id": zona, "mes_num": mes, "hora": hora}),
        ("zona + dia_semana + hora", {"origen_id": zona, "dia_semana": dia_semana, "hora": hora}),
        ("zona + hora", {"origen_id": zona, "hora": hora}),
        ("zona", {"origen_id": zona}),
    ]

    for nombre, filtro in niveles:
        mask = pd.Series(True, index=df.index)
        for col, valor in filtro.items():
            mask &= df[col].astype(int) == int(valor)

        filtrado = df.loc[mask]
        if not filtrado.empty:
            if len(filtrado) == 1:
                return nombre, filtrado.iloc[0]

            resumen = {}
            for col in filtrado.columns:
                serie = filtrado[col].dropna()
                if serie.empty:
                    resumen[col] = np.nan
                elif pd.api.types.is_numeric_dtype(serie):
                    resumen[col] = serie.mean()
                else:
                    resumen[col] = serie.mode().iloc[0]
            return nombre, pd.Series(resumen)

    return "sin contexto", None


def print_row(title: str, level: str, row: pd.Series | None, columns: list[str]) -> None:
    print(f"\n## {title}")
    print(f"nivel usado: {level}")
    if row is None:
        print("No hay contexto para esa zona.")
        return

    for col in columns:
        if col in row.index:
            print(f"{col}: {row[col]}")


def main() -> None:
    args = parse_args()
    fecha = datetime.strptime(args.fecha, "%Y-%m-%d")
    mes = fecha.month
    dia_semana = fecha.weekday()
    hora = max(0, min(int(args.hora), 23))

    print("Consulta objetivo")
    print(f"zona: {args.zona}")
    print(f"fecha: {fecha.date()}  mes_num: {mes}  dia_semana: {dia_semana}  hora: {hora}")

    p2 = load_context(args.data_root, "contexto_p2.parquet")
    p5 = load_context(args.data_root, "contexto_p5.parquet")

    nivel_p2, row_p2 = buscar_contexto(p2, args.zona, mes, dia_semana, hora)
    nivel_p5, row_p5 = buscar_contexto(p5, args.zona, mes, dia_semana, hora)

    print_row(
        "Taxi / P2",
        nivel_p2,
        row_p2,
        [
            "origen_id",
            "mes_num",
            "dia_semana",
            "hora",
            "_n_contexto",
            "n_viajes",
            "oferta_inferida",
            "tasa_historica",
            "espera_media",
            "demanda_p1",
            "temp_c",
            "precipitation",
            "num_eventos",
        ],
    )

    print_row(
        "VTC / P5",
        nivel_p5,
        row_p5,
        [
            "origen_id",
            "mes_num",
            "dia_semana",
            "hora",
            "_n_contexto",
            "origen_zona",
            "origen_barrio",
            "tipo_vehiculo",
            "distancia",
            "duracion_min",
            "espera_min",
            "precio_total_est",
            "velocidad_mph",
            "evento_tipo",
            "franja_horaria",
            "trafico_denso",
        ],
    )


if __name__ == "__main__":
    main()
