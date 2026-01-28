import argparse
from pathlib import Path

from src.config import RAW_DIR, PROCESSED_DIR, DATA_DIR
from src.io.download_tlc import download, month_range
from src.processing.clean_tlc import clean_file
from src.processing.enrich_tlc import enrich_with_zones


def main():
    parser = argparse.ArgumentParser("Pipeline completo TLC")
    parser.add_argument("--from", dest="start", required=True)
    parser.add_argument("--to", dest="end", required=True)
    parser.add_argument("--services", nargs="+", default=["yellow"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-enrich", action="store_true", help="Saltar el enriquecimiento de zonas")

    args = parser.parse_args()

    services = [s.lower() for s in args.services]

    LOOKUP_PATH = DATA_DIR / "taxi_zone_lookup.csv"

    # 1) Descargar
    for service in services:
        download(service, args.start, args.end, RAW_DIR)

    # 2) Limpiar
    months = month_range(args.start, args.end)
    for service in services:
        for mm in months:
            year = mm[:4]
            in_path = RAW_DIR / "tlc" / service / year / f"{service}_tripdata_{mm}.parquet"
            out_path = (
                PROCESSED_DIR
                / "tlc_clean"
                / service
                / year
                / f"clean_{service}_tripdata_{mm}.parquet"
            )

            if not in_path.exists():
                print(f"FAIL {in_path.name} (no descargado)")
                continue

            msg = clean_file(in_path, out_path, service=service, overwrite=args.overwrite)
            print(msg)

            # Enriquecer con Zonas
            if not args.skip_enrich and out_path.exists():
                if not LOOKUP_PATH.exists():
                    print(f" Error: No encuentro el CSV de zonas en {LOOKUP_PATH}")
                else:
                    msg_enrich = enrich_with_zones(out_path, LOOKUP_PATH)
                    print(f" {msg_enrich}")

    print("\n✅ PIPELINE COMPLETADO")


if __name__ == "__main__":
    main()
