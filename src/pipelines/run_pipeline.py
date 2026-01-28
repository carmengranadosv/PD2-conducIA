import argparse
from pathlib import Path

from src.config import RAW_DIR, PROCESSED_DIR
from src.io.download_tlc import download, month_range
from src.processing.clean_tlc import clean_file


def main():
    parser = argparse.ArgumentParser("Pipeline completo TLC")
    parser.add_argument("--from", dest="start", required=True)
    parser.add_argument("--to", dest="end", required=True)
    parser.add_argument("--services", nargs="+", default=["yellow"])
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    services = [s.lower() for s in args.services]

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

    print("\n✅ PIPELINE COMPLETADO")


if __name__ == "__main__":
    main()
