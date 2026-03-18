import argparse
from pathlib import Path

# Configuración
from src.config import RAW_DIR, PROCESSED_DIR, DATA_DIR

# Descargas
from src.io.download_tlc import download, month_range
from src.io.download_tlc import ensure_taxi_zone_lookup
from src.io.weather import download_weather_data

# Procesamiento
from src.processing.clean_tlc import clean_file
from src.processing.enrich_tlc import enrich_data

from src.processing.stats import print_dataset_stats



def main():
    parser = argparse.ArgumentParser("Pipeline completo TLC")
    parser.add_argument("--from", dest="start", required=True)
    parser.add_argument("--to", dest="end", required=True)
    parser.add_argument("--services", nargs="+", default=["yellow"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-enrich", action="store_true", help="Saltar el enriquecimiento de zonas")

    args = parser.parse_args()

    services = [s.lower() for s in args.services]

    # RUTAS
    # Rutas de datos adicionales
    LOOKUP_PATH = DATA_DIR / "external" / "taxi_zone_lookup.csv"
    WEATHER_PATH = DATA_DIR / "external" / "weather" / f"nyc_weather_{args.start}_{args.end}.parquet"
    HOLIDAYS_PATH = DATA_DIR / "external" / "holidays" / f"nyc_holidays_{args.start}_{args.end}.parquet"
    EVENTS_PATH = DATA_DIR / "external" / "events" / f"nyc_events_{args.start}_{args.end}.parquet"

    # Generar datos auxiliares (festivos, eventos, tráfico) si no existen
    if not args.skip_enrich:  
        from src.io.download_holidays import create_holidays_calendar_range
        from src.io.download_events import create_major_events
        
        print("\n" + "="*60)
        print("GENERANDO DATOS AUXILIARES")
        print("="*60)
        
        # 1. Festivos
        create_holidays_calendar_range(args.start, args.end, HOLIDAYS_PATH, overwrite=args.overwrite)
        
        # 2. Eventos
        create_major_events(EVENTS_PATH, overwrite=args.overwrite)
        
        print()

    # Descargar zonas y clima
    if not args.skip_enrich:
        print("="*60)
        print("DESCARGANDO DATOS BASE")
        print("="*60)
        print(ensure_taxi_zone_lookup(LOOKUP_PATH, overwrite=args.overwrite))
        download_weather_data(args.start, args.end, WEATHER_PATH, overwrite=args.overwrite)
        print()

    # Descargar TLC
    print("="*60)
    print("DESCARGANDO DATOS TLC")
    print("="*60)
    for service in services:
        download(service, args.start, args.end, RAW_DIR)
    print()

    # Limpiar y Enriquecer
    print("="*60)
    print("LIMPIEZA Y ENRIQUECIMIENTO")
    print("="*60)
    
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

            # Limpiar
            msg = clean_file(in_path, out_path, service=service, overwrite=args.overwrite)
            print(msg)

            # Enriquecer
            if not args.skip_enrich and out_path.exists():
                if not LOOKUP_PATH.exists():
                    print(f"   No encuentro {LOOKUP_PATH}")
                else:
                    msg_enrich = enrich_data(
                        out_path, 
                        LOOKUP_PATH, 
                        WEATHER_PATH if WEATHER_PATH.exists() else None,
                        HOLIDAYS_PATH if HOLIDAYS_PATH.exists() else None,
                        EVENTS_PATH if EVENTS_PATH.exists() else None,
                    )
                    print(f"  {msg_enrich}")

            print_dataset_stats(out_path)



    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETADO")
    print("="*60)


if __name__ == "__main__":
    main()