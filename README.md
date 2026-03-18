# PD2-conducIA

Estructura: 

```

PD2-conducIA/
│
├── README.md
│
├── data/
│   ├── raw/
│   │   └── tlc/
│   │       └── <service>/
│   │           └── <year>/
│   │               └── <service>_tripdata_YYYY-MM.parquet
│   │       
│   │
│   ├── processed/
│   │   └── tlc_clean/
│   │       └── <service>/
│   │           └── <year>/
│   │               └── clean_<service>_tripdata_YYYY-MM.parquet
│   │
│   └── external/
│       └── taxi_zone_lookup.csv
│
├── src/
│   ├── config.py (rutas y configuración)
│   │
│   ├── io/
│   │   └── download_tlc.py (descargar csvs y parquets)
│   │
│   ├── analysis/
│   │   └── 1_data_analysis.ipynb (notebooks de analisis)
│   │
│   ├── processing/
│   │   ├── columnas.py (columnas de taxis y ubers)
│   │   ├── clean_tlc.py (limpieza de parquets y preparación de datos)
│   │   └── enrich_tlc.py (añadido los datos de lookup de zonas)
│   │
│   └── pipelines/
│       └── run_pipeline.py (main principal ETL)
│
└── docs/
    ├── datos.md
    └── estudio_mercado.md
    
```
