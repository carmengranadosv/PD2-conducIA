# PD2-conducIA

Repositorio para la preparación de datos, análisis y modelado de varios problemas relacionados con demanda de taxis y apoyo a la toma de decisiones del conductor.

## Estructura del proyecto

```text
PD2-conducIA/
├── README.md
├── pyproject.toml                # Dependencias y configuración del proyecto
├── uv.lock                       # Lockfile de uv
│
├── data/
│   ├── raw/                      # Datos originales descargados
│   │   └── tlc/
│   ├── processed/                # Datos limpios e intermedios
│   │   └── tlc_clean/
│   ├── model_ready/              # Datos listos para consumo por modelos
│   ├── outputs/                  # Salidas generadas por pipelines
│   │   └── problema1/
│   └── external/                 # Fuentes auxiliares externas
│       ├── events/
│       └── holidays/
│
├── docs/
│   ├── datos.md
│   ├── estudio_mercado.md
│   ├── funcionalidad_demanda_zonas.md
│   ├── resumen_problema1_problema2.md
│   └── memory/                   # Material de entrega en HTML/CSS
│
├── models/
│   ├── problema1/                # Modelos y transformadores entrenados
│   └── problema2/
│
├── reports/
│   ├── funcionalidades/          # Metadatos y resultados de utilidades
│   ├── problema1/                # Resultados, plots y análisis
│   │   ├── plots/
│   │   ├── resultados/
│   │   └── ejemplos_random_forest/
│   └── problema2/
│       ├── plots/
│       └── resultados/
│
└── src/
    ├── config.py                 # Rutas base del proyecto
    ├── analysis/                 # Notebooks de exploración
    ├── funcionalidades/          # Funcionalidades orientadas a uso práctico
    ├── io/                       # Descarga e ingesta de datos externos
    ├── pipelines/                # Pipelines generales del proyecto
    ├── processing/               # Limpieza, enriquecimiento y estadísticas
    └── modelos/
        ├── preparar_datosFinales/ # Construcción del dataset final consolidado
        ├── problema1/            # Predicción de demanda por zona/franja
        ├── problema2/            # Clasificación/recomendación de zonas
        ├── problema4/            # Modelado basado en grafos / STGCN
        └── problema5/            # Modelos adicionales experimentales
```

## Resumen rápido por carpetas

- `data/`: datasets crudos, procesados y salidas intermedias.
- `src/io/`: descarga de TLC, eventos, festivos y weather.
- `src/processing/`: limpieza de datos, enriquecimiento y utilidades de procesamiento.
- `src/modelos/`: preparación de datasets, entrenamiento y evaluación por problema.
- `src/funcionalidades/`: scripts orientados a consultas o casos de uso concretos.
- `models/`: artefactos entrenados reutilizables.
- `reports/`: métricas, gráficas y análisis exportados.
- `docs/`: documentación de apoyo y resúmenes del proyecto.

## Problemas principales

- `problema1`: predicción de demanda por `zona x franja temporal`.
- `problema2`: recomendación o clasificación de zonas favorables para el conductor.
- `problema4`: análisis y modelado de eficiencia en relación con clima y contexto operativo.
- `problema5`: predicción y análisis de propinas.

## Punto de partida recomendado

Si quieres ubicarte rápido en el repositorio:

1. Lee `docs/resumen_problema1_problema2.md`.
2. Revisa `src/modelos/problema1/` y `src/modelos/problema2/`.
3. Consulta `reports/` para ver resultados ya generados.
