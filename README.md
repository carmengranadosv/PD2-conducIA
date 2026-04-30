# ConducIA

ConducIA es un proyecto de analisis, modelado y despliegue orientado a la movilidad urbana en Nueva York. Usa datos historicos de viajes TLC, clima, eventos y festivos para apoyar decisiones de conductores de taxi y VTC mediante modelos de Machine Learning.

La aplicacion final esta construida con FastAPI y ofrece paneles web para estimar demanda, evaluar zonas, analizar trayectos y consultar funcionalidades agregadas por zona y franja horaria.

## Que incluye

- Prediccion de demanda de taxis por zona y hora.
- Recomendacion de zonas con mayor probabilidad de exito para taxistas.
- Auditoria de trayectos VTC con estimacion de propina, velocidad y potencial de retorno.
- Consulta de demanda por zona y franja horaria.
- Mapa coropletico de zonas, si el HTML generado esta disponible.
- Scripts de preparacion, limpieza, entrenamiento, evaluacion y despliegue.

## Tecnologias principales

- Python
- FastAPI
- Jinja2
- Pandas / NumPy / PyArrow
- Scikit-learn
- Keras
- JAX
- XGBoost
- Uvicorn
- Docker
- uv

## Estructura del proyecto

```text
PD2-conducIA/
├── README.md
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── Dockerfile
│
├── data/
│   ├── raw/                         # Datos originales
│   ├── processed/                   # Datos procesados e intermedios
│   ├── model_ready/                 # Datos listos para modelos
│   ├── outputs/                     # Salidas generadas
│   ├── external/                    # Fuentes auxiliares
│   └── funcionalidades/             # Recursos generados para funcionalidades
│
├── despliegue/
│   ├── main.py                      # Aplicacion FastAPI
│   ├── preparar_contexto_web.py     # Genera contexto ligero para la web
│   ├── verificar_contexto_web.py    # Comprueba contexto usado por la web
│   ├── modelos_finales/             # Modelos finales usados en despliegue
│   ├── reentrenar/                  # Scripts de reentrenamiento
│   └── templates/                   # Vistas HTML
│
├── docs/                            # Documentacion del proyecto
├── models/                          # Modelos de experimentacion
├── reports/                         # Metricas, graficas y resultados
│
└── src/
    ├── analysis/                    # Notebooks de analisis
    ├── funcionalidades/             # Consultas y utilidades de negocio
    ├── io/                          # Descarga e ingesta de datos
    ├── processing/                  # Limpieza y enriquecimiento
    ├── pipelines/                   # Pipelines generales
    └── modelos/                     # Preparacion, entrenamiento y evaluacion
```

## Modelos usados en la aplicacion

La app de despliegue carga los artefactos desde `despliegue/modelos_finales/`:

| Problema | Artefacto | Uso |
| --- | --- | --- |
| P1 | `modelo_p1_rf.joblib` | Prediccion de demanda por zona y hora |
| P2 | `modelo_p2_mlp.keras` | Probabilidad de exito de una zona para taxi |
| P2 | `modelo_p2_mlp_scaler.pkl` | Escalado de variables del modelo P2 |
| P2 | `modelo_p2_zona_encoder.pkl` | Codificacion de zonas del modelo P2 |
| P4 | `modelo_p4_red_neuronal.keras` | Prediccion de velocidad segun zona y contexto |
| P4 | `modelo_p4_scaler_clima.joblib` | Escalado de variables climaticas |
| P4 | `modelo_p4_label_encoder_zonas.joblib` | Codificacion de zonas del modelo P4 |
| P5 | `modelo_p5_xgboost.joblib` | Estimacion de propina |

## Instalacion local

Requisitos recomendados:

- Python compatible con el proyecto. El `pyproject.toml` declara `>=3.13`.
- `uv` instalado.

Desde la raiz del repositorio:

```bash
uv sync
```

Si prefieres usar `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datos necesarios

Para que la aplicacion funcione correctamente, debe existir la carpeta `data/` con los ficheros procesados esperados. Las rutas principales que usa `despliegue/main.py` son:

```text
data/processed/tlc_clean/datos_final.parquet
data/processed/tlc_clean/contexto_web/contexto_p2.parquet
data/processed/tlc_clean/contexto_web/contexto_p5.parquet
data/processed/tlc_clean/problema2/features/train.parquet
data/processed/tlc_clean/problema5/train.parquet
data/external/taxi_zone_lookup.csv
data/funcionalidades/mapa_poder_barrios.html
```

La app intenta usar primero los contextos ligeros de `contexto_web/`. Si no existen, usa como respaldo un primer batch de los datasets de entrenamiento de P2 y P5.

Para generar los contextos web:

```bash
uv run python despliegue/preparar_contexto_web.py --data-root data
```

Para comprobar que contexto usaria la web en una zona y hora concretas:

```bash
uv run python despliegue/verificar_contexto_web.py --zona 230 --fecha 2026-04-03 --hora 18 --data-root data
```


## Ejecutar con Docker

Construir la imagen:

```bash
docker build -t conducia .
```

Ejecutar el contenedor montando los datos locales:

```bash
docker run --rm -p 8000:8000 -v "$(pwd)/data:/app/data" conducia
```

La variable `CONDUCIA_DATA_DIR` permite indicar otra raiz de datos:

```bash
docker run --rm -p 8000:8000 -e CONDUCIA_DATA_DIR=/app/data -v "$(pwd)/data:/app/data" conducia
```

Rutas principales:

- `/`: pantalla de inicio.
- `/taxi`: panel de posicionamiento para taxistas.
- `/vtc`: auditoria de trayecto VTC.
- `/funcionalidades`: consulta de demanda por zona/franja y acceso al mapa.
- `/documentacion`: vista de documentacion.
- `/funcionalidades/mapa`: mapa coropletico, si esta generado.

## Funcionalidades por linea de comandos

Generar o actualizar la demanda por zona y franja horaria:

```bash
uv run python -m src.funcionalidades.demanda_zona_franja
```

Consultar ejemplos:

```bash
uv run python -m src.funcionalidades.demanda_zona_franja --consultar --zona Midtown --franja tarde
uv run python -m src.funcionalidades.demanda_zona_franja --consultar --nivel alta --franja noche --top 5
uv run python -m src.funcionalidades.demanda_zona_franja --consultar --zona 161
```

Generar el mapa coropletico:

```bash
uv run python -m src.funcionalidades.mapa_coropletico
```

## Flujo general de trabajo

1. Descargar o preparar los datos base en `data/`.
2. Limpiar y enriquecer los datos con los modulos de `src/processing/`.
3. Construir datasets finales desde `src/modelos/preparar_datosFinales/`.
4. Preparar features y entrenar modelos por problema en `src/modelos/`.
5. Copiar o generar los artefactos finales en `despliegue/modelos_finales/`.
6. Generar contexto web con `despliegue/preparar_contexto_web.py`.
7. Levantar la aplicacion FastAPI.

## Problemas modelados

### Problema 1: demanda por zona y hora

Predice cuantos viajes se esperan en una zona y hora concretas. La aplicacion usa este modelo para estimar demanda y para calcular el potencial de retorno en destino.

### Problema 2: exito de zona para taxi

Clasifica si una zona es favorable para posicionarse, teniendo en cuenta demanda, oferta inferida, hora, calendario, clima y eventos.

### Problema 4: velocidad y eficiencia

Predice velocidad esperada a partir de zona, clima y contexto temporal. Se usa en el panel VTC para evaluar eficiencia del trayecto.

### Problema 5: propinas

Estima la propina esperada para un trayecto segun precio, zona, tipo de vehiculo, clima, eventos, duracion y otras variables contextuales.

## Documentacion adicional

- `docs/datos.md`
- `docs/estudio_mercado.md`
- `docs/resumen_problema1_problema2.md`
- `docs/funcionalidad_demanda_zonas.md`
- `docs/contexto_web.md`
- `docs/memory/`

