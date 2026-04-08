# Resumen Facil de Problema 1 y Problema 2

Este documento resume, de forma corta y práctica, qué hace cada problema, cuál parece ser el flujo actual y qué scripts parecen antiguos o ya no formar parte del pipeline principal.

## Problema 1

### Qué hace

El Problema 1 intenta predecir la demanda de taxis por `zona x hora`.

En otras palabras:
- para cada zona de NYC,
- usando el historial reciente y variables de contexto,
- intenta predecir cuántos viajes habrá en la siguiente hora.

Esto sirve para estimar en qué zonas habrá más demanda.

### Flujo actual que parece usarse

Según el código actual, el flujo más claro y moderno es:

1. [preparar_datos.py](/home/marcospoza/PD2/PD2-conducIA/src/modelos/problema1/preparar_datos.py)
2. [features.py](/home/marcospoza/PD2/PD2-conducIA/src/modelos/problema1/features.py)
3. [baseline.py](/home/marcospoza/PD2/PD2-conducIA/src/modelos/problema1/baseline.py)
4. [lstm.py](/home/marcospoza/PD2/PD2-conducIA/src/modelos/problema1/lstm.py)
5. [lstm2.py](/home/marcospoza/PD2/PD2-conducIA/src/modelos/problema1/lstm2.py)
6. [transformer.py](/home/marcospoza/PD2/PD2-conducIA/src/modelos/problema1/transformer.py)
7. [comparar_modelos.py](/home/marcospoza/PD2/PD2-conducIA/src/modelos/problema1/comparar_modelos.py)
8. [analizar_predicciones.py](/home/marcospoza/PD2/PD2-conducIA/src/modelos/problema1/analizar_predicciones.py)

### Qué hace cada script importante

`preparar_datos.py`
- Lee `data/processed/tlc_clean/datos_final.parquet`.
- Agrega directamente el dataset completo a `zona x hora`.
- Hace el split temporal `train/val/test`.
- Deja los datos en `data/processed/tlc_clean/problema1/raw/`.

`features.py`
- Parte de los datos agregados.
- Crea `target`, lags, rolling means y `media_hist`.
- Guarda las features finales en `data/processed/tlc_clean/problema1/features/`.

`baseline.py`
- Entrena 3 modelos simples:
  - `Naive`
  - `Media_hist`
  - `RandomForest`
- Guarda el Random Forest en `models/problema1/`.
- Guarda métricas y gráficos en `reports/problema1/`.

`lstm.py`
- Entrena un modelo LSTM para comparar con el Random Forest.
- Guarda resultados y gráficas en `models/problema1/` y `reports/problema1/`.

`lstm2.py`
- Es otra versión útil del modelo LSTM.
- Puede servir para comparar variantes o conservar una iteración diferente del enfoque secuencial.

`transformer.py`
- Es un modelo adicional útil para probar una arquitectura distinta al LSTM y al Random Forest.
- Sirve como comparación extra dentro del Problema 1.

`comparar_modelos.py`
- Lee los resultados ya generados y saca una comparativa conjunta.

`analizar_predicciones.py`
- Sirve para entender mejor cómo predice el Random Forest.
- Enseña ejemplos reales, ventanas de 24 horas y casos buenos/malos.

`probar_ejemplos_reales.py`
- Es una utilidad útil para revisar ejemplos concretos del comportamiento del modelo.
- Ayuda a interpretar mejor qué tan bien funciona en casos reales.

### Qué sale de Problema 1

Lo más importante que produce es:

- datos agregados en `data/processed/tlc_clean/problema1/raw/`
- features en `data/processed/tlc_clean/problema1/features/`
- modelo RF en `models/problema1/baseline_random_forest.pkl`
- resultados en `reports/problema1/resultados/`
- gráficos y análisis en `reports/problema1/`

### Scripts que parecen antiguos o duplicados

Estos siguen en el repo, pero por el código parecen haber quedado como versiones previas, alternativas o experimentos:

`samplear_datos.py`
- Parece antiguo.
- La propia idea de `preparar_datos.py` es sustituir el pipeline con sampleo y trabajar ya con agregación directa.

`division_datos.py`
- Parece parte del flujo antiguo basado en sampleo.
- Divide viajes individuales antes de agregar.

`agregacion.py`
- Parece anterior a `features.py`.
- En [features.py](/home/marcospoza/PD2/PD2-conducIA/src/modelos/problema1/features.py) aparece incluso el comentario: "puedes borrar agregacion.py — este lo reemplaza".

`modelo1_lstm.py`
- Parece una versión anterior del LSTM.
- El script más consistente con el pipeline actual es `lstm.py`.

`predicciones_formateadas.py`
- No es entrenamiento.
- Es más bien una utilidad de exportación de predicciones ya hechas.

`problema1.md`
- Documentación antigua.
- Ojo: describe un flujo donde `agregacion.py` y `samplear_datos.py` son centrales, pero eso ya no parece el camino más actual del repo.

### Resumen corto de Problema 1

Problema 1 es el módulo de predicción de demanda por zona y hora.  
La ruta actual más limpia parece ser:

`datos_final.parquet -> preparar_datos.py -> features.py -> baseline.py / lstm.py -> comparar y analizar`

Y el modelo más importante ahora mismo parece ser el `Random Forest`, porque además se reutiliza después en Problema 2 como señal de entrada.

---

## Problema 2

### Qué hace

El Problema 2 intenta decidir si una zona es una buena opción para un conductor en un momento dado.

No predice directamente "cuántos viajes habrá", sino algo más tipo:
- qué zonas tienen más probabilidad de ser buenas,
- o dónde un taxi podría encontrar un viaje con mejores condiciones.

En el código actual esto se formula como una clasificación binaria:
- `target = 1` si la zona está entre las mejores de esa ventana temporal,
- `target = 0` si no.

### La parte importante: hay dos versiones

En el repo existen dos ramas de código para este problema:

1. [src/modelos/problema2](/home/marcospoza/PD2/PD2-conducIA/src/modelos/problema2)
2. [src/modelos/p2](/home/marcospoza/PD2/PD2-conducIA/src/modelos/p2)

Por cómo está escrito el código, la carpeta que parece actual es:

[src/modelos/p2](/home/marcospoza/PD2/PD2-conducIA/src/modelos/p2)

La carpeta `src/modelos/problema2` parece más antigua o de transición.

### Flujo actual que parece usarse

El flujo más coherente en el código actual es:

1. [preparar_datos.py](/home/marcospoza/PD2/PD2-conducIA/src/modelos/p2/preparar_datos.py)
2. [features.py](/home/marcospoza/PD2/PD2-conducIA/src/modelos/p2/features.py)
3. [baseline.py](/home/marcospoza/PD2/PD2-conducIA/src/modelos/p2/baseline.py)
4. [mlp.py](/home/marcospoza/PD2/PD2-conducIA/src/modelos/p2/mlp.py)
5. [lightgbm.py](/home/marcospoza/PD2/PD2-conducIA/src/modelos/p2/lightgbm.py)
6. [gnn.py](/home/marcospoza/PD2/PD2-conducIA/src/modelos/p2/gnn.py)
7. [comparar_modelos.py](/home/marcospoza/PD2/PD2-conducIA/src/modelos/p2/comparar_modelos.py)

### Qué hace cada script importante

`p2/preparar_datos.py`
- Lee `datos_final.parquet`.
- Agrega a `zona x ventana de 10 minutos`.
- Calcula:
  - `oferta_inferida`
  - `tasa_exito`
  - `espera_media`
  - `target` relativo por ventana
- Guarda en `data/processed/tlc_clean/problema2/raw/`.

`p2/features.py`
- Añade `tasa_historica`.
- Añade `demanda_p1`, que es una señal derivada del Random Forest de Problema 1.
- Esto conecta los dos problemas: Problema 2 usa información del Problema 1.

`p2/baseline.py`
- Entrena una regresión logística como baseline.

`p2/mlp.py`
- Entrena una red densa para clasificación binaria.

`p2/lightgbm.py`
- Entrena un modelo de boosting.
- Por estructura, parece uno de los candidatos más serios del módulo.

`p2/gnn.py`
- Construye un grafo de zonas origen-destino.
- Añade propagación entre zonas vecinas antes de clasificar.
- Es el modelo más experimental/sofisticado.

`p2/comparar_modelos.py`
- Compara métricas de los modelos entrenados.

### Qué sale de Problema 2

Lo más importante que produce es:

- datos raw agregados en `data/processed/tlc_clean/problema2/raw/`
- features en `data/processed/tlc_clean/problema2/features/`
- modelos en `models/problema2/`
- resultados y plots en `reports/problema2/`

### Scripts que parecen antiguos o menos usados

Dentro de [src/modelos/problema2](/home/marcospoza/PD2/PD2-conducIA/src/modelos/problema2) hay varios scripts que parecen de una versión previa:

`preparar_dataset_p2.py`
- Parece una preparación inicial antigua.
- Genera `dataset_p2.parquet`, pero no coincide con el pipeline actual de `p2/preparar_datos.py`.

`division_datos.py`
- Probablemente pertenece a ese flujo antiguo.

`baseline_p2.py`
- Parece baseline de la versión previa.

`modelo1_mlp.py`
- Parece un experimento antiguo de MLP "cascading".
- Usa placeholders y una lógica distinta del pipeline actual.

`generar_tasas_p2.py`
- Suena a utilidad intermedia antigua.

`verificar_datos.py`
- Parece script de comprobación, no parte del pipeline principal.

### Resumen corto de Problema 2

Problema 2 es un problema de recomendación/clasificación de zonas buenas para el conductor, trabajando en ventanas de 10 minutos.  
La ruta actual más clara parece ser:

`datos_final.parquet -> p2/preparar_datos.py -> p2/features.py -> baseline/mlp/lightgbm/gnn -> comparar`

La idea más importante es esta:
- Problema 1 predice demanda.
- Problema 2 reutiliza esa señal como `demanda_p1` para ayudar a decidir qué zonas son mejores.

---

## Resumen General

### Qué parece vigente ahora mismo

`Problema 1`
- pipeline principal: `preparar_datos.py`, `features.py`, `baseline.py`, `lstm.py`
- análisis: `comparar_modelos.py`, `analizar_predicciones.py`

`Problema 2`
- pipeline principal: carpeta `src/modelos/p2`
- no parece que la carpeta `src/modelos/problema2` sea la versión principal actual

### Qué parece legado o sobrante

`Problema 1`
- `samplear_datos.py`
- `division_datos.py`
- `agregacion.py`
- `modelo1_lstm.py`

`Problema 2`
- casi toda la carpeta `src/modelos/problema2/` parece versión anterior o paralela

### En una sola frase

Ahora mismo el repo parece estar evolucionando desde pipelines antiguos a otros más directos:
- en Problema 1, el flujo nuevo evita el sampleo y trabaja agregando directamente;
- en Problema 2, la carpeta activa parece ser `p2`, no `problema2`;
- y ambos problemas están conectados porque P2 aprovecha la predicción de demanda de P1.

---

## Nota importante

Esto está deducido leyendo el código actual del repo.  
Si hay scripts que sí seguís usando en clase o en vuestro flujo real aunque parezcan antiguos, dímelo y te actualizo el documento para que refleje exactamente vuestra forma de trabajo.
