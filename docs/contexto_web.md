# Contexto historico para la web

La web no debe cargar los datasets completos de viajes al arrancar. Son demasiado grandes para un despliegue normal y, ademas, una prediccion solo necesita un resumen historico comparable. Por eso se generan dos tablas pequeñas en:

```text
data/processed/tlc_clean/contexto_web/contexto_p2.parquet
data/processed/tlc_clean/contexto_web/contexto_p5.parquet
```

Estas tablas se generan con:

```bash
uv run python despliegue/preparar_contexto_web.py --data-root data --batch-size 500000
```

Docker no necesita copiar estos Parquet dentro de la imagen. Con el volumen:

```bash
docker run -p 8000:8000 -v "$(pwd)/data:/app/data" conducia-app
```

la app los ve dentro del contenedor en:

```text
/app/data/processed/tlc_clean/contexto_web/
```

## Que contiene `contexto_p2.parquet`

Este archivo alimenta el panel Taxi y los modelos P1/P2.

Cada fila representa un patron historico agregado por:

```text
origen_id + mes_num + dia_semana + hora
```

Columnas principales:

```text
origen_id
mes_num
dia_semana
hora
n_viajes
espera_media
hora_sen
hora_cos
temp_c
precipitation
viento_kmh
lluvia
nieve
es_festivo
num_eventos
oferta_inferida
tasa_historica
demanda_p1
_n_contexto
```

Uso en Taxi:

- `oferta_inferida`, `tasa_historica`, clima y eventos ayudan a construir las features del Random Forest P1.
- `n_viajes`, `espera_media`, `oferta_inferida`, `tasa_historica` y `demanda_p1` alimentan la MLP P2.
- `_n_contexto` indica cuantas filas historicas se agregaron para crear ese patron.

Ejemplo conceptual:

```text
Zona 230, abril, viernes, 18:00
```

La app busca el historico comparable de:

```text
origen_id = 230
mes_num = 4
dia_semana = 4
hora = 18
```

Y usa las medias historicas de esa combinacion.

## Que contiene `contexto_p5.parquet`

Este archivo alimenta el panel VTC y los modelos P4/P5.

Cada fila tambien representa:

```text
origen_id + mes_num + dia_semana + hora
```

Columnas principales:

```text
origen_id
mes_num
dia_semana
hora
num_pasajeros
distancia
duracion_min
velocidad_mph
precio_base
precio_total_est
espera_min
temp_c
precipitation
viento_kmh
lluvia
nieve
es_festivo
num_eventos
es_fin_semana
hora_sen
hora_cos
rentabilidad_base_min
trafico_denso
_n_contexto
tipo_vehiculo
origen_zona
origen_barrio
evento_tipo
franja_horaria
```

Uso en VTC:

- P4 predice velocidad usando zona, clima, eventos, hora y dia.
- P5 predice propina usando precio introducido por el usuario mas contexto historico de origen, clima, duracion, distancia, espera y trafico.
- Las columnas categoricas (`origen_zona`, `origen_barrio`, `evento_tipo`, `franja_horaria`) se calculan con la moda historica.

Ejemplo conceptual:

```text
Origen 161, abril, viernes, 18:00, precio base 25.50
```

La app busca:

```text
origen_id = 161
mes_num = 4
dia_semana = 4
hora = 18
```

Despues combina ese contexto historico con el precio que escribe el usuario.

## Busqueda con fallbacks

Si no existe una combinacion exacta, la app no usa una fila aleatoria. Busca de mas especifico a mas general:

```text
1. zona + mes + dia_semana + hora
2. zona + mes + hora
3. zona + dia_semana + hora
4. zona + hora
5. zona
```

Esto hace que las predicciones tengan mas sentido que antes, porque se basan en patrones historicos comparables y no en las primeras 100.000 filas del Parquet.

## Como comprobar que funciona

Ejecuta:

```bash
uv run python despliegue/verificar_contexto_web.py --zona 230 --fecha 2026-04-03 --hora 18
```

El script muestra:

- que contexto P2 usaria Taxi,
- que contexto P5 usaria VTC,
- que nivel de fallback ha aplicado,
- cuantas filas historicas forman ese patron (`_n_contexto`),
- y algunas variables clave que deberian cambiar al variar zona, hora o fecha.

Prueba varias combinaciones:

```bash
uv run python despliegue/verificar_contexto_web.py --zona 230 --fecha 2026-04-03 --hora 18
uv run python despliegue/verificar_contexto_web.py --zona 132 --fecha 2026-04-03 --hora 18
uv run python despliegue/verificar_contexto_web.py --zona 230 --fecha 2026-04-03 --hora 8
```

Si los valores cambian, la app esta usando contexto historico distinto.

## Validaciones de sentido comun

Se hicieron comprobaciones sobre los contextos ya generados para confirmar que los patrones agregados son razonables. La variable principal usada para esta lectura es `n_viajes` media del contexto P2, agrupada por zona y hora.

Perfil medio por zonas del desplegable:

| zona | 03:00 | 08:00 | 18:00 | 22:00 | 23:00 |
|---|---:|---:|---:|---:|---:|
| Times Sq/Theatre District (230) | 6.7 | 19.6 | 36.4 | 45.6 | 33.0 |
| Midtown Center (161) | 3.7 | 15.4 | 55.3 | 40.3 | 28.6 |
| East Village (79) | 15.4 | 12.6 | 26.1 | 38.4 | 36.6 |
| West Village (249) | 9.0 | 11.2 | 22.1 | 31.6 | 29.6 |
| Upper East Side South (237) | 2.0 | 24.1 | 43.5 | 24.7 | 15.7 |
| JFK Airport (132) | 6.5 | 20.4 | 44.1 | 50.5 | 49.0 |
| LaGuardia Airport (138) | 2.4 | 21.2 | 46.7 | 45.3 | 47.6 |

Lectura:

- Zonas de trabajo/centro como Midtown, Times Sq y Upper East Side South suben claramente por la mañana y al final de la tarde.
- Zonas de ocio como East Village y West Village tienen mas peso por la noche que por la mañana.
- Los aeropuertos mantienen demanda alta por la tarde/noche, algo esperable por llegadas y salidas.
- En madrugada bajan mucho Midtown y Upper East Side South, mientras que East Village conserva mas actividad nocturna.

Tambien se reviso el contexto P5 de velocidad/trafico:

- Times Sq pasa aproximadamente de 18.6 mph a las 03:00 a 8.2 mph a las 18:00.
- Midtown Center pasa aproximadamente de 16.9 mph a las 03:00 a 8.1 mph a las 18:00.
- JFK y LaGuardia mantienen velocidades mas altas y trafico denso bajo, coherente con trayectos mas largos y menos callejeo de Manhattan.

Ejemplos de zonas con muy baja actividad a las 03:00:

```text
Crotona Park
Great Kills Park
Astoria Park
Battery Park
City Island
Pelham Bay Park
```

Esto refuerza que el contexto no esta plano: zonas residenciales/parques o zonas menos centrales caen en madrugada, mientras ocio, centro y aeropuertos conservan mas actividad.
