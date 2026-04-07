# Funcionalidad: demanda por zona y franja horaria

Esta funcionalidad clasifica cada combinacion `zona x franja_horaria` en nivel de demanda:
`baja`, `media` o `alta`.

El script esta en:

```bash
src/funcionalidades/demanda_zona_franja.py
```

## Generar resultados

Desde la raiz del proyecto:

```bash
uv run python -m src.funcionalidades.demanda_zona_franja
```

Esto crea o actualiza los archivos en:

```bash
reports/funcionalidades/
```

Archivos principales:

- `demanda_zona_franja.csv`
- `demanda_zona_franja.parquet`
- `top20_demanda_alta.csv`
- `demanda_zona_franja_metadata.json`

## Consultar resultados

Puedes buscar indicando uno o varios filtros. Si no indicas algun dato, el script lo completa en la salida.

Ejemplo por zona y franja:

```bash
uv run python -m src.funcionalidades.demanda_zona_franja --consultar --zona Midtown --franja tarde
```

Ejemplo por nivel y franja:

```bash
uv run python -m src.funcionalidades.demanda_zona_franja --consultar --nivel alta --franja noche --top 5
```

Ejemplo por demanda minima:

```bash
uv run python -m src.funcionalidades.demanda_zona_franja --consultar --demanda-min 250 --top 5
```

Ejemplo por zona usando ID:

```bash
uv run python -m src.funcionalidades.demanda_zona_franja --consultar --zona 161
```

## Filtros disponibles

- `--zona`: ID o texto de zona/barrio. Ejemplos: `161`, `Midtown`, `Queens`, `Airport`.
- `--franja`: `madrugada`, `manana`, `mediodia`, `tarde`, `noche`.
- `--nivel`: `baja`, `media`, `alta`.
- `--demanda-min`: demanda media minima.
- `--demanda-max`: demanda media maxima.
- `--top`: numero maximo de resultados a mostrar.

Nota: en modo consulta debes introducir al menos un filtro.
