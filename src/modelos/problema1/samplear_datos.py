# src/modelos/problema1/samplear_datos.py

"""
Samplea por FILAS (row groups) sin cargar todo en RAM
Método optimizado para máquinas con poca RAM
"""

import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

PARQUET_PATH = "data/processed/tlc_clean/datos_final.parquet"
OUTPUT_PATH = "data/processed/tlc_clean/datos_sample_10pct.parquet"
SAMPLE_FRACTION = 0.10  # 10% (REDUCIDO para RAM)

print("="*60)
print(" SAMPLING EFICIENTE (para RAM limitada)")
print("="*60)

# Verificar archivo existe
if not Path(PARQUET_PATH).exists():
    raise FileNotFoundError(f" No existe: {PARQUET_PATH}")

print(f"\n Archivo: {PARQUET_PATH}")
size_gb = Path(PARQUET_PATH).stat().st_size / 1024**3
print(f"   Tamaño: {size_gb:.2f} GB")
print(f"   Sample: {SAMPLE_FRACTION*100}%")

# ============================================================
# PASO 1: Leer metadata (NO carga datos)
# ============================================================

print(f"\n Leyendo metadata...")

parquet_file = pq.ParquetFile(PARQUET_PATH)
n_total = parquet_file.metadata.num_rows
n_row_groups = parquet_file.num_row_groups

print(f" Metadata:")
print(f"   Total filas: {n_total:,}")
print(f"   Row groups: {n_row_groups}")
print(f"   Filas/group: {n_total // n_row_groups:,}")

# ============================================================
# PASO 2: Samplear POR ROW GROUPS (chunk a chunk)
# ============================================================

n_sample_target = int(n_total * SAMPLE_FRACTION)
print(f"\n Objetivo: {n_sample_target:,} filas ({SAMPLE_FRACTION*100}%)")

np.random.seed(42)

# Calcular cuántos row groups samplear
n_groups_to_sample = max(1, int(n_row_groups * SAMPLE_FRACTION))

print(f"\n Estrategia:")
print(f"   Samplearemos {n_groups_to_sample} de {n_row_groups} row groups")

# Seleccionar row groups aleatorios
step = n_row_groups // n_groups_to_sample
selected_groups = np.arange(0, n_row_groups, step)[:n_groups_to_sample]
selected_groups = np.sort(selected_groups)

print(f"   Row groups seleccionados: {list(selected_groups[:10])}{'...' if len(selected_groups) > 10 else ''}")

# ============================================================
# PASO 3: Leer SOLO los row groups seleccionados
# ============================================================

print(f"\n Leyendo row groups seleccionados...")

tables = []
total_rows = 0

for i, group_idx in enumerate(selected_groups):
    print(f"\r   Procesando {i+1}/{len(selected_groups)} groups...", end='', flush=True)
    
    # Leer UN SOLO row group (pequeño, cabe en RAM)
    table = parquet_file.read_row_group(group_idx)
    
    tables.append(table)
    total_rows += len(table)

print(f"\n Leídos {len(tables)} row groups = {total_rows:,} filas")

# ============================================================
# PASO 4: Combinar y samplear filas individuales
# ============================================================

print(f"\n Combinando tablas...")

# Concatenar todos los row groups
combined_table = pa.concat_tables(tables)

print(f" Tabla combinada: {len(combined_table):,} filas")

# Si tenemos MÁS filas de las necesarias, samplear
if len(combined_table) > n_sample_target:
    print(f"\n Sampleando {n_sample_target:,} filas de {len(combined_table):,}...")
    
    indices = np.random.choice(len(combined_table), n_sample_target, replace=False)
    indices = np.sort(indices)
    
    final_table = combined_table.take(indices)
else:
    final_table = combined_table

print(f" Sample final: {len(final_table):,} filas")

# ============================================================
# PASO 5: Guardar
# ============================================================

print(f"\n Guardando: {OUTPUT_PATH}")

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

pq.write_table(
    final_table, 
    OUTPUT_PATH, 
    compression='snappy'
)

# Verificar
if Path(OUTPUT_PATH).exists():
    size_mb = Path(OUTPUT_PATH).stat().st_size / 1024**2
    print(f" Guardado: {size_mb:.1f} MB")
    
    # Verificar contenido
    verify = pq.read_table(OUTPUT_PATH)
    print(f" Verificado: {len(verify):,} filas")
    
    del verify
else:
    print(" Error al guardar")

print("\n" + "="*60)
print(" COMPLETADO")
print("="*60)

print(f"\n📊 Resumen:")
print(f"   Original:  {n_total:,} filas ({size_gb:.2f} GB)")
print(f"   Sample:    {len(final_table):,} filas (~{size_mb:.0f} MB)")
print(f"   Reducción: {(1 - len(final_table)/n_total)*100:.1f}%")

print(f"\n Siguiente paso:")
print(f"   Modificar division_datos.py línea 17:")
print(f"   PARQUET_PATH = 'data/processed/tlc_clean/datos_sample_5pct.parquet'")