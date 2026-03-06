"""Para preparar el dataset final de ConducIA, he implementado un proceso de limpieza de alta eficiencia diseñado para datasets masivos (2 millones de filas) evitando el desbordamiento de memoria RAM mediante procesamiento por bloques.

La estrategia de limpieza aplicada se basa en tres pilares:

Eliminación de redundancia: Se ha retirado la columna tipo_pago al no ser relevante para la predicción de propinas.

Imputación inteligente de datos nulos: * Pasajeros: Se ha realizado una imputación basada en la distribución de probabilidad original de la muestra, manteniendo la integridad estadística del dataset.

Propinas: He aplicado un filtro condicional de doble capa:

Los valores NaN se sustituyen por la mediana calculada sobre los registros positivos.

Los 'falsos ceros' (casos donde el sistema registra $0 en viajes de larga distancia, indicando un error de registro en lugar de una intención real de no dar propina) se detectan mediante un umbral de total_amount > $10 y se corrigen aplicando también la mediana. Los ceros en viajes de bajo coste se conservan como comportamientos legítimos.

Este enfoque asegura que el modelo de IA aprenda de patrones de propina coherentes, eliminando el ruido técnico sin sesgar el comportamiento real de los usuarios en trayectos cortos."""

import pandas as pd
import numpy as np
import os
import pyarrow.parquet as pq
import pyarrow as pa

# --- CONFIGURACIÓN ---
BASE_DIR = r'/mnt/c/Users/carla/Desktop/PD2/PD2-conducIA'
DIRECTORIO_DATOS = os.path.join(BASE_DIR, 'data', 'processed', 'tlc_clean')
INPUT_FILE = os.path.join(DIRECTORIO_DATOS, 'dataset_final.parquet')
OUTPUT_FILE = os.path.join(DIRECTORIO_DATOS, 'datos_final.parquet')

def limpiar_extremo_ram(ruta_input, ruta_output):
    if not os.path.exists(ruta_input):
        print(f"❌ Error: No se encuentra {ruta_input}")
        return

    print(f"📖 Modo supervivencia: Imputando falsos ceros...")
    
    parquet_file = pq.ParquetFile(ruta_input)
    # Usamos una muestra para obtener la mediana de las propinas reales
    df_mini_sample = parquet_file.read_row_group(0).to_pandas()
    mediana_val = df_mini_sample[df_mini_sample['propina'] > 0]['propina'].median()
    dist_pasajeros = df_mini_sample['num_pasajeros'].value_counts(normalize=True)
    
    print(f"✅ Mediana calculada para imputar: ${mediana_val:.2f}")
    del df_mini_sample 

    writer = None
    
    try:
        for i in range(parquet_file.num_row_groups):
            df_chunk = parquet_file.read_row_group(i).to_pandas()
            
            # --- TRANSFORMACIONES ---
            if 'tipo_pago' in df_chunk.columns:
                df_chunk = df_chunk.drop(columns=['tipo_pago'])
            
            # FILTRO INTELIGENTE:
            # 1. Identificamos los "falsos ceros": propina es 0 Y el viaje fue mayor a $10
            # (Asegúrate de que 'total_amount' sea el nombre correcto de tu columna de precio)
            if 'propina' in df_chunk.columns and 'total_amount' in df_chunk.columns:
                es_falso_cero = (df_chunk['propina'] == 0) & (df_chunk['total_amount'] > 10)
                df_chunk.loc[es_falso_cero, 'propina'] = mediana_val
                
            # 2. Imputamos los Nulos (NaN) que ya existían
            if 'propina' in df_chunk.columns:
                df_chunk['propina'] = df_chunk['propina'].fillna(mediana_val)
                
            # 3. Limpieza de pasajeros
            if 'num_pasajeros' in df_chunk.columns:
                mask = df_chunk['num_pasajeros'].isna()
                if mask.any():
                    df_chunk.loc[mask, 'num_pasajeros'] = np.random.choice(
                        dist_pasajeros.index, size=mask.sum(), p=dist_pasajeros.values
                    )

            table = pa.Table.from_pandas(df_chunk)
            if writer is None:
                writer = pq.ParquetWriter(ruta_output, table.schema)
            writer.write_table(table)
            
            print(f"⏳ Bloque {i+1}/{parquet_file.num_row_groups} procesado.")
            del df_chunk 
            
    finally:
        if writer:
            writer.close()
            print(f"🚀 ¡ÉXITO! Archivo guardado en: {ruta_output}")

if __name__ == "__main__":
    limpiar_extremo_ram(INPUT_FILE, OUTPUT_FILE)