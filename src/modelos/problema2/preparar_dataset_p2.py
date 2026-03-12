"""Hemos añadido la columna 'oferta_inferida' que representa 
cuántos taxis terminaron un viaje en esa misma zona en los 
20 min anteriores.

Hemos añadido la columna 'ventana_inicio' que es un redondeo de la hroa de inciio
en bloques de 10 min. Nos sirve para agrupar los datos y que el modelo
aprenda patrones por tramos horarios en lugar de segundos exactos."""

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import os
import glob

def preparar_datos():
    input_path = 'data/processed/tlc_clean/datos_final.parquet'
    output_dir = 'data/processed/tlc_clean/problema2'
    output_path = os.path.join(output_dir, 'dataset_p2.parquet')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Calcular la oferta global (usando menos memoria)
    print("Calculando oferta inferida...")
    # Leemos solo lo mínimo necesario para la oferta
    table_oferta = pq.read_table(input_path, columns=['destino_id', 'fecha_fin'])
    df_oferta = table_oferta.to_pandas()
    df_oferta['ventana_fin'] = pd.to_datetime(df_oferta['fecha_fin']).dt.floor('20min')
    
    oferta = df_oferta.groupby(['destino_id', 'ventana_fin']).size().reset_index(name='oferta_inferida')
    del df_oferta, table_oferta # Limpieza inmediata
    
    # 2. Procesar y Guardar de forma incremental (Evita el Killed final)
    print("Procesando bloques y guardando...")
    parquet_file = pq.ParquetFile(input_path)
    writer = None
    
    for i in range(parquet_file.num_row_groups):
        chunk = parquet_file.read_row_group(i).to_pandas()
        
        # Asegurar formato de fecha para el merge
        chunk['fecha_inicio'] = pd.to_datetime(chunk['fecha_inicio'])
        chunk['ventana_inicio'] = chunk['fecha_inicio'].dt.floor('10min')
        
        # Merge con la oferta calculada
        chunk_procesado = chunk.merge(
            oferta, 
            left_on=['origen_id', 'ventana_inicio'], 
            right_on=['destino_id', 'ventana_fin'], 
            how='left'
        )
        
        # Limpieza de nulos y tipos
        chunk_procesado['oferta_inferida'] = chunk_procesado['oferta_inferida'].fillna(0).astype('float32')
        
        # MUY IMPORTANTE: Eliminar columnas basura del merge para evitar conflictos de tipos
        #destino_id y ventana_fin vienen de la tabla 'oferta' y ya no sirven.
        columnas_sobrantes = ['destino_id', 'ventana_fin']
        chunk_procesado = chunk_procesado.drop(columns=[c for c in columnas_sobrantes if c in chunk_procesado.columns])
        
        # Guardado incremental: Escribimos al archivo final bloque a bloque
        table = pa.Table.from_pandas(chunk_procesado)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
        
        print(f" Bloque {i+1}/{parquet_file.num_row_groups} procesado.")
        del chunk, chunk_procesado, table # Liberar memoria

    if writer:
        writer.close()
    
    print(f"\n¡ÉXITO! Dataset de P2 guardado en: {output_path}")

if __name__ == "__main__":
    preparar_datos()