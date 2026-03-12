import pyarrow.parquet as pq
import pyarrow as pa
import os

def dividir_datos_temporales_eficiente():
    # Ruta de entrada y salida ajustadas
    input_path = 'data/processed/tlc_clean/problema2/dataset_p2.parquet'
    output_dir = 'data/processed/tlc_clean/problema2'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Obtener información del archivo sin cargarlo en RAM
    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows
    
    # Calcular cortes
    train_end = int(total_rows * 0.70)
    val_end = int(total_rows * 0.85)
    
    print(f"Total registros: {total_rows:,}")
    print(f"Rangos: Train [0 - {train_end:,}], Val [{train_end:,} - {val_end:,}], Test [{val_end:,} - {total_rows:,}]")

    # 2. Función para procesar y escribir particiones
    def escribir_particion(idx_inicio, idx_fin, nombre_archivo):
        print(f"Procesando y guardando: {nombre_archivo}...")
        writer = None
        filas_procesadas = 0
        
        for i in range(parquet_file.num_row_groups):
            # Leemos grupo a grupo
            table = parquet_file.read_row_group(i)
            num_filas_grupo = table.num_rows
            
            grupo_inicio = filas_procesadas
            grupo_fin = filas_procesadas + num_filas_grupo
            
            # Verificamos si este grupo tiene intersección con el rango objetivo
            if grupo_fin > idx_inicio and grupo_inicio < idx_fin:
                df = table.to_pandas()
                
                # Calculamos el slice relativo al bloque actual
                rel_inicio = max(0, idx_inicio - grupo_inicio)
                rel_fin = min(num_filas_grupo, idx_fin - grupo_inicio)
                
                df_slice = df.iloc[rel_inicio:rel_fin]
                table_slice = pa.Table.from_pandas(df_slice)
                
                if writer is None:
                    writer = pq.ParquetWriter(os.path.join(output_dir, nombre_archivo), table_slice.schema)
                
                writer.write_table(table_slice)
            
            filas_procesadas = grupo_fin
        
        if writer:
            writer.close()

    # 3. Ejecutar la división
    escribir_particion(0, train_end, 'train.parquet')
    escribir_particion(train_end, val_end, 'val.parquet')
    escribir_particion(val_end, total_rows, 'test.parquet')
    
    print(f"\n¡ÉXITO! Particiones guardadas en: {output_dir}")

if __name__ == "__main__":
    dividir_datos_temporales_eficiente()