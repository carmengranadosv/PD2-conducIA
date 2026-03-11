"""Preprocesamkkiento de los datos, tanto el baseline como los modelos 
del problema 2 leerán el archivo que genere este script.
"""

import pandas as pd
import pyarrow.parquet as pq
import os
import glob

def preparar_datos():
    input_path = 'data/processed/tlc_clean/datos_final.parquet'
    temp_dir = 'data/processed/temp_chunks'
    output_dir = 'data/processed/problema2'
    output_path = os.path.join(output_dir, 'dataset_p2.parquet')
    
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Calcular la oferta global
    print("Calculando oferta inferida...")
    table = pq.read_table(input_path, columns=['destino_id', 'fecha_fin'])
    df_oferta = table.to_pandas()
    df_oferta['ventana_fin'] = pd.to_datetime(df_oferta['fecha_fin']).dt.floor('20min')
    oferta = df_oferta.groupby(['destino_id', 'ventana_fin']).size().reset_index(name='oferta_inferida')
    del df_oferta, table 
    
    # 2. Procesar por bloques
    print("Procesando bloques y guardando temporales...")
    parquet_file = pq.ParquetFile(input_path)
    
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=500_000)):
        chunk = batch.to_pandas()
        chunk['fecha_inicio'] = pd.to_datetime(chunk['fecha_inicio'])
        chunk['ventana_inicio'] = chunk['fecha_inicio'].dt.floor('10min')
        
        # Merge evitando columnas duplicadas usando 'on'
        chunk_procesado = chunk.merge(oferta, 
                                     left_on=['origen_id', 'ventana_inicio'], 
                                     right_on=['destino_id', 'ventana_fin'], 
                                     how='left')
        
        # Rellenar nulos
        chunk_procesado['oferta_inferida'] = chunk_procesado['oferta_inferida'].fillna(0)
        
        # Convertir a tipos compatibles con Parquet antes de guardar
        # Esto asegura que no haya objetos de tipo Timestamp mezclados con enteros
        chunk_procesado['oferta_inferida'] = chunk_procesado['oferta_inferida'].astype('float32')
        
        chunk_procesado.to_parquet(os.path.join(temp_dir, f'chunk_{i}.parquet'))
        print(f"Bloque {i} guardado.")

    # 3. Unir todo
    print("Uniendo bloques finales...")
    all_files = glob.glob(os.path.join(temp_dir, "*.parquet"))
    df_final = pd.concat((pd.read_parquet(f) for f in all_files))
    df_final.to_parquet(output_path)
    
    print(f"Dataset de P2 guardado exitosamente en: {output_path}")

if __name__ == "__main__":
    preparar_datos()