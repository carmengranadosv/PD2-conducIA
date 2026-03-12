"""Verifica que los datos con los que vamos a trabajar en el 
Problema 2 se han creado perfectamente.

El dataset cuenta con 35 variables."""

import pandas as pd
import pyarrow.parquet as pq
import os

def verificar_dataset_ligero():
    path = 'data/processed/problema2/dataset_p2.parquet'
    
    if not os.path.exists(path):
        print(f"❌ Error: No se encuentra el archivo en {path}")
        return

    print("--- ✅ VERIFICACIÓN LIGERA DE DATASET P2 ---")
    
    # 1. Leer solo el esquema para ver las columnas
    parquet_file = pq.ParquetFile(path)
    print(f"Columnas detectadas ({len(parquet_file.schema.names)}):")
    print(parquet_file.schema.names)
    
    # 2. Leer solo las primeras 10 filas (esto no consume RAM)
    df_mini = parquet_file.read_row_group(0).to_pandas().head(10)
    
    print("\n--- 📊 ESTADÍSTICAS RÁPIDAS (Primeras filas) ---")
    if 'oferta_inferida' in df_mini.columns:
        print(f"Oferta detectada en primeras filas: {df_mini['oferta_inferida'].unique()}")
    
    if 'propina' in df_mini.columns:
        print(f"Muestra de propinas: {df_mini['propina'].tolist()}")

    print("\n--- 🔍 VISTA PREVIA ---")
    cols_interes = ['fecha_inicio', 'origen_id', 'oferta_inferida', 'propina']
    # Solo mostramos las que existan de esa lista
    cols_existentes = [c for c in cols_interes if c in df_mini.columns]
    print(df_mini[cols_existentes])

if __name__ == "__main__":
    verificar_dataset_ligero()