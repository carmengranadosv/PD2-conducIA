import pandas as pd
import numpy as np
import os
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parents[3] 
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'tlc_clean', 'datos_finales.parquet')

# Creamos una carpeta específica para los datos del modelo
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'tlc_clean', 'problema5')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Porcentajes de división
TRAIN_PCT = 0.70
VAL_PCT = 0.15
# El Test será el resto (15%)

def preparar_y_dividir_datos(ruta_input, directorio_output):
    print(" Iniciando preparación de datos para modelado del problema 5...")
    
    if not os.path.exists(ruta_input):
        print(f" Error: No se encuentra {ruta_input}")
        return

    # 1. Cargar datos
    print(" Cargando dataset enriquecido...")
    df = pd.read_parquet(ruta_input, engine='pyarrow')
    
    # 2. --- NUEVAS VARIABLES ---
    print(" Creando nuevas variables...")
    
    # Rentabilidad base por minuto
    df['rentabilidad_base_min'] = np.where(
        df['duracion_min'] > 0, 
        df['precio_base'] / df['duracion_min'], 
        0
    ).astype('float32')

    # Tráfico denso (1 si va a menos de 10 mph, 0 en caso contrario)
    df['trafico_denso'] = (df['velocidad_mph'] < 10).astype('int8')    
    
    # 3. Cálculo de índices para el corte temporal
    n_total = len(df)
    idx_train = int(n_total * TRAIN_PCT)
    idx_val = int(n_total * (TRAIN_PCT + VAL_PCT))
    
    print(f" Total de registros procesados: {n_total:,}")
    
    # Mostrar hasta qué fecha llega cada partición ANTES de borrar las fechas
    print(f"   -> Train: {idx_train:,} ({TRAIN_PCT*100:.0f}%) | Fechas: {df['fecha_inicio'].iloc[0].date()} a {df['fecha_inicio'].iloc[idx_train-1].date()}")
    print(f"   -> Val:   {idx_val-idx_train:,} ({VAL_PCT*100:.0f}%) | Fechas: {df['fecha_inicio'].iloc[idx_train].date()} a {df['fecha_inicio'].iloc[idx_val-1].date()}")
    print(f"   -> Test:  {n_total-idx_val:,} ({(1-TRAIN_PCT-VAL_PCT)*100:.0f}%) | Fechas: {df['fecha_inicio'].iloc[idx_val].date()} a {df['fecha_inicio'].iloc[-1].date()}")

    print("Eliminando columnas que el modelo no necesita ...")
    columnas_a_eliminar = ['fecha_inicio', 'fecha_fin']
    columnas_a_eliminar = [col for col in columnas_a_eliminar if col in df.columns]
    df = df.drop(columns=columnas_a_eliminar)
    
    # 4. División (Slicing) de los datos (manteniendo propina dentro)
    df_train = df.iloc[:idx_train]
    df_val = df.iloc[idx_train:idx_val]
    df_test = df.iloc[idx_val:]
    
    # 5. Guardar los sets generados
    print(f"Guardando archivos parquet en: {directorio_output}")
    df_train.to_parquet(os.path.join(directorio_output, 'train.parquet'), index=False)
    df_val.to_parquet(os.path.join(directorio_output, 'val.parquet'), index=False)
    df_test.to_parquet(os.path.join(directorio_output, 'test.parquet'), index=False)
    
    print("¡Datos preparados y guardados con éxito!")

if __name__ == "__main__":
    preparar_y_dividir_datos(INPUT_FILE, OUTPUT_DIR)