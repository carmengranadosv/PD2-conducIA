"""
DESCRIPCIÓN:
    Este script es el núcleo de preparación de datos para el Problema 4 (Tráfico y Clima).
    Transforma el archivo 'datos_final.parquet' en conjuntos de entrenamiento, validación y prueba.

FLUJO DE TRABAJO:
    1. CARGA OPTIMIZADA: Solo columnas necesarias y tipos de datos ligeros.
    2. FILTRADO: Velocidades entre 2 y 85 mph.
    3. CLIMA: Imputación de nulos y creación de flags (hay_lluvia/nieve).
    4. SPLIT TEMPORAL TRIPLE:
       - Train: Dic 2024 - Ago 2025.
       - Validation: Sep 2025 (Para ajuste de modelos).
       - Test: Oct 2025 - Nov 2025 (Examen final).
    5. GUARDADO: Almacena en 'data/processed/tlc_clean/problema4/'.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'tlc_clean', 'datos_final.parquet')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'tlc_clean', 'problema4')

# Crear el directorio de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

def flujo_preprocesamiento_base():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"No se encuentra el archivo en: {INPUT_PATH}")
    
    # 1. CARGA OPTIMIZADA (Seleccionamos columnas para ahorrar RAM de entrada)
    print("1. Cargando columnas seleccionadas...")
    cols_interes = [
        'fecha_inicio', 'origen_id', 'destino_id', 'velocidad_mph', 
        'temp_c', 'precipitation', 'viento_kmh', 'lluvia', 'nieve', 
        'es_festivo', 'num_eventos', 'mes_num', 'dia_semana', 
        'es_fin_semana', 'hora_sen', 'hora_cos', 'tipo_vehiculo', 'franja_horaria'
    ]
    
    df = pd.read_parquet(INPUT_PATH, columns=cols_interes)
    
    # 2. FILTRADO DE CALIDAD
    print("2. Filtrando ruido y optimizando tipos de datos...")
    df = df[(df['velocidad_mph'] >= 2) & (df['velocidad_mph'] <= 85)]
    
    # Optimizar RAM: Convertir float64 a float32 e int64 a int32
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')

    # 3. TRATAMIENTO DE CLIMA
    print("3. Procesando variables climáticas...")
    cols_clima = ['temp_c', 'precipitation', 'viento_kmh', 'lluvia', 'nieve']
    df[cols_clima] = df[cols_clima].fillna(0)
    
    df['hay_lluvia'] = (df['lluvia'] > 0).astype('int8')
    df['hay_nieve'] = (df['nieve'] > 0).astype('int8')

    return df

def realizar_split_y_guardar(df):
    """
    Divide en 3 sets cronológicos y los guarda en disco.
    """
    print("4. Realizando Split Temporal Triple (Train / Val / Test)...")
    
    # Definimos cortes
    # Mes 10 y 11 -> Test
    # Mes 9 -> Validación
    # Resto (12, 1-8) -> Train
    
    df_test = df[df['mes_num'].isin([10, 11])].copy()
    df_val = df[df['mes_num'] == 9].copy()
    df_train = df[~df['mes_num'].isin([9, 10, 11])].copy()
    
    # Liberar memoria del dataframe original
    del df
    
    print(f"   > Train: {len(df_train):,} filas")
    print(f"   > Val:   {len(df_val):,} filas")
    print(f"   > Test:  {len(df_test):,} filas")
    
    # 5. GUARDADO FÍSICO
    print(f"5. Guardando archivos en {OUTPUT_DIR}...")
    df_train.to_parquet(os.path.join(OUTPUT_DIR, 'train_p4.parquet'), index=False)
    df_val.to_parquet(os.path.join(OUTPUT_DIR, 'val_p4.parquet'), index=False)
    df_test.to_parquet(os.path.join(OUTPUT_DIR, 'test_p4.parquet'), index=False)

if __name__ == "__main__":
    try:
        datos = flujo_preprocesamiento_base()
        realizar_split_y_guardar(datos)
        print("\n[ÉXITO] Archivos generados y guardados correctamente.")
    except Exception as e:
        print(f"\n[ERROR] Ocurrió un problema: {e}")