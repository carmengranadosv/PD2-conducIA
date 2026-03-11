"""
DESCRIPCIÓN:
    Este script es el núcleo de preparación de datos para el Problema 4 (Tráfico y Clima).
    Transforma el archivo 'datos_final.parquet' en conjuntos de entrenamiento y prueba.

FLUJO DE TRABAJO QUE REALIZA ESTE SCRIPT:
    1. CARGA CRONOLÓGICA: Lectura del Parquet (Dic 2024 - Nov 2025).
    2. FILTRADO DE CALIDAD: Eliminación de ruidos (velocidades < 2mph o > 85mph).
    3. TRATAMIENTO DE CLIMA: Imputación de nulos y creación de indicadores (Flags)
       de lluvia y nieve para facilitar el aprendizaje de los modelos.
    4. SELECCIÓN DE FEATURES: Limpieza de variables "post-viaje" (propinas, precios)
       y mantenimiento de variables cíclicas (Sen/Cos) y espaciales (IDs).
    5. SPLIT TEMPORAL: División estricta por meses:
       - Entrenamiento (Train): Diciembre 2024 hasta Septiembre 2025.
       - Evaluación (Test): Octubre y Noviembre 2025.

NOTAS:
    - Este archivo garantiza que todos los modelos compitan bajo las mismas condiciones.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
# Buscamos la raíz del proyecto (PD2-conducIA) subiendo 3 niveles desde este script
BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'tlc_clean', 'datos_final.parquet')

def flujo_preprocesamiento_base():
    """
    Realiza la limpieza e ingeniería de variables común a todos los modelos.
    """
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"No se encuentra el archivo Parquet en: {PARQUET_PATH}")
    
    print("1. Cargando datos (Dic 2024 - Nov 2025)...")
    df = pd.read_parquet(PARQUET_PATH)
    
    # 2. LIMPIEZA DE CALIDAD (Outliers de Velocidad)
    # Filtramos velocidades extremas que suelen ser errores de GPS o atascos estáticos.
    # El rango 2-85 mph es el estándar para tráfico urbano real en NY.
    print("2. Filtrando ruido y velocidades imposibles...")
    df = df[(df['velocidad_mph'] >= 2) & (df['velocidad_mph'] <= 85)].copy()
    
    # 3. TRATAMIENTO ATMOSFÉRICO (Core del Problema 4)
    print("3. Estandarizando variables climáticas...")
    # Rellenamos nulos en clima con 0 (asumimos que no hubo precipitación si no hay registro)
    cols_clima = ['lluvia', 'nieve', 'precipitation', 'temp_c', 'viento_kmh']
    df[cols_clima] = df[cols_clima].fillna(0)
    
    # Creamos 'Flags' binarios. A veces al tráfico le afecta el 'hecho de que llueva'
    # más que los milímetros exactos. Esto ayuda mucho al Baseline y a la Red Neuronal.
    df['hay_lluvia'] = (df['lluvia'] > 0).astype(int)
    df['hay_nieve'] = (df['nieve'] > 0).astype(int)

    # 4. SELECCIÓN DE VARIABLES (Features)
    print("4. Seleccionando columnas útiles para los modelos...")
    # Mantenemos: IDs (para ST-GCN), Clima (Problema 4) y Cíclicas (Red Neuronal)
    # Eliminamos: propinas, precios y esperas (son datos que no se saben al inicio del viaje)
    features_modelo = [
        'fecha_inicio', 'origen_id', 'destino_id', 'velocidad_mph', 
        'tipo_vehiculo', 'temp_c', 'lluvia', 'nieve', 'hay_lluvia', 
        'hay_nieve', 'viento_kmh', 'es_festivo', 'num_eventos', 
        'mes_num', 'hora', 'dia_semana', 'es_fin_semana', 
        'franja_horaria', 'hora_sen', 'hora_cos'
    ]
    
    df = df[features_modelo].copy()
    
    print(f"   > Preprocesamiento base listo. Registros: {len(df):,}")
    return df

def realizar_split_temporal(df):
    """
    Divide el dataset cronológicamente.
    Train: Mes 12 (Dic 2024) + Meses 1 al 9 (2025).
    Test: Meses 10 y 11 (Octubre y Noviembre 2025).
    """
    print("5. Realizando split temporal (Train: Pasado -> Sep | Test: Oct-Nov)...")
    
    # Definimos los meses finales como nuestro conjunto de evaluación (futuro)
    meses_test = [10, 11]
    
    df_train = df[~df['mes_num'].isin(meses_test)].copy()
    df_test = df[df['mes_num'].isin(meses_test)].copy()
    
    print(f"   > Entrenamiento (Train): {len(df_train):,} registros.")
    print(f"   > Evaluación (Test): {len(df_test):,} registros.")
    
    return df_train, df_test

if __name__ == "__main__":
    # --- PRUEBA DE EJECUCIÓN ---
    # Este bloque solo se ejecuta si lanzas el script directamente.
    datos_base = flujo_preprocesamiento_base()
    train_set, test_set = realizar_split_temporal(datos_base)
    
    print("\n[ÉXITO] Preprocesamiento base finalizado correctamente.")