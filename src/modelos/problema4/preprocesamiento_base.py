"""
DESCRIPCIÓN:
    Este script es el núcleo de preparación de datos para el Problema 4 (Tráfico y Clima).
    Transforma el archivo 'datos_final.parquet' en conjuntos de entrenamiento y prueba.

FLUJO DE TRABAJO:
    1. CARGA CRONOLÓGICA: Lectura del Parquet (Dic 2024 - Nov 2025).
    2. FILTRADO DE CALIDAD: Eliminación de ruidos (velocidades < 2mph o > 85mph).
    3. TRATAMIENTO DE CLIMA: Imputación de nulos y creación de indicadores (Flags)
       de lluvia y nieve.
    4. SELECCIÓN DE FEATURES: Mantenimiento de variables espaciales (IDs), climáticas,
       eventos y cíclicas, descartando solo variables económicas y post-viaje.
    5. SPLIT TEMPORAL: Train (Dic 2024 - Sep 2025) y Test (Oct - Nov 2025).
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
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
    print("2. Filtrando ruido y velocidades imposibles (2-85 mph)...")
    df = df[(df['velocidad_mph'] >= 2) & (df['velocidad_mph'] <= 85)].copy()
    
    # 3. TRATAMIENTO ATMOSFÉRICO (Core del Problema 4)
    print("3. Estandarizando variables climáticas...")
    # Rellenamos nulos en clima con 0 (asumimos ausencia de fenómeno si no hay registro)
    cols_clima = ['temp_c', 'precipitation', 'viento_kmh', 'lluvia', 'nieve']
    df[cols_clima] = df[cols_clima].fillna(0)
    
    # Creamos 'Flags' binarios para capturar el impacto visual/físico del clima
    df['hay_lluvia'] = (df['lluvia'] > 0).astype(int)
    df['hay_nieve'] = (df['nieve'] > 0).astype(int)

    # 4. SELECCIÓN DE VARIABLES (Features finales acordadas)
    print("4. Seleccionando columnas útiles para los modelos...")
    # Hemos incluido origen/destino para Grafos (ST-GCN) y eventos para el tráfico
    features_modelo = [
        # Identificadores Temporales y Espaciales
        'fecha_inicio', 'origen_id', 'destino_id', 
        
        # Objetivo (Target)
        'velocidad_mph', 
        
        # Clima (Problema 4)
        'temp_c', 'precipitation', 'viento_kmh', 'lluvia', 'nieve', 'hay_lluvia', 'hay_nieve',
        
        # Calendario y Eventos
        'es_festivo', 'num_eventos', 'mes_num', 'dia_semana', 'es_fin_semana',
        
        # Variables de tiempo cíclico
        'hora_sen', 'hora_cos',
        
        # Categorías (serán procesadas en cada modelo)
        'tipo_vehiculo', 'franja_horaria'
    ]
    
    # Filtramos el DF solo con estas columnas
    df = df[features_modelo].copy()
    
    print(f"   > Preprocesamiento base listo. Registros: {len(df):,}")
    return df

def realizar_split_temporal(df):
    """
    Divide el dataset cronológicamente.
    Train: Mes 12 (Dic 2024) + Meses 1 al 9 (2025).
    Test: Meses 10 y 11 (Octubre y Noviembre 2025).
    """
    print("5. Realizando split temporal (Train: Dic -> Sep | Test: Oct-Nov)...")
    
    # Octubre y Noviembre son los meses más "futuros" de nuestro dataset
    meses_test = [10, 11]
    
    df_train = df[~df['mes_num'].isin(meses_test)].copy()
    df_test = df[df['mes_num'].isin(meses_test)].copy()
    
    print(f"   > Entrenamiento (Train): {len(df_train):,} registros.")
    print(f"   > Evaluación (Test): {len(df_test):,} registros.")
    
    return df_train, df_test

if __name__ == "__main__":
    # --- PRUEBA DE EJECUCIÓN ---
    try:
        datos_base = flujo_preprocesamiento_base()
        train_set, test_set = realizar_split_temporal(datos_base)
        print("\n[ÉXITO] Preprocesamiento base finalizado correctamente.")
    except Exception as e:
        print(f"\n[ERROR] Ocurrió un problema: {e}")