"""
DESCRIPCIÓN:
    Este script realiza el preprocesamiento común y centralizado para todos los modelos 
    del Problema 4 (Navegación Basada en Eficiencia y Clima).

OBJETIVO:
    Transformar el dataset crudo 'dataset_final.parquet' en un conjunto de datos limpio
    y estructurado antes de que cada modelo específico (Baseline, 
    LSTM, ST-GCN, Red Neuronal) aplique su propia transformación final.

FLUJO PRINCIPAL:
    1. Carga de datos desde Parquet (optimizado).
    2. Limpieza de Outliers (velocidades imposibles, nulos en clima).
    3. Ingeniería de Variables:
        - Temporales: Extracción de hora, día de la semana, fin de semana.
        - Atmosféricas: Codificación básica de lluvia/nieve.
    4. Split Temporal: División de datos en Entrenamiento (Ene-Oct) y Test (Nov-Dic)
       para asegurar una evaluación realista basada en el tiempo.

NOTAS:
    - Mantener este archivo como la "única fuente de verdad" para los datos de entrada.
    - Los cambios aquí afectarán a todos los modelos del pipeline.
========================================================================================
"""