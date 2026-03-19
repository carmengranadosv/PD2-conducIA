"""
MÓDULO: generar_grafo.py
DESCRIPCIÓN: Crea la matriz de adyacencia necesaria para el ST-GCN.
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'tlc_clean', 'problema4')

def crear_matriz_adyacencia():
    print("Generando estructura de grafo (Zonas de NY)...")
    
    # Cargamos una muestra de train para ver qué IDs de zona existen realmente
    df = pd.read_parquet(os.path.join(DATA_DIR, 'train_p4.parquet'), columns=['origen_id', 'destino_id'])
    
    # Obtenemos todos los IDs únicos de zonas
    nodos = sorted(pd.concat([df['origen_id'], df['destino_id']]).unique())
    num_nodos = len(nodos)
    id_map = {id_zona: i for i, id_zona in enumerate(nodos)}
    
    # Creamos matriz vacía
    adj_matrix = np.zeros((num_nodos, num_nodos), dtype='float32')
    
    # Lógica de conexión: Conectamos zonas si hay flujo constante entre ellas
    # Esto simula la contigüidad física en ausencia de un mapa GIS
    print("Calculando conexiones entre zonas...")
    conexiones = df.groupby(['origen_id', 'destino_id']).size().reset_index(name='conteo')
    
    # Solo conectamos si hay un flujo significativo (umbral)
    umbral = conexiones['conteo'].quantile(0.75) 
    
    for _, row in conexiones[conexiones['conteo'] > umbral].iterrows():
        idx_o = id_map[row['origen_id']]
        idx_d = id_map[row['destino_id']]
        adj_matrix[idx_o, idx_d] = 1
        adj_matrix[idx_d, idx_o] = 1 # Grafo no dirigido
        
    # Guardar matriz y mapa de IDs
    np.save(os.path.join(DATA_DIR, 'adj_matrix.npy'), adj_matrix)
    pd.Series(id_map).to_json(os.path.join(DATA_DIR, 'mapa_nodos.json'))
    
    print(f"Grafo generado: {num_nodos} nodos (zonas).")
    return adj_matrix, id_map

if __name__ == "__main__":
    crear_matriz_adyacencia()