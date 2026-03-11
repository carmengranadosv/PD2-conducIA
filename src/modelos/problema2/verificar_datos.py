import pandas as pd

def verificar():
    # 1. Cargar el dataset que acabas de generar
    path = 'data/processed/problema2/dataset_p2.parquet'
    df = pd.read_parquet(path)
    
    # 2. Ver las primeras filas
    print("--- Primeras 5 filas del dataset ---")
    print(df.head())
    
    # 3. Ver columnas y tipos de datos (para asegurar que los números sean números)
    print("\n--- Estructura del dataset ---")
    print(df.info())
    
    # 4. Ver si hay valores vacíos (importante porque usamos fillna(0))
    print("\n--- Conteo de valores nulos ---")
    print(df.isnull().sum())

if __name__ == "__main__":
    verificar()