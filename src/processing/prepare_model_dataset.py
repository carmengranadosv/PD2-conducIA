import pandas as pd
import numpy as np
import glob
import os
import sys

# Configuración de rutas 
# Si ejecutas desde src/processing, ajusta los ../
INPUT_PATH = "data/processed/tlc_clean/*/*/*.parquet"
OUTPUT_DIR = "data/model_ready"
OUTPUT_FILE = "dataset.parquet"

def cargar_y_unificar(ruta_patron):
    """Carga todos los parquets y les asigna su etiqueta de tipo."""
    archivos = glob.glob(ruta_patron)
    print(f"Detectados {len(archivos)} archivos. Iniciando carga...")
    
    dfs = []
    for f in archivos:
        try:
            df_temp = pd.read_parquet(f)
            
            # Etiquetado automático
            if 'yellow' in f.lower():
                df_temp['tipo'] = 'Yellow Taxi'
            elif 'fhvhv' in f.lower():
                df_temp['tipo'] = 'VTC'
            
            # Opcional: Si el dataset es GIGANTE, podrías hacer un sample aquí
            # df_temp = df_temp.sample(frac=0.5) 
            
            dfs.append(df_temp)
        except Exception as e:
            print(f"Error leyendo {f}: {e}")
            
    if not dfs:
        raise ValueError("No se pudieron cargar datos. Verifica la ruta.")
        
    print("Unificando DataFrames...")
    return pd.concat(dfs, ignore_index=True)

def imputar_logica_negocio(df):
    """Aplica las reglas de imputación (Taxi=0 espera, VTC=Probabilidad)."""
    print("Aplicando lógica de imputación...")
    
    # 1. IMPUTACIÓN TAXIS (Espera = 0)
    mask_taxi = df['tipo'] == 'Yellow Taxi'
    n_taxis = df.loc[mask_taxi, 'espera_min'].isna().sum()
    df.loc[mask_taxi, 'espera_min'] = df.loc[mask_taxi, 'espera_min'].fillna(0.0)
    print(f"   -> Taxis: {n_taxis:,} valores de espera rellenados con 0.")

    # 2. IMPUTACIÓN VTC (Pasajeros = Probabilístico)
    # Calculamos la distribución real de los Taxis
    distribucion_real = df.loc[mask_taxi, 'num_pasajeros'].value_counts(normalize=True)
    
    # Localizamos huecos en VTC
    mask_vtc_vacios = (df['tipo'] == 'VTC') & (df['num_pasajeros'].isna())
    cantidad_vtc = mask_vtc_vacios.sum()
    
    if cantidad_vtc > 0:
        # Generamos valores simulados
        valores_simulados = np.random.choice(
            distribucion_real.index.values, 
            size=cantidad_vtc, 
            p=distribucion_real.values
        )
        df.loc[mask_vtc_vacios, 'num_pasajeros'] = valores_simulados
        print(f"   -> VTC: {cantidad_vtc:,} valores de pasajeros simulados estadísticamente.")
    
    return df

def limpieza_final_tipos(df):
    """Asegura que los tipos de datos sean eficientes."""
    print("Optimizando tipos de datos...")
    
    # Rellenar cualquier residuo con valores por defecto seguros
    df['num_pasajeros'] = df['num_pasajeros'].fillna(1).astype('int32')
    df['espera_min'] = df['espera_min'].fillna(0).astype('float32')
    
    # Convertir a categorías para ahorrar memoria
    cols_cat = ['tipo', 'origen_zona', 'destino_zona', 'tipo_vehiculo']
    for col in cols_cat:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    return df

def main():
    # 1. Cargar
    df = cargar_y_unificar(INPUT_PATH)
    
    # 2. Imputar
    df = imputar_logica_negocio(df)
    
    # 3. Limpiar tipos
    df = limpieza_final_tipos(df)
    
    # 4. Guardar
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    ruta_final = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    print(f"Guardando dataset maestro en: {ruta_final}")
    df.to_parquet(ruta_final, index=False)
    print("¡Proceso terminado con éxito!")

if __name__ == "__main__":
    main()