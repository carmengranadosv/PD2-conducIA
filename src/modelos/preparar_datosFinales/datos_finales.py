import pandas as pd
import os
import glob
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'tlc_clean')
# Cambiamos la extensión a .parquet
OUTPUT_FILE = os.path.join(INPUT_DIR, 'dataset_final.parquet')

# Muestra del 60% de los datos para cada mes (1=100%, 0.6=60%, etc.)
SAMPLE_RATE = 0.4 

def carga_datos_hibrida(anio_actual, anio_anterior):
    lista_muestras = []
    
    for tipo in ['yellow', 'fhvhv']:
        # Definimos los objetivos: meses 1-11 del año actual y mes 12 del anterior
        objetivos = [
            (anio_actual, list(range(1, 12))), 
            (anio_anterior, [12])
        ]
        
        for anio_objetivo, meses_a_buscar in objetivos:
            ruta_busqueda = os.path.join(INPUT_DIR, tipo, str(anio_objetivo), "*.parquet")
            archivos_disponibles = sorted(glob.glob(ruta_busqueda))
            
            for archivo in archivos_disponibles:
                # Extraer el mes del nombre del archivo para filtrar
                try:
                    nombre_archivo = Path(archivo).stem
                    mes_archivo = int(nombre_archivo.split('-')[-1])
                except:
                    continue

                # Solo procesamos si el mes está en nuestra lista de interés para ese año
                if mes_archivo in meses_a_buscar:
                    df_mes = pd.read_parquet(archivo)
                    
                    # 1. Aplicar Sampling
                    df_sample = df_mes.sample(frac=SAMPLE_RATE, random_state=42)
                    
                    # 2. Etiquetado y limpieza de fecha
                    df_sample['tipo_vehiculo'] = 'Yellow Taxi' if tipo == 'yellow' else 'VTC'
                    df_sample['fecha_inicio'] = pd.to_datetime(df_sample['fecha_inicio'])
                    
                    # 3. Forzamos el mes_num para que diciembre (del año pasado) 
                    # aparezca correctamente en las gráficas de este año
                    df_sample['mes_num'] = mes_archivo
                    
                    lista_muestras.append(df_sample)
                    print(f"Cargado Mes {mes_archivo} del año {anio_objetivo} para {tipo} con {len(df_sample):,} registros.")

    # Concatenación y ordenación final
    df_final = pd.concat(lista_muestras, ignore_index=True)

    # --- NUEVO: ORDENACIÓN CRONOLÓGICA ---
    print("\nOrdenando el dataset cronológicamente...")
    # Ordenamos por la fecha completa (Año-Mes-Día Hora) para que sea exacto
    df_final = df_final.sort_values(by=['fecha_inicio']).reset_index(drop=True)
    
    print(f"\nDataset Híbrido Creado: {len(df_final):,} registros totales.")

    # --- EXPORTACIÓN A PARQUET ---
    print(f"Guardando en: {OUTPUT_FILE}...")
    # Usamos motor 'pyarrow' o 'fastparquet' (por defecto suele ser snappy)
    df_final.to_parquet(OUTPUT_FILE, index=False, compression='snappy')
    print("¡Hecho!")
    
    return df_final

if __name__ == "__main__":
    # Ejecutamos con tus parámetros
    df_resultado = carga_datos_hibrida(anio_actual=2025, anio_anterior=2024)