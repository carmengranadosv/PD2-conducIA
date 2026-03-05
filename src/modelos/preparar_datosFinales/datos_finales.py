import pandas as pd
import os
import glob
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS Y PARÁMETROS ---
# Ruta base donde están tus archivos parquet originales
BASE_DIR = r'C:\Users\PC\Desktop\3ºGIDIA\PD2\PD2-conducIA\data\raw' 
# Ruta de salida para el CSV final
OUTPUT_DIR = r'C:\Users\PC\Desktop\3ºGIDIA\PD2\PD2-conducIA\data\processed\tlc_clean'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'csv_final.csv')

SAMPLE_RATE = 0.05  # Ajusta este valor según el sampling que necesites (ej. 5%)

def carga_datos_hibrida(anio_actual, anio_anterior):
    lista_muestras = []
    
    # Asegurarnos de que la carpeta de salida existe
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Carpeta creada: {OUTPUT_DIR}")

    for tipo in ['yellow', 'fhvhv']:
        # Definimos los objetivos: meses 1-11 del año actual y mes 12 del anterior
        objetivos = [
            (anio_actual, list(range(1, 12))), 
            (anio_anterior, [12])
        ]
        
        for anio_objetivo, meses_a_buscar in objetivos:
            ruta_busqueda = os.path.join(BASE_DIR, tipo, str(anio_objetivo), "*.parquet")
            archivos_disponibles = sorted(glob.glob(ruta_busqueda))
            
            for archivo in archivos_disponibles:
                # Extraer el mes del nombre del archivo para filtrar
                try:
                    nombre_archivo = Path(archivo).stem
                    # Se espera formato: tipo-año-mes (ej: yellow_tripdata_2023-01)
                    mes_archivo = int(nombre_archivo.split('-')[-1])
                except Exception as e:
                    print(f"Error procesando nombre de archivo {archivo}: {e}")
                    continue

                # Solo procesamos si el mes está en nuestra lista de interés para ese año
                if mes_archivo in meses_a_buscar:
                    df_mes = pd.read_parquet(archivo)
                    
                    # 1. Aplicar Sampling
                    df_sample = df_mes.sample(frac=SAMPLE_RATE, random_state=42)
                    
                    # 2. Etiquetado y limpieza de fecha
                    df_sample['tipo_vehiculo'] = 'Yellow Taxi' if tipo == 'yellow' else 'VTC'
                    df_sample['fecha_inicio'] = pd.to_datetime(df_sample['fecha_inicio'])
                    
                    # 3. Forzamos el mes_num
                    df_sample['mes_num'] = mes_archivo
                    
                    lista_muestras.append(df_sample)
                    print(f"Cargado Mes {mes_archivo} del año {anio_objetivo} para {tipo} con {len(df_sample):,} registros.")

    if not lista_muestras:
        print("No se encontraron datos para procesar.")
        return None

    # Concatenación y ordenación final
    df_final = pd.concat(lista_muestras, ignore_index=True)
    df_final = df_final.sort_values(by=['mes_num', 'tipo_vehiculo']).reset_index(drop=True)
    
    print(f"\nDataset Híbrido Creado: {len(df_final):,} registros totales.")
    
    # --- EXPORTACIÓN A CSV ---
    print(f"Guardando archivo en: {OUTPUT_FILE}...")
    df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print("¡Archivo csv_final.csv creado con éxito!")
    
    return df_final

if __name__ == "__main__":
    # Ejecución del script (ejemplo con 2024 y 2023)
    # Ajusta los años según tus datos disponibles
    df_resultado = carga_datos_hibrida(anio_actual=2024, anio_anterior=2023)