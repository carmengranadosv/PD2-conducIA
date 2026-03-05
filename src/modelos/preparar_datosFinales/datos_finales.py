import pandas as pd
import os
import glob
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
# Como el script está en src/modelos/preparar_datosFinales/, subimos niveles para llegar a la raíz
BASE_DIR = r'C:\Users\PC\Desktop\3ºGIDIA\PD2\PD2-conducIA'
# Ruta donde están tus parquets limpios (según me has indicado)
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'tlc_clean')
# Ruta de salida (la misma donde están los datos)
OUTPUT_FILE = os.path.join(INPUT_DIR, 'csv_final.csv')

SAMPLE_RATE = 0.05  # Ajusta según necesites (5% para que no pese demasiado)

def carga_datos_hibrida(anio_actual, anio_anterior):
    lista_muestras = []
    
    # Verificamos que la carpeta de entrada existe
    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: No se encuentra la carpeta de datos en: {INPUT_DIR}")
        return None

    for tipo in ['yellow', 'fhvhv']:
        # Definimos los objetivos: meses 1-11 del año actual y mes 12 del anterior
        objetivos = [
            (anio_actual, list(range(1, 12))), 
            (anio_anterior, [12])
        ]
        
        for anio_objetivo, meses_a_buscar in objetivos:
            # Buscamos los archivos .parquet dentro de las subcarpetas de año
            ruta_busqueda = os.path.join(INPUT_DIR, tipo, str(anio_objetivo), "*.parquet")
            archivos_disponibles = sorted(glob.glob(ruta_busqueda))
            
            if not archivos_disponibles:
                print(f"Aviso: No hay archivos en {ruta_busqueda}")
                continue

            for archivo in archivos_disponibles:
                try:
                    nombre_archivo = Path(archivo).stem
                    # Extraemos el mes (se espera que el nombre termine en -MM)
                    mes_archivo = int(nombre_archivo.split('-')[-1])
                except:
                    continue

                if mes_archivo in meses_a_buscar:
                    df_mes = pd.read_parquet(archivo)
                    
                    # 1. Aplicar Sampling para manejo de memoria
                    df_sample = df_mes.sample(frac=SAMPLE_RATE, random_state=42)
                    
                    # 2. Etiquetado y limpieza
                    df_sample['tipo_vehiculo'] = 'Yellow Taxi' if tipo == 'yellow' else 'VTC'
                    df_sample['fecha_inicio'] = pd.to_datetime(df_sample['fecha_inicio'])
                    df_sample['mes_num'] = mes_archivo
                    
                    lista_muestras.append(df_sample)
                    print(f"Cargado: {tipo} | Año {anio_objetivo} | Mes {mes_archivo} ({len(df_sample):,} filas)")

    if not lista_muestras:
        print("No se ha podido cargar ningún dato. Revisa las rutas.")
        return None

    # Concatenamos y ordenamos por mes
    df_final = pd.concat(lista_muestras, ignore_index=True)
    df_final = df_final.sort_values(by=['mes_num', 'tipo_vehiculo']).reset_index(drop=True)
    
    print(f"\nProceso terminado. Registros totales: {len(df_final):,}")
    
    # Exportación final
    print(f"Guardando en: {OUTPUT_FILE}...")
    df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print("¡Hecho!")
    
    return df_final

if __name__ == "__main__":
    # Ejecutamos el proceso para consolidar el dataset híbrido
    carga_datos_hibrida(anio_actual=2025, anio_anterior=2024)