import os
from pathlib import Path
from src.modelos.preparar_datosFinales.datos_finales import carga_datos_hibrida
from src.modelos.preparar_datosFinales.limpieza_nulos import limpiar_y_enriquecer_extremo_ram

# --- RUTAS ---
BASE_DIR = Path(__file__).resolve().parents[3] 
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'tlc_clean')

# Definimos los archivos
ARCHIVO_INTERMEDIO = os.path.join(INPUT_DIR, 'dataset_intermedio.parquet')
ARCHIVO_FINAL = os.path.join(INPUT_DIR, 'datos_final.parquet')

if __name__ == "__main__":
    print("Iniciando Pipeline de Datos ConducIA...")
    
    # 1. Si el dataset final existe hacemos skip de todo
    if os.path.exists(ARCHIVO_FINAL):
        print(f" ¡El archivo final ya existe en {ARCHIVO_FINAL}!")
        print(" SKIP de ambas fases. Todo está listo.")
    
    else:
        # Si el archivo unificado existe hacemos SKIP de esa fase
        if os.path.exists(ARCHIVO_INTERMEDIO):
            print(" El archivo intermedio ya existe. SKIP Fase de Carga.")
        else:
            print(" Carga y unión de datos...")
            carga_datos_hibrida(INPUT_DIR, ARCHIVO_INTERMEDIO, anio_actual=2025, anio_anterior=2024)
        
        # 3. Ejecutamos la limpieza 
        print(" Limpieza y creación de variables...")
        limpiar_y_enriquecer_extremo_ram(ARCHIVO_INTERMEDIO, ARCHIVO_FINAL)
        
        #if os.path.exists(ARCHIVO_INTERMEDIO):
            #os.remove(ARCHIVO_INTERMEDIO)
            #print(" Archivo intermedio eliminado para liberar espacio.")
            
        print(" Pipeline finalizado. Listo para entrenar modelos.")