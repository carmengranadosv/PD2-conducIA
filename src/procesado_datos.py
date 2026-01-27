import pandas as pd
import os
import glob
import argparse
import sys

# Estandarización de Columnas
COLUMNAS_YELLOW = {
    'tpep_pickup_datetime': 'fecha_inicio',
    'tpep_dropoff_datetime': 'fecha_fin',
    'PULocationID': 'origen_id',
    'DOLocationID': 'destino_id',
    'trip_distance': 'distancia',
    'total_amount': 'precio'
}

COLUMNAS_VTC = {
    'pickup_datetime': 'fecha_inicio',
    'dropoff_datetime': 'fecha_fin',
    'PULocationID': 'origen_id',
    'DOLocationID': 'destino_id',
    'trip_miles': 'distancia',
    'base_passenger_fare': 'precio'
}

def limpiar_y_guardar(ruta_entrada, ruta_salida, tipo):
    """
    Lee un parquet, limpia outliers, estandariza columnas y guarda.
    """
    try:
        print(f" Procesando: {os.path.basename(ruta_entrada)}...")
        df = pd.read_parquet(ruta_entrada)
        
        # 1. Renombrar columnas
        cols_map = COLUMNAS_YELLOW if tipo == 'yellow' else COLUMNAS_VTC
        # Solo seleccionamos las columnas que existen en el archivo
        cols_existentes = [c for c in cols_map.keys() if c in df.columns]
        df = df[cols_existentes].rename(columns=cols_map)
        
        # 2. Calcular Duración en minutos
        df['fecha_inicio'] = pd.to_datetime(df['fecha_inicio'])
        df['fecha_fin'] = pd.to_datetime(df['fecha_fin'])
        df['duracion_min'] = (df['fecha_fin'] - df['fecha_inicio']).dt.total_seconds() / 60
        
        # Eliminar nulos en campos críticos
        df.dropna(subset=['origen_id', 'destino_id', 'precio', 'distancia'], inplace=True)
        
        # Filtros de rango para eliminar errores técnicos y datos irreales
        mask_precio = (df['precio'] > 0) & (df['precio'] < 500)
        mask_distancia = (df['distancia'] > 0) & (df['distancia'] < 1000)
        mask_tiempo = (df['duracion_min'] > 1) & (df['duracion_min'] < 300) # 5 horas max
        
        df = df[mask_precio & mask_distancia & mask_tiempo]
        
        # 3. Optimización de Tipos de Datos
        df['tipo_vehiculo'] = tipo
        # Convertimos a tipos más ligeros (int32 y float32)
        df['origen_id'] = df['origen_id'].astype('int32')
        df['destino_id'] = df['destino_id'].astype('int32')
        df['precio'] = df['precio'].astype('float32')
        df['distancia'] = df['distancia'].astype('float32')
        df['duracion_min'] = df['duracion_min'].astype('float32')

        # 4. Guardar en destino
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        df.to_parquet(ruta_salida, index=False)
        
    except Exception as e:
        print(f" ERROR CRÍTICO en {ruta_entrada}: {e}")

def main():
    # Configuración de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Script de limpieza Taxi/VTC para el equipo.")
    
    parser.add_argument('--year', type=str, required=True, help='Año a procesar (ej: 2024)')
    parser.add_argument('--data_dir', type=str, default='datos', help='Ruta a la carpeta datos')

    args = parser.parse_args()

    # Definir rutas según tu estructura
    # Entrada: datos/2024
    ruta_entrada_anio = os.path.join(args.data_dir, args.year)
    # Salida: datos/datos_clean/2024
    ruta_salida_anio = os.path.join(args.data_dir, 'datos_clean', args.year)

    # Validaciones
    if not os.path.exists(ruta_entrada_anio):
        print(f" Error: No encuentro la carpeta de entrada: {ruta_entrada_anio}")
        print(f" Asegúrate de haber creado la carpeta 'datos/{args.year}' y puesto los parquets dentro.")
        sys.exit(1)

    # Buscar archivos .parquet
    archivos = glob.glob(os.path.join(ruta_entrada_anio, "*.parquet"))
    
    if not archivos:
        print(f" No hay archivos .parquet en {ruta_entrada_anio}")
        sys.exit(1)

    print(f"--- INICIANDO PROCESADO: AÑO {args.year} ---")
    print(f" Leyendo de: {ruta_entrada_anio}")
    print(f" Guardando en: {ruta_salida_anio}")
    print(f" Archivos detectados: {len(archivos)}\n")

    for archivo in archivos:
        filename = os.path.basename(archivo)
        
        # Lógica para detectar tipo por el nombre del archivo
        if "yellow" in filename.lower():
            ruta_salida = os.path.join(ruta_salida_anio, f"clean_{filename}")
            limpiar_y_guardar(archivo, ruta_salida, "yellow")
            
        elif "fhvhv" in filename.lower(): # Uber/Lyft
            ruta_salida = os.path.join(ruta_salida_anio, f"clean_{filename}")
            limpiar_y_guardar(archivo, ruta_salida, "vtc")
            
        else:
            print(f" Archivo ignorado (no es yellow ni fhvhv): {filename}")

    print(f"\n ¡Proceso finalizado exitosamente para {args.year}!")

if __name__ == "__main__":
    main()