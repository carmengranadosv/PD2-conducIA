"""empaquetador final: junta todos los archivos limpios 
y enriquecidos en uno solo para que la IA pueda leerlo"""
import pandas as pd
import numpy as np
import glob
import os

# Configuración de rutas 
INPUT_PATH = "data/processed/tlc_clean/*/*/*.parquet"
OUTPUT_DIR = "data/model_ready"
OUTPUT_FILE = "dataset.parquet"

def cargar_y_unificar(ruta_patron):
    """Carga todos los parquets y les asigna su etiqueta de tipo."""
    archivos = glob.glob(ruta_patron)
    print(f" Detectados {len(archivos)} archivos. Iniciando carga...")
    
    dfs = []
    for f in archivos:
        try:
            df_temp = pd.read_parquet(f)
            
            # Etiquetado automático
            if 'yellow' in f.lower():
                df_temp['tipo'] = 'Yellow Taxi'
            elif 'fhvhv' in f.lower():
                df_temp['tipo'] = 'VTC'
            
            print(f"  ✅ {os.path.basename(f)}: {len(df_temp):,} registros")
            dfs.append(df_temp)
            
        except Exception as e:
            print(f"   Error leyendo {f}: {e}")
            
    if not dfs:
        raise ValueError("No se pudieron cargar datos. Verifica la ruta.")
        
    print("\n Unificando DataFrames...")
    return pd.concat(dfs, ignore_index=True)

def imputar_logica_negocio(df):
    """Aplica las reglas de imputación (Taxi=0 espera, VTC=Probabilidad, Propinas=Mediana)."""
    print("\n Aplicando lógica de imputación...")
    
    # 1. IMPUTACIÓN TAXIS (Espera = 0)
    if 'espera_min' in df.columns:
        mask_taxi = df['tipo'] == 'Yellow Taxi'
        n_taxis = df.loc[mask_taxi, 'espera_min'].isna().sum()
        df.loc[mask_taxi, 'espera_min'] = df.loc[mask_taxi, 'espera_min'].fillna(0.0)
        print(f"   ⏱  Taxis: {n_taxis:,} valores de espera rellenados con 0")
    
    # 2. IMPUTACIÓN VTC (Pasajeros = Probabilístico)
    if 'num_pasajeros' in df.columns:
        mask_taxi = df['tipo'] == 'Yellow Taxi'
        if mask_taxi.sum() > 0 and not df.loc[mask_taxi, 'num_pasajeros'].isna().all():
            distribucion_real = df.loc[mask_taxi, 'num_pasajeros'].value_counts(normalize=True)
            mask_vtc_vacios = (df['tipo'] == 'VTC') & (df['num_pasajeros'].isna())
            cantidad_vtc = mask_vtc_vacios.sum()
            
            if cantidad_vtc > 0:
                valores_simulados = np.random.choice(
                    distribucion_real.index.values, 
                    size=cantidad_vtc, 
                    p=distribucion_real.values
                )
                df.loc[mask_vtc_vacios, 'num_pasajeros'] = valores_simulados
                print(f"    VTC: {cantidad_vtc:,} valores de pasajeros simulados")

    # 3. NUEVA IMPUTACIÓN DE PROPINAS 
    if 'propina' in df.columns:
        # Calculamos la mediana usando solo los valores que NO son nulos (propinas de tarjeta)
        mediana_propina = df['propina'].median()
        n_nulos_propina = df['propina'].isna().sum()
        
        # Rellenamos los NaN (efectivo/desconocidos) con esa mediana
        df['propina'] = df['propina'].fillna(mediana_propina)
        print(f" Propinas: {n_nulos_propina:,} valores nulos imputados con la mediana (${mediana_propina:.2f})")
    
    return df

def limpieza_final_tipos(df):
    """Asegura que los tipos de datos sean eficientes."""
    print("\n  Optimizando tipos de datos...")
    
    # Columnas numéricas con defaults seguros
    if 'num_pasajeros' in df.columns:
        df['num_pasajeros'] = df['num_pasajeros'].fillna(1).astype('int32')
    
    if 'espera_min' in df.columns:
        df['espera_min'] = df['espera_min'].fillna(0).astype('float32')
    
    # NUEVAS COLUMNAS: Festivos y Eventos (rellenar con 0 si faltan)
    if 'es_festivo' in df.columns:
        df['es_festivo'] = df['es_festivo'].fillna(0).astype('int8')
    
    if 'hay_evento' in df.columns:
        df['hay_evento'] = df['hay_evento'].fillna(0).astype('int8')
    
    # Clima (rellenar NaNs con valores neutros)
    if 'lluvia' in df.columns:
        df['lluvia'] = df['lluvia'].fillna(0).astype('int8')
    
    if 'nieve' in df.columns:
        df['nieve'] = df['nieve'].fillna(0).astype('int8')
    
    if 'temp_c' in df.columns:
        df['temp_c'] = df['temp_c'].fillna(df['temp_c'].median()).astype('float32')
    
    if 'viento_kmh' in df.columns:
        df['viento_kmh'] = df['viento_kmh'].fillna(df['viento_kmh'].median()).astype('float32')
    
    # Convertir a categorías para ahorrar memoria
    cols_cat = [
        'tipo', 
        'origen_zona', 
        'destino_zona', 
        'tipo_vehiculo',
        'origen_barrio',
        'destino_barrio'
    ]
    
    for col in cols_cat:
        if col in df.columns:
            df[col] = df[col].astype('category')
            print(f"    {col}: {df[col].nunique()} categorías únicas")
    
    return df

def estadisticas_dataset(df):
    """Muestra estadísticas del dataset final."""
    print("\n" + "="*60)
    print(" ESTADÍSTICAS DEL DATASET FINAL")
    print("="*60)
    
    print(f"\n Dimensiones:")
    print(f"   Filas: {len(df):,}")
    print(f"   Columnas: {len(df.columns)}")
    
    print(f"\n Distribución por tipo:")
    print(df['tipo'].value_counts().to_string().replace('\n', '\n   '))
    
    # Rango temporal
    if 'fecha_inicio' in df.columns:
        print(f"\n Período temporal:")
        print(f"   Desde: {df['fecha_inicio'].min()}")
        print(f"   Hasta: {df['fecha_inicio'].max()}")
    
    # Festivos y eventos
    if 'es_festivo' in df.columns:
        n_festivos = (df['es_festivo'] == 1).sum()
        print(f"\n Viajes en festivos: {n_festivos:,} ({n_festivos/len(df)*100:.2f}%)")
    
    if 'hay_evento' in df.columns:
        n_eventos = (df['hay_evento'] == 1).sum()
        print(f" Viajes con eventos: {n_eventos:,} ({n_eventos/len(df)*100:.2f}%)")
    
    # Clima
    if 'lluvia' in df.columns:
        n_lluvia = (df['lluvia'] == 1).sum()
        print(f"  Viajes con lluvia: {n_lluvia:,} ({n_lluvia/len(df)*100:.2f}%)")
    
    if 'nieve' in df.columns:
        n_nieve = (df['nieve'] == 1).sum()
        print(f"  Viajes con nieve: {n_nieve:,} ({n_nieve/len(df)*100:.2f}%)")
    
    # Memoria
    memoria_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"\n Uso de memoria: {memoria_mb:.1f} MB")
    
    print("\n Columnas disponibles:")
    for i, col in enumerate(df.columns, 1):
        tipo = str(df[col].dtype)
        print(f"   {i:2d}. {col:25s} ({tipo})")

def main():
    print("="*60)
    print(" PREPARACIÓN DATASET MAESTRO")
    print("="*60)
    
    # 1. Cargar
    df = cargar_y_unificar(INPUT_PATH)
    
    # 2. Imputar
    df = imputar_logica_negocio(df)
    
    # 3. Limpiar tipos
    df = limpieza_final_tipos(df)
    
    # 4. Estadísticas
    estadisticas_dataset(df)
    
    if 'tipo' in df.columns:
        print("\n  Eliminando columna auxiliar 'tipo' del dataset final")
        df = df.drop(columns=['tipo'])

    # 5. Guardar
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    ruta_final = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    print(f"\n Guardando dataset maestro en: {ruta_final}")
    df.to_parquet(ruta_final, index=False)
    
    # Verificar archivo guardado
    size_mb = os.path.getsize(ruta_final) / 1024**2
    print(f"    Archivo guardado: {size_mb:.1f} MB")
    
    print("\n" + "="*60)
    print(" PROCESO COMPLETADO CON ÉXITO")
    print("="*60)

if __name__ == "__main__":
    main()