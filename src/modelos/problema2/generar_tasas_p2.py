import pandas as pd
import os

def generar_tasas():
    # Nueva ruta unificada
    ruta_p2 = 'data/processed/tlc_clean/problema2'
    path_train = os.path.join(ruta_p2, 'train.parquet')
    
    if not os.path.exists(path_train):
        print(f"❌ Error: No se encuentra {path_train}. ¿Ejecutaste division_datos.py?")
        return

    print("Calculando tasas y demanda histórica...")
    df = pd.read_parquet(path_train, columns=['origen_id', 'hora', 'espera_min'])
    
    # Éxito si espera <= 10 min
    df['exito'] = (df['espera_min'] <= 10).astype(int)
    
    # Agrupamos por zona y hora sacando media (tasa) y conteo (demanda)
    agregado = df.groupby(['origen_id', 'hora']).agg(
        tasa_historica=('exito', 'mean'),
        demanda_real=('exito', 'size')
    ).reset_index()
    
    # Normalizamos la demanda para que sea un score entre 0 y 1
    agregado['demanda_score'] = agregado['demanda_real'] / agregado['demanda_real'].max()
    
    # Guardamos en la misma carpeta del Problema 2
    output_csv = os.path.join(ruta_p2, 'tasas_base.csv')
    agregado.to_csv(output_csv, index=False)
    print(f"✅ Tasas y demanda guardadas en: {output_csv}")

if __name__ == "__main__":
    generar_tasas()