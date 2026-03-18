"""Generamos las tasas bases, que son un porcentaje de éxito. 
Le sirven al conductor para no perder el tiempo y saber
dónde es más probable que salte el próximo viaje. """

import pandas as pd
import os

def generar_tasas():
    # Nueva ruta unificada
    ruta_p2 = 'data/processed/tlc_clean/problema2'
    path_train = os.path.join(ruta_p2, 'train.parquet')
    
    if not os.path.exists(path_train):
        print(f"❌ Error: No se encuentra {path_train}. ¿Ejecutaste division_datos.py?")
        return

    print("Calculando tasas históricas...")
    df = pd.read_parquet(path_train, columns=['origen_id', 'hora', 'espera_min'])
    
    # Éxito si espera <= 10 min
    df['exito'] = (df['espera_min'] <= 10).astype(int)
    
    # Agrupamos por zona y hora
    tasa_id_hora = df.groupby(['origen_id', 'hora'])['exito'].mean().reset_index(name='tasa_historica')
    
    # Guardamos en la misma carpeta del Problema 2
    output_csv = os.path.join(ruta_p2, 'tasas_base.csv')
    tasa_id_hora.to_csv(output_csv, index=False)
    print(f"✅ Tasas guardadas en: {output_csv}")

if __name__ == "__main__":
    generar_tasas()