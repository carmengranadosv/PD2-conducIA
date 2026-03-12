"""
BASELINE PARA PROBLEMA 2: Clasificación de éxito para el conductor.

OBJETIVO:
Este script establece una línea base (Baseline) utilizando Regresión Logística
para predecir la probabilidad de éxito de un conductor.

¿QUÉ HACEMOS?
1. DEFINICIÓN DE ÉXITO: Consideramos un "éxito" (1) si el tiempo de espera 
   del conductor para encontrar un viaje es <= 10 minutos.
2. MODELADO: Usamos variables de entorno (oferta inferida, clima, tiempo) 
   para entrenar un clasificador lineal.
3. EVALUACIÓN: Medimos Accuracy y F1-Score para saber si nuestra 
   lógica de 'oferta_inferida' realmente predice la rentabilidad.
"""

import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def preparar_y_entrenar():
    ruta_base = 'data/processed/tlc_clean/problema2'
    
    # 1. Cargar datasets (usamos solo las columnas necesarias para ahorrar RAM)
    cols_a_usar = ['espera_min', 'oferta_inferida', 'temp_c', 'hora_sen', 'hora_cos', 'es_fin_semana']
    
    print("Cargando datos...")
    train_df = pd.read_parquet(os.path.join(ruta_base, 'train.parquet'), columns=cols_a_usar)
    val_df = pd.read_parquet(os.path.join(ruta_base, 'val.parquet'), columns=cols_a_usar)
    
    # 2. Ingeniería de variables: El "Target"
    # Éxito (1) = Espera <= 10 min | Fallo (0) = Espera > 10 min
    train_df['exito'] = (train_df['espera_min'] <= 10).astype(int)
    val_df['exito'] = (val_df['espera_min'] <= 10).astype(int)
    
    # 3. Separar X (features) e y (target)
    features = ['oferta_inferida', 'temp_c', 'hora_sen', 'hora_cos', 'es_fin_semana']
    X_train, y_train = train_df[features].fillna(0), train_df['exito']
    X_val, y_val = val_df[features].fillna(0), val_df['exito']
    
    # 4. Entrenar el Baseline
    print(f"Entrenando con {len(X_train)} registros...")
    modelo = LogisticRegression(solver='liblinear')
    modelo.fit(X_train, y_train)
    
    # 5. Evaluar
    predicciones = modelo.predict(X_val)
    
    print("\n--- RESULTADOS DEL MODELO BASELINE ---")
    print(f"Accuracy: {accuracy_score(y_val, predicciones):.4f}")
    print("\nReporte detallado (Precision/Recall/F1-Score):")
    print(classification_report(y_val, predicciones))
    
    # Explicación de la importancia de variables (Coeficientes)
    importancia = pd.DataFrame({'Variable': features, 'Importancia': modelo.coef_[0]})
    print("\n--- PESO DE CADA VARIABLE (Logit Coefs) ---")
    print(importancia.sort_values(by='Importancia', ascending=False))

if __name__ == "__main__":
    preparar_y_entrenar()