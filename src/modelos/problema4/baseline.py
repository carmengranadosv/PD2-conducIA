"""
DESCRIPCIÓN:
    Este script implementa el modelo BASELINE para el Problema 4 (Eficiencia y Clima).
    Utiliza una Regresión Ridge (Lineal Regularizada) para predecir 'velocidad_mph'.

OBJETIVO:
    Establecer el nivel mínimo de precisión que los modelos complejos 
    (ST-GCN y Red Neuronal) deben superar.

FLUJO DE TRABAJO:
    1. Importa funciones de 'preprocesamiento_base.py'.
    2. Aplica One-Hot Encoding a variables categóricas (zonas, franjas, vehículos).
    3. Entrena el modelo Ridge con el set de Entrenamiento (Dic-Sep).
    4. Evalúa el rendimiento con el set de Test (Oct-Nov) usando MAE, RMSE y R2.
    5. Muestra la importancia de las variables (qué clima afecta más al tráfico).

"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Importamos nuestro preprocesamiento centralizado
# Nota: Asegúrate de que el archivo preprocesamiento_base.py esté en la misma carpeta
from preprocesamiento_base import flujo_preprocesamiento_base, realizar_split_temporal

def preparar_features_baseline(df):
    """
    Transforma variables categóricas en columnas numéricas (Encoding).
    """
    print("Transformando variables categóricas para la Regresión...")
    
    # Seleccionamos las variables para el modelo
    # Nota: No incluimos fecha_inicio ni destino_id para simplificar el baseline
    cols_categoricas = ['tipo_vehiculo', 'franja_horaria']
    cols_numericas = [
        'temp_c', 'precipitation', 'viento_kmh', 'lluvia', 'nieve', 
        'hay_lluvia', 'hay_nieve', 'es_festivo', 'num_eventos', 
        'dia_semana', 'es_fin_semana', 'hora_sen', 'hora_cos'
    ]
    
    # Creamos variables Dummy (One-Hot Encoding)
    df_encoded = pd.get_dummies(df[cols_categoricas + cols_numericas], 
                                 columns=cols_categoricas, 
                                 drop_first=True)
    
    return df_encoded, df['velocidad_mph']

def ejecutar_baseline():
    # 1. Obtener datos del preprocesamiento base
    df_base = flujo_preprocesamiento_base()
    train_raw, test_raw = realizar_split_temporal(df_base)
    
    # 2. Preparar X e y para Train y Test
    X_train, y_train = preparar_features_baseline(train_raw)
    X_test, y_test = preparar_features_baseline(test_raw)
    
    # Alineamos columnas por si falta alguna categoría en test
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    # 3. Entrenar Modelo Ridge
    print("\nEntrenando Regresión Ridge (Baseline)...")
    modelo = Ridge(alpha=1.0) # alpha es la fuerza de regularización
    modelo.fit(X_train, y_train)
    
    # 4. Predicción y Evaluación
    y_pred = modelo.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n==========================================")
    print("RESULTADOS DEL BASELINE (Test: Oct-Nov)")
    print("==========================================")
    print(f"MAE (Error medio): {mae:.2f} mph")
    print(f"RMSE: {rmse:.2f} mph")
    print(f"R2 Score: {r2:.4f}")
    print("==========================================\n")
    
    # 5. Importancia de las variables (Coeficientes)
    importancia = pd.Series(modelo.coef_, index=X_train.columns).sort_values()
    print("Top 5 factores que más REDUCEN la velocidad:")
    print(importancia.head(5))
    
    return modelo

if __name__ == "__main__":
    modelo_final = ejecutar_baseline()