import pandas as pd
import numpy as np
import os
from pathlib import Path
import pyarrow.parquet as pq

# Scikit-Learn Pipeline y Preprocesamiento
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parents[3] 
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'tlc_clean', 'problema5')

TRAIN_FILE = os.path.join(DATA_DIR, 'train.parquet')
VAL_FILE = os.path.join(DATA_DIR, 'val.parquet')

def entrenar_baseline():
    print(" Iniciando entrenamiento del Baseline (Regresión Lineal)...")

    # CARGAR DATOS
    parquet_train = pq.ParquetFile(TRAIN_FILE)
    parquet_val = pq.ParquetFile(VAL_FILE)

    # Solo leemos el primer grupo de filas 
    df_train = parquet_train.read_row_group(0).to_pandas()
    df_val = parquet_val.read_row_group(0).to_pandas()

    # SEPARAR X e y (Variable Objetivo: propina)
    columnas_a_ignorar = ['propina', 'origen_id', 'destino_id']

    X_train = df_train.drop(columns=[col for col in columnas_a_ignorar if col in df_train.columns])
    y_train = df_train['propina']
    
    X_val = df_val.drop(columns=[col for col in columnas_a_ignorar if col in df_val.columns])
    y_val = df_val['propina']

    print(f" Entrenando con {len(X_train):,} registros. Validando con {len(X_val):,}.")


    # DEFINIR COLUMNAS POR TIPO
    columnas_categoricas = [
        'tipo_vehiculo',   # "Yellow Taxi" o "VTC"
        'origen_zona',     # Texto de la zona
        'origen_barrio',   # Texto del barrio (Manhattan, Queens...)
        'destino_zona',    # Texto de la zona
        'destino_barrio',  # Texto del barrio
        'evento_tipo',     # "No hay", "Concierto", etc.
        'franja_horaria'   # "Madrugada", "Noche", etc.
    ]

    # El resto son numéricas (incluyendo las binarias como es_fin_semana, nieve, trafico_denso)
    columnas_numericas = [col for col in X_train.columns if col not in columnas_categoricas]

    # CREAR EL PREPROCESADOR (ColumnTransformer)
    preprocesador = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), columnas_numericas),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), columnas_categoricas)
        ],
        remainder='passthrough'
    )

    # CREAR EL PIPELINE FINAL
    pipeline_lr = Pipeline(steps=[
        ('preprocesamiento', preprocesador),
        ('modelo', LinearRegression())
    ])

    # ENTRENAMIENTO
    print(" Ajustando el Pipeline y entrenando el modelo...")
    pipeline_lr.fit(X_train, y_train)

    # COMPROBAR LA IMPORTANCIA DE LAS VARIABLES
    print(" Analizando la importancia de las variables...")
    modelo_lr = pipeline_lr.named_steps['modelo']
    preprocesador_ajustado = pipeline_lr.named_steps['preprocesamiento']

    nombres_num = columnas_numericas
    nombres_cat = preprocesador_ajustado.named_transformers_['cat'].get_feature_names_out(columnas_categoricas)
    nombres_features = np.concatenate([nombres_num, nombres_cat])
    
    coeficientes = modelo_lr.coef_
    
    df_importancia = pd.DataFrame({
        'Variable': nombres_features,
        'Peso_Coeficiente': coeficientes,
        'Importancia_Absoluta': np.abs(coeficientes)
    })
    
    df_importancia = df_importancia.sort_values(by='Importancia_Absoluta', ascending=False)
    print("\n TOP 10 VARIABLES QUE MÁS AFECTAN A LA PROPINA:")
    print(df_importancia[['Variable', 'Peso_Coeficiente']].head(10).to_string(index=False))

    # PREDICCIÓN SOBRE VALIDACIÓN
    print("\n Realizando predicciones sobre Validación...")
    y_pred = pipeline_lr.predict(X_val)
    
    # CÁLCULO DE MÉTRICAS
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    mean_error = np.mean(y_pred - y_val) 
    epsilon = 1e-8
    mape = np.mean(np.abs((y_val - y_pred) / (y_val + epsilon))) * 100
    
    print("\n" + "="*45)
    print(" REPORTE BASELINE (MODO MEMORIA BAJA)")
    print("="*45)
    print(f"MAE:  ${mae:.4f} (Error absoluto medio)")
    print(f"RMSE: ${rmse:.4f} (Raíz del error cuadrático)")
    print(f"R2:    {r2:.4f} (Varianza explicada)")
    print(f"BIAS: ${mean_error:.4f} ({'Al alza' if mean_error > 0 else 'A la baja'})")
    print(f"MAPE:  {mape:.2f}% (Error porcentual)")
    print("="*45)

if __name__ == "__main__":
    entrenar_baseline()

"""
ANÁLISIS DE RESULTADOS DEL BASELINE (Para la memoria):
El modelo de Regresión Lineal ha reveladoresultados clave sobre el comportamiento 
de los pasajeros en Nueva York. Como era de esperar, el coste del viaje 
(precio_total_est) es el factor principal que impulsa la propina. Sin embargo, 
el modelo ha detectado un fuerte componente geográfico y cultural: los tradicionales 
Yellow Taxis reciben consistentemente más propina que los VTCs, y comenzar un viaje 
en zonas como el Aeropuerto JFK o Jamaica Bay incrementa notablemente la propina, 
mientras que salir de Randalls Island la penaliza.

A nivel predictivo, nuestro modelo base logra un Error Absoluto Medio (MAE) de 1.63$, 
explicando un 30% del comportamiento de las propinas (R2 de 0.3047). Es un punto 
de partida sólido para un modelo lineal simple, pero el RMSE de 2.66$ nos indica 
que el modelo sufre mucho intentando predecir viajes anómalos o propinas extremas.

Desde la perspectiva de ConducIA, la métrica más crítica que hemos 
descubierto es el sesgo (BIAS). Nuestro baseline tiene un error al alza de 0.37$ por viaje. 
Aunque parezca poco, un modelo sistemáticamente "optimista" es peligroso para el negocio, 
ya que generaría falsas expectativas en el conductor, prometiéndole más dinero del 
que realmente va a ganar y dañando la confianza en la aplicación. Además, el desorbitado 
MAPE nos confirma empíricamente la alta presencia de viajes con 0$ de propina.

En conclusión, el baseline demuestra que existen reglas matemáticas claras detrás de 
las propinas, pero la relación no es puramente lineal. El objetivo de nuestro siguiente 
modelo complejo (ej. XGBoost) será reducir el error a menos de 1 dólar, capturar 
ese 70% de varianza oculta y, sobre todo, corregir el sesgo optimista para ofrecer 
al conductor una herramienta honesta y fiable.
"""