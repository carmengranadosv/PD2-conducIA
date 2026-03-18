import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from pathlib import Path

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
    df_train = pd.read_parquet(TRAIN_FILE, engine='pyarrow')
    df_val = pd.read_parquet(VAL_FILE, engine='pyarrow')

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

    # CONFIGURAR MLFLOW Y ENTRENAR
    # Le damos un nombre al experimento para tenerlo organizado en la interfaz web
    mlflow.set_experiment("ConducIA_Problema5_Propinas")
    
    with mlflow.start_run(run_name="Baseline_Regresion_Lineal"):
        print(" Ajustando el Pipeline y entrenando el modelo...")
        
        # ENTRENAMIENTO
        pipeline_lr.fit(X_train, y_train)

        # COMPROBAR LA IMPORTANCIA DE LAS VARIABLES
        print(" Analizando la importancia de las variables...")

        # Sacar el modelo matemático y el preprocesador de dentro del Pipeline
        modelo_lr = pipeline_lr.named_steps['modelo']
        preprocesador = pipeline_lr.named_steps['preprocesamiento']

        # Recuperar los nombres de las columnas en el orden exacto
        nombres_num = columnas_numericas
        # Luego las categóricas transformadas (ej. 'origen_barrio_Manhattan')
        nombres_cat = preprocesador.named_transformers_['cat'].get_feature_names_out(columnas_categoricas)
        # Juntamos todos los nombres
        nombres_features = np.concatenate([nombres_num, nombres_cat])
        
        # Emparejamos los nombres con los pesos (coeficientes) del modelo
        coeficientes = modelo_lr.coef_
        
        df_importancia = pd.DataFrame({
            'Variable': nombres_features,
            'Peso_Coeficiente': coeficientes,
            'Importancia_Absoluta': np.abs(coeficientes) # Para ver el impacto total, sea + o -
        })
        
        # Ordenar y mostrar el Top 10
        df_importancia = df_importancia.sort_values(by='Importancia_Absoluta', ascending=False)
        print("\n TOP 10 VARIABLES QUE MÁS AFECTAN A LA PROPINA:")
        print(df_importancia[['Variable', 'Peso_Coeficiente']].head(10).to_string(index=False))
        
        # Guardar este CSV en MLflow para tenerlo de prueba
        df_importancia.to_csv("importancia_variables_baseline.csv", index=False)
        mlflow.log_artifact("importancia_variables_baseline.csv")
        
        # PREDICCIÓN SOBRE VALIDACIÓN
        print(" Realizando predicciones sobre Validación...")
        y_pred = pipeline_lr.predict(X_val)
        
        # CÁLCULO DE MÉTRICAS
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        
        # Métricas de Sesgo (Bias)
        mean_error = np.mean(y_pred - y_val) 
        # MAPE: Añadimos epsilon para evitar división por 0 si la propina real fue 0
        epsilon = 1e-8
        mape = np.mean(np.abs((y_val - y_pred) / (y_val + epsilon))) * 100
        
        print("\n" + "="*30)
        print(" RESULTADOS DEL BASELINE")
        print("="*30)
        print(f"MAE:  ${mae:.4f}")
        print(f"RMSE: ${rmse:.4f}")
        print(f"R2:    {r2:.4f}")
        print(f"BIAS (Mean Error): ${mean_error:.4f} ({'Al alza' if mean_error > 0 else 'A la baja'})")
        print("="*30)
        
        # REGISTRAR EN MLFLOW
        # Guardamos hiperparámetros básicos
        mlflow.log_param("modelo", "Regresion Lineal Multiple")
        mlflow.log_param("features_usadas", len(X_train.columns))
        
        # Guardamos las métricas
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mean_error_bias", mean_error)
        mlflow.log_metric("mape", mape)
        
        # Guardamos el pipeline completo
        mlflow.sklearn.log_model(pipeline_lr, "pipeline_baseline")
        
        print(" Experimento guardado en MLflow correctamente.")

if __name__ == "__main__":
    entrenar_baseline()

