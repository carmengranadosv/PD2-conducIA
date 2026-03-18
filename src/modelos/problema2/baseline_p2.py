import pandas as pd
import os
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

def ejecutar_baseline():
    ruta_p2 = 'data/processed/tlc_clean/problema2'
    ruta_tasas = os.path.join(ruta_p2, 'tasas_base.csv')
    
    # 1. Cargar solo las tasas (es un CSV pequeño, cabe en RAM)
    df_tasas = pd.read_csv(ruta_tasas)
    df_tasas['origen_id'] = df_tasas['origen_id'].astype(float).astype(int).astype(str)

    # 2. Función para procesar trozos (chunks)
    def preparar_chunk(df):
        df['target'] = (df['espera_min'] <= 10).astype(int)
        df['origen_id'] = df['origen_id'].astype(float).astype(int).astype(str)
        df = df.merge(df_tasas, on=['origen_id', 'hora'], how='left')
        df['tasa_historica'] = df['tasa_historica'].fillna(df['tasa_historica'].mean())
        features = ['oferta_inferida', 'temp_c', 'lluvia', 'tasa_historica']
        return df[features].fillna(0), df['target']

    # 3. Leer solo el primer "Row Group" del Parquet para entrenar (Eficiencia máxima)
    print("Leyendo datos de forma eficiente...")
    parquet_train = pq.ParquetFile(os.path.join(ruta_p2, 'train.parquet'))
    parquet_val = pq.ParquetFile(os.path.join(ruta_p2, 'val.parquet'))

    # Solo leemos el primer grupo de filas (suele ser ~100k - 500k filas)
    train_chunk = parquet_train.read_row_group(0).to_pandas()
    val_chunk = parquet_val.read_row_group(0).to_pandas()

    X_train, y_train = preparar_chunk(train_chunk)
    X_val, y_val = preparar_chunk(val_chunk)

    print(f"Entrenando con {len(X_train)} filas...")
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=200)
    model.fit(X_train, y_train)

    # 4. Evaluación
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]

    print("\n" + "="*45)
    print("REPORTE BASELINE (MODO MEMORIA BAJA)")
    print("="*45)
    print(classification_report(y_val, preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_val, probs):.4f}")

if __name__ == "__main__":
    ejecutar_baseline()


"""El modelo de Regresión Logística ha demostrado qeu la combinación 
de la experiencia histórica (tasas) y la oferta en tiempo real (oferta inferida)
permite predecir con éxito la probabilidad de que un conductor encuentre
un cliente en menos de 10 min.
Aunque la mayor parte de los datos son 'casos de éxito' (clase 1), 
el modelo no se ha dejado engañar ya que hemos conseguido un recall de 0.69 para la clase 0.
El sistema es capaz de detectar casi el 70% de las zonas muertas. Lo cual
es vital para el conductor ya que le permite evitar áreas donde perdería tiempo
y gasolina.
Además, con una precisión del 98% para la clase 1, cuando el modelo le dice al conductor que vaya 
a una zona porque va a encontrar viaje, acierta el 98% de las veces. 
Por úlitmo, como el ROC-AUC es casi del 0.7, queda demostrado que existe una correlación real 
entre cuántos taxis llegan a una zona y cuánto tiempo tarda el siguiente conductor en salir de ella."""