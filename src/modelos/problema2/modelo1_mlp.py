import pandas as pd
import pyarrow.parquet as pq
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# Cargamos los datos (Cargamos TODO lo numérico para que el algoritmo elija)
ruta = 'data/processed/tlc_clean/problema2/train.parquet'
columnas_candidatas = [
    'oferta_inferida', 'temp_c', 'lluvia', 'precipitation', 'viento_kmh', 
    'nieve', 'es_festivo', 'num_eventos', 'hora_sen', 'hora_cos', 
    'dia_semana', 'es_fin_semana', 'origen_id', 'espera_min'
]

df = pq.read_table(ruta, columns=columnas_candidatas).to_pandas().head(100000)

# Creamos el objetivo (Target)
df['target'] = (df['espera_min'] <= 10).astype(int)
df['demanda_score'] = 0.5

# Definimos X inicial con todas las posibles
X_inicial = df.drop(['espera_min', 'target'], axis=1).fillna(0)
y = df['target']

# --- FUNDAMENTO MATEMÁTICO: SELECCIÓN AUTOMÁTICA ---
print("Ejecutando selección automática de variables significativas...")
# Usamos un Random Forest como 'juez' para medir la importancia real
juez = RandomForestClassifier(n_estimators=50, random_state=42)
selector = SelectFromModel(juez, threshold="mean") # Solo pasan las que aportan más que la media
selector.fit(X_inicial, y)

# Transformamos X para quedarnos solo con las elegidas por el algoritmo
X_significativas = selector.transform(X_inicial)
columnas_elegidas = X_inicial.columns[selector.get_support()]

print(f"El algoritmo ha decidido que las variables más significativas son: {list(columnas_elegidas)}")

# Escalado de las variables elegidas
scaler = StandardScaler()
X_escalado = scaler.fit_transform(X_significativas)

# Configuramos el cerebro (MLP)
print("Entrenando la Red Neuronal con la selección automática...")
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32), 
    activation='relu', 
    solver='adam', 
    max_iter=20, 
    random_state=42,
    verbose=True 
)

mlp.fit(X_escalado, y)

# Evaluamos el modelo
probabilidades = mlp.predict_proba(X_escalado)[:, 1]
score = roc_auc_score(y, probabilidades)

print("\n" + "="*30)
print(f"RESULTADO MODELO 1 (MLP)")
print(f"ROC-AUC FINAL: {score:.4f}")
print(f"Variables finales: {list(columnas_elegidas)}")
print("="*30)