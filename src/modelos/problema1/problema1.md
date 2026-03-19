# Problema 1: Predicción de Demanda por Zona y Hora

##  Descripción

Este módulo implementa un sistema completo de predicción de demanda de viajes en taxi por zona geográfica y franja horaria en la ciudad de Nueva York.

**Objetivo:** Predecir cuántos viajes habrá en cada zona en la próxima hora, permitiendo a los conductores posicionarse estratégicamente para maximizar ganancias.

---

##  Estructura del Proyecto
```
src/modelos/problema1/
├── samplear_datos.py      # 1. Muestreo del dataset completo
├── division_datos.py       # 2. División temporal (train/val/test)
├── agregacion.py           # 3. Agregación y feature engineering
├── baseline.py             # 4. Modelos baseline (Naive, Media hist, RF)
└── modelo1_lstm.py         # 5. Modelo LSTM (deep learning)
```

---

##  Pipeline de Ejecución

### **Orden de ejecución (OBLIGATORIO):**
```bash
# Desde la raíz del proyecto: ~/PD2/PD2-conducIA/

# 1. Samplear datos (10% del dataset original)
uv run python src/modelos/problema1/samplear_datos.py

# 2. Dividir temporalmente (70% train, 15% val, 15% test)
uv run python src/modelos/problema1/division_datos.py

# 3. Agregar por zona×hora y crear features
uv run python src/modelos/problema1/agregacion.py

# 4. Entrenar modelos baseline
uv run python src/modelos/problema1/baseline.py

# 5. Entrenar modelo LSTM
uv run python src/modelos/problema1/modelo1_lstm.py
```

**⚠️ IMPORTANTE:** Ejecutar en orden. Cada script depende de la salida del anterior.

---

## 📂 Datos de Entrada/Salida

### **Entrada inicial:**
```
data/processed/tlc_clean/datos_final.parquet
└── 80M filas (~2.6 GB)
```
---

## 📄 Descripción Detallada de Cada Script

---

### **1. samplear_datos.py**

**Función:** Reduce el dataset original de 80M filas a 10% para trabajar con RAM limitada.

**Método:**
- Lee metadata del Parquet (sin cargar datos)
- Selecciona 7 de 77 row groups distribuidos uniformemente
- Garantiza cobertura temporal completa del dataset

**Entrada:**
```
data/processed/tlc_clean/datos_final.parquet (80M filas, 2.6 GB)
```

**Salida:**
```
data/processed/tlc_clean/datos_sample_10pct.parquet (7.3M filas, 242 MB)
```

**Configuración:**
```python
SAMPLE_FRACTION = 0.10  # 10% del dataset
```

**Ejecución:**
```bash
uv run python src/modelos/problema1/samplear_datos.py
```

**Notas:**
- Ajustar `SAMPLE_FRACTION` si tienes más/menos RAM
- Usa sampling **distribuido** (no aleatorio) para mantener cobertura temporal

---

### **2. division_datos.py**

**Función:** Divide el dataset en train/val/test por **fechas** (validación temporal).

**Método:**
- Lee rango temporal completo del dataset
- Calcula cortes: 70% train, 15% val, 15% test
- Verifica que NO haya solapamiento temporal

**Entrada:**
```
data/processed/tlc_clean/datos_sample_10pct.parquet (7.3M filas)
```

**Salida:**
```
data/processed/tlc_clean/problema1/raw/
├── train.parquet (5.2M filas: 2024-12-01 → 2025-06-28)
├── val.parquet   (1.0M filas: 2025-08-19 → 2025-08-24)
├── test.parquet  (1.0M filas: 2025-10-12 → 2025-10-17)
└── metadata.json (fechas corte, porcentajes, stats)
```

**Splits:**
| Split | % | Filas | Período |
|-------|---|-------|---------|
| Train | 70% | 5.2M | Dic 2024 - Jun 2025 |
| Val | 15% | 1.0M | Ago 2025 |
| Test | 15% | 1.0M | Oct 2025 |

**Ejecución:**
```bash
uv run python src/modelos/problema1/division_datos.py
```

**Notas:**
- **NO** usa split aleatorio (invalidaría series temporales)
- Garantiza train < val < test temporalmente

---

### **3. agregacion.py**

**Función:** Convierte viajes individuales en series temporales agregadas por zona×hora y crea features para predicción.

**Método:**

1. **Agregación:**
   - Agrupa por `(origen_id, timestamp_hora)`
   - Cuenta viajes (demanda)
   - Promedia clima, velocidad, etc.

2. **Feature Engineering:**
   - **Target:** `target = demanda(t+1)` (siguiente hora)
   - **Lags:** `lag_1h, lag_2h, lag_3h, lag_6h, lag_12h, lag_24h`
   - **Rolling stats:** `roll_mean_3h, roll_std_3h, roll_mean_24h, roll_std_24h`
   - **Media histórica:** `media_hist = mean(zona, hora, día_semana)` calculada SOLO de train

3. **Limpieza:**
   - Elimina filas sin target ni lags mínimos
   - Rellena lags largos con media histórica

**Entrada:**
```
data/processed/tlc_clean/problema1/raw/{train,val,test}.parquet
```

**Salida:**
```
data/processed/tlc_clean/problema1/features/
├── train.parquet (133K registros zona×hora)
├── val.parquet   (29K registros)
├── test.parquet  (27K registros)
└── metadata.json (lista features, target, stats demanda)
```

**Features generados (26 total):**
```python
FEATURES = [
    # Temporales
    'origen_id', 'hora', 'dia_semana', 'dia_mes', 'mes_num', 'es_finde',
    
    # Demanda
    'demanda',  # Demanda hora actual
    
    # Lags (autocorrelación)
    'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h',
    
    # Rolling (tendencias)
    'roll_mean_3h', 'roll_std_3h', 'roll_mean_24h', 'roll_std_24h',
    
    # Patrón histórico
    'media_hist',  # Media histórica (zona, hora, día_semana)
    
    # Contextuales
    'temp_c', 'precipitation', 'viento_kmh', 'velocidad_mph',
    'lluvia', 'nieve', 'es_festivo', 'num_eventos',
]

TARGET = 'target'  # Demanda siguiente hora
```

**Ejecución:**
```bash
uv run python src/modelos/problema1/agregacion.py
```

**Notas:**
- `media_hist` se calcula SOLO de train → sin data leakage
- Lags usan `shift()` → NO miran hacia el futuro

---

### **4. baseline.py**

**Función:** Entrena 3 modelos baseline para comparación con deep learning.

**Modelos implementados:**

1. **Naive (Persistencia):**
```
   demanda(t+1) = demanda(t)
```
   Asume que la demanda siguiente = actual

2. **Media Histórica:**
```
   demanda(t+1) = mean(zona, hora, día_semana)
```
   Usa patrón histórico promedio

3. **Random Forest:**
```
   RF(lag_1h, lag_2h, ..., clima, hora, ...) → demanda(t+1)
```
   Modelo ML con 200 árboles

**Entrada:**
```
data/processed/tlc_clean/problema1/features/{train,val,test}.parquet
```

**Salida:**
```
models/problema1/
├── baseline_random_forest.pkl (modelo entrenado)
├── baseline_results.json (métricas de los 3 modelos)
└── feature_importance.csv (importancia de cada feature)

reports/problema1/
└── baseline_resultados.png (gráficos comparativos)
```

**Métricas calculadas:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (R-squared)
- MAPE (Mean Absolute Percentage Error)

**Ejecución:**
```bash
uv run python src/modelos/problema1/baseline.py
```

---

### **5. modelo1_lstm.py**

**Función:** Entrena modelo LSTM (Long Short-Term Memory) para capturar patrones temporales complejos.

**Arquitectura:**
```
Input(lookback=12, features=25)
    ↓
LSTM(64) → BatchNorm → LSTM(32) → BatchNorm
    ↓
Dense(16, relu) → Dropout(0.2) → Dense(1)
    ↓
Output: demanda siguiente hora
```

**Características:**
- **Lookback:** 12 horas de historia
- **Secuencias:** Crea ventanas deslizantes por zona
- **Normalización:** StandardScaler (fit solo en train)
- **Early stopping:** Patience=10 epochs
- **ReduceLR:** Reduce learning rate si no mejora

**Entrada:**
```
data/processed/tlc_clean/problema1/features/{train,val,test}.parquet
```

**Salida:**
```
models/problema1/
├── lstm_model.keras (modelo entrenado)
├── scaler.pkl (normalizador)
├── zona_encoder.pkl (encoder zonas)
├── config.json (hiperparámetros + métricas)
└── training.png (curvas loss/MAE)
```

**Configuración:**
```python
CONFIG = {
    'lookback': 12,           # 12 horas de historia
    'lstm_units_1': 64,       # Primera capa LSTM
    'lstm_units_2': 32,       # Segunda capa LSTM
    'dense_units': 16,        # Capa densa
    'dropout': 0.2,           # Regularización
    'epochs': 50,             # Máximo epochs
    'batch_size': 64,         # Tamaño batch
    'learning_rate': 0.001,   # Learning rate inicial
    'patience': 10,           # Early stopping
}
```

**Ejecución:**
```bash
uv run python src/modelos/problema1/modelo1_lstm.py
```

**Nota:** Compatible con Python 3.14 usando Keras 3 + JAX backend.

---

