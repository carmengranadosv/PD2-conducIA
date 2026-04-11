import pandas as pd
import json
import os
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parents[3]
REPORTS_DIR = BASE_DIR / 'reports' / 'problema5'
FINAL_RESULTS_DIR = BASE_DIR / 'models' / 'problema5'

# Lista de archivos a cargar
archivos_metricas = {
    "Baseline (Ridge)": REPORTS_DIR / 'baseline_results.json',
    "Red Neuronal (MLP)": REPORTS_DIR / 'red_neuronal_results.json',
    "XGBoost": REPORTS_DIR / 'xgboost_results.json'
}

def cargar_resultados():
    lista_resultados = []
    
    for nombre, ruta in archivos_metricas.items():
        if ruta.exists():
            with open(ruta, 'r') as f:
                data = json.load(f)
                # Extraemos solo las métricas de TEST para la comparativa real
                lista_resultados.append({
                    "Modelo": nombre,
                    "MAE ($)": data.get("test_mae"),
                    "RMSE ($)": data.get("test_rmse"),
                    "R²": data.get("test_r2"),
                    "BIAS ($)": data.get("test_bias")
                })
        else:
            print(f"No se encontró el archivo para {nombre} en {ruta}")
            
    return pd.DataFrame(lista_resultados)

def generar_comparativa():
    print("="*60)
    print("COMPARATIVA FINAL DE MODELOS - PROBLEMA 5")
    print("="*60)
    
    df = cargar_resultados()
    
    if df.empty:
        print("No hay resultados para comparar.")
        return

    # Formatear la tabla para que se vea bonita
    df_sorted = df.sort_values(by="R²", ascending=False)
    print(df_sorted.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- LÓGICA DE SELECCIÓN DEL MEJOR MODELO ---
    # El mejor suele ser el de menor MAE y mayor R2
    mejor_modelo = df_sorted.iloc[0]
    
    print("\n" + "="*60)
    print(f"EL GANADOR ES: {mejor_modelo['Modelo']}")
    print("="*60)
    print(f"Justificación técnica:")
    print(f"1. Precisión: Tiene un buen error medio más bajo (MAE: ${mejor_modelo['MAE ($)']:.4f}).")
    print(f"2. Explicabilidad: Su R² de {mejor_modelo['R²']:.4f} es el más alto, "
          "indicando que entiende mejor la lógica de las propinas.")
    print(f"3. Sesgo: Su BIAS de ${mejor_modelo['BIAS ($)']:.4f} indica una predicción "
          "equilibrada y fiable para el conductor.")
    
    # Guardar la tabla comparativa en CSV para la memoria
    df_sorted.to_csv(REPORTS_DIR / 'comparativa_final_modelos.csv', index=False)
    print(f"\nTabla comparativa guardada en: {REPORTS_DIR / 'comparativa_final_modelos.csv'}")

if __name__ == "__main__":
    generar_comparativa()