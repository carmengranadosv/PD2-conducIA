import plotly.express as px
import pandas as pd
import json
from pathlib import Path
import requests

# 1. CONFIGURACIÓN DE RUTAS
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/processed/tlc_clean/datos_final.parquet"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
GEOJSON_LOCAL_PATH = EXTERNAL_DIR / "nyc_boroughs.geojson"
MAP_OUTPUT_PATH = PROJECT_ROOT / "data" / "funcionalidades" / "mapa_poder_barrios.html"

GEOJSON_URL = "https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/new-york-city-boroughs.geojson"

def obtener_geojson(url, local_path):
    """Descarga el GeoJSON si no existe y lo carga con codificación UTF-8."""
    if not local_path.exists():
        print(f"Descargando GeoJSON desde {url}...")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url)
        with open(local_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
    
    with open(local_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def calcular_indice_economico(df):
    """
    Calcula el 'Poder Adquisitivo' por zona basado en:
    - Propinas medias
    - Número de pasajeros
    - Volumen de viajes
    """
    # 1. Agrupar datos por zona de origen y calcular las métricas 
    stats = df.groupby("origen_barrio").agg(
        propina_media=("propina", "mean"),
        pasajeros_medios=("num_pasajeros", "mean"),
        volumen_viajes=("origen_barrio", "count")
    ).reset_index()  

    return stats

def normalize(col):
    """Función para normalizar una columna entre 0 y 1"""
    return (col - col.min()) / (col.max() - col.min())

def generar_mapa_plotly():

    # 1. Cargar datos 
    df = pd.read_parquet(DATA_PATH)
    geojson_data = obtener_geojson(GEOJSON_URL, GEOJSON_LOCAL_PATH)

    # 2. Calcular índice económico
    stats = calcular_indice_economico(df)

    # 3. Normalizar y calcular índice final
    stats["n_propina"] = normalize(stats["propina_media"])
    stats["n_pasajeros"] = normalize(stats["pasajeros_medios"])
    stats["n_volumen"] = normalize(stats["volumen_viajes"])

    # Calculamos el índice en una escala de 0 a 100
    stats["Poder_Adquisitivo"] = (
        stats["n_propina"] * 0.4 + 
        stats["n_pasajeros"] * 0.3 + 
        stats["n_volumen"] * 0.3
    ) * 100

    # 6. CREAR EL MAPA COROPLÉTICO
    print("Generando mapa interactivo...")
    fig = px.choropleth_map(
        stats,
        geojson=geojson_data,
        locations="origen_barrio",      # Columna del DataFrame
        featureidkey="properties.name", # Clave dentro del JSON de GitHub
        color="Poder_Adquisitivo",
        color_continuous_scale="Viridis",
        range_color=(0, 100),           # Al ser barrios, la escala es más estable
        map_style="carto-positron",
        zoom=9, 
        center={"lat": 40.7128, "lon": -74.0060},
        opacity=0.7,
        hover_data={
            "origen_barrio": True,
            "propina_media": ":.2f",
            "volumen_viajes": True,
            "Poder_Adquisitivo": ":.1f"
        },
        labels={
            "Poder_Adquisitivo": "Índice de Poder",
            "propina_media": "Propina Media ($)",
            "origen_barrio": "Barrio"
        },
        title="Análisis de Poder Adquisitivo por Barrio (NYC)"
    )

    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})

    # Mostrar y Guardar
    fig.write_html(MAP_OUTPUT_PATH)
    print(f"Mapa guardado en: {MAP_OUTPUT_PATH}")
    fig.show()

if __name__ == "__main__":
    generar_mapa_plotly()