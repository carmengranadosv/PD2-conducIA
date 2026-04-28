import plotly.express as px
import pandas as pd
import geopandas as gpd
import json
from pathlib import Path
import requests

# 1. CONFIGURACIÓN DE RUTAS
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data/processed/tlc_clean/datos_final.parquet"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
SHAPEFILE_PATH = EXTERNAL_DIR / "taxi_zones.shp"
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
    
def cargar_geojson_zonas_tlc(path):
    """Convierte el Shapefile de la TLC a GeoJSON."""
    gdf = gpd.read_file(path)
    # Asegurar que el sistema de coordenadas sea WGS84 para Plotly
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return json.loads(gdf.to_json())

def normalize(col):
    """Función para normalizar una columna entre 0 y 1"""
    return (col - col.min()) / (col.max() - col.min()) if (col.max() - col.min()) != 0 else 0
    
def calcular_indice_economico(df, columna):
    """
    Calcula el 'Poder Adquisitivo' por zona basado en:
    - Propinas medias
    - Número de pasajeros
    - Volumen de viajes
    """
    # 1. Agrupar datos por zona de origen y calcular las métricas
    stats = df.groupby(columna).agg(
        propina_media=("propina", "mean"),
        pasajeros_medios=("num_pasajeros", "mean"),
        volumen_viajes=("origen_barrio", "count")
    ).reset_index()  

    stats["Poder_Adquisitivo"] = (
        normalize(stats["propina_media"]) * 0.4 + 
        normalize(stats["pasajeros_medios"]) * 0.3 + 
        normalize(stats["volumen_viajes"]) * 0.3
    ) * 100
    
    return stats

def generar_mapa_plotly():

    # 1. Cargar datos 
    print("Cargando datos y procesando geometrías...")
    df = pd.read_parquet(DATA_PATH)
    geo_barrios = obtener_geojson(GEOJSON_URL, GEOJSON_LOCAL_PATH)
    geo_zonas_tlc = cargar_geojson_zonas_tlc(SHAPEFILE_PATH)

    # 2. Calcular índice económico
    stats_barrios = calcular_indice_economico(df, "origen_barrio")
    stats_zonas = calcular_indice_economico(df, "origen_id")

    # 6. CREAR EL MAPA COROPLÉTICO
    print("Generando mapa interactivo...")
    # Iniciamos con la vista de Barrios
    fig = px.choropleth_map(
        stats_barrios,
        geojson=geo_barrios,
        locations="origen_barrio",
        featureidkey="properties.name",
        color="Poder_Adquisitivo",
        color_continuous_scale="Viridis",
        range_color=(0, 100),
        map_style="carto-positron",
        zoom=10, center={"lat": 40.7128, "lon": -74.0060},
        opacity=0.7,
        title="Análisis de Poder Adquisitivo NYC"
    )

    # Añadimos los botones para cambiar entre Barrios y Zonas TLC
    fig.update_layout(
        title={'text': "Análisis de Poder Adquisitivo NYC", 'x': 0.5, 'xanchor': 'center'},
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.5, xanchor="center", y=1.05,
                buttons=[
                    dict(
                        label="Ver por Barrios",
                        method="restyle",
                        args=[{
                            "z": [stats_barrios["Poder_Adquisitivo"].tolist()],
                            "locations": [stats_barrios["origen_barrio"].tolist()],
                            "geojson": geo_barrios,
                            "featureidkey": "properties.name"
                        }]
                    ),
                    dict(
                        label="Ver por Zonas TLC (Detallado)",
                        method="restyle",
                        args=[{
                            "z": [stats_zonas["Poder_Adquisitivo"].tolist()],
                            "locations": [stats_zonas["origen_id"].tolist()],
                            "geojson": geo_zonas_tlc,
                            "featureidkey": "properties.LocationID" # Clave del Shapefile que subiste
                        }]
                    )
                ]
            )
        ],
        margin={"r":0, "t":80, "l":0, "b":0}
    )

    # Mostrar y Guardar
    fig.write_html(MAP_OUTPUT_PATH)
    print(f"Mapa guardado en: {MAP_OUTPUT_PATH}")
    fig.show()

if __name__ == "__main__":
    generar_mapa_plotly()