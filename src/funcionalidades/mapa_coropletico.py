import plotly.express as px
import pandas as pd
import geopandas as gpd
import json
from pathlib import Path
import os
import requests
import pyarrow.parquet as pq
import zipfile

# 1. CONFIGURACIÓN DE RUTAS
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.getenv("CONDUCIA_DATA_DIR", str(PROJECT_ROOT / "data")))

DATA_PATH = DATA_ROOT / "processed/tlc_clean/datos_final.parquet"
EXTERNAL_DIR = DATA_ROOT / "external"
SHAPEFILE_PATH = EXTERNAL_DIR / "taxi_zones.shp"
SHAPEFILE_PATH_ALT = EXTERNAL_DIR / "taxi_zones" / "taxi_zones.shp"
GEOJSON_LOCAL_PATH = EXTERNAL_DIR / "nyc_boroughs.geojson"
TAXI_ZONES_GEOJSON_PATH = EXTERNAL_DIR / "taxi_zones.geojson"
MAP_OUTPUT_PATH = DATA_ROOT / "funcionalidades" / "mapa_poder_barrios.html"

GEOJSON_URL = "https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/new-york-city-boroughs.geojson"
TAXI_ZONES_ZIP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

def obtener_geojson(url, local_path):
    """Descarga el GeoJSON si no existe y lo carga con codificación UTF-8."""
    if not local_path.exists():
        print(f"Descargando GeoJSON desde {url}...")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        with open(local_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
    
    with open(local_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def descargar_zip_taxi_zones():
    """Descarga y extrae taxi_zones.zip oficial de TLC en data/external."""
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = EXTERNAL_DIR / "taxi_zones.zip"
    print(f"Descargando Taxi Zones desde {TAXI_ZONES_ZIP_URL}...")
    response = requests.get(TAXI_ZONES_ZIP_URL, timeout=30)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        f.write(response.content)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(EXTERNAL_DIR)
    print("Taxi Zones descargado y extraído en data/external.")
    
def cargar_geojson_zonas_tlc(path):
    """Convierte el Shapefile de la TLC a GeoJSON."""
    # Intenta regenerar .shx si falta (GDAL/OGR).
    os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")
    gdf = gpd.read_file(path)
    # Si no trae CRS en el shapefile, asumimos WGS84 para visualización web.
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    # Asegurar que el sistema de coordenadas sea WGS84 para Plotly
    if str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return json.loads(gdf.to_json())


def seleccionar_shapefile_tlc() -> Path | None:
    """Elige el shapefile TLC más completo disponible."""
    candidatos = [SHAPEFILE_PATH_ALT, SHAPEFILE_PATH]
    for shp in candidatos:
        if not shp.exists():
            continue
        dbf = shp.with_suffix(".dbf")
        shx = shp.with_suffix(".shx")
        if dbf.exists() and shx.exists():
            return shp
    for shp in candidatos:
        if shp.exists():
            return shp
    return None


def geojson_tiene_location_id(geojson_obj: dict) -> bool:
    try:
        features = geojson_obj.get("features", [])
        if not features:
            return False
        props = features[0].get("properties", {})
        return "LocationID" in props
    except Exception:
        return False


def obtener_geojson_zonas_tlc():
    """
    Intenta cargar zonas TLC desde shapefile local.
    Si no hay LocationID, usa fallback GeoJSON oficial cacheado.
    """
    shp_path = seleccionar_shapefile_tlc()
    if shp_path is not None:
        try:
            geo = cargar_geojson_zonas_tlc(shp_path)
            if geojson_tiene_location_id(geo):
                return geo
            print(
                "⚠️ Shapefile sin 'LocationID'. "
                "Intentando descargar shapefile oficial TLC..."
            )
        except Exception as e:
            print(f"⚠️ Error leyendo shapefile TLC ({e}). Intentando descarga oficial...")

    try:
        descargar_zip_taxi_zones()
        shp_path = seleccionar_shapefile_tlc()
        if shp_path is not None:
            geo = cargar_geojson_zonas_tlc(shp_path)
            if geojson_tiene_location_id(geo):
                return geo
        # Ultimo fallback: geojson local si existe y tiene LocationID
        if TAXI_ZONES_GEOJSON_PATH.exists():
            geo = obtener_geojson("", TAXI_ZONES_GEOJSON_PATH)
            if geojson_tiene_location_id(geo):
                return geo
        print("⚠️ No hay geometría TLC con LocationID tras los intentos de recuperación.")
        return None
    except Exception as e:
        print(f"⚠️ No se pudo recuperar geometría TLC detallada: {e}")
        return None

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

def calcular_indice_economico_parquet(path, columna):
    """Calcula el indice leyendo solo columnas necesarias por row groups."""
    parquet = pq.ParquetFile(path)
    cols = set(parquet.schema_arrow.names)
    necesarias = {columna, "propina", "num_pasajeros"}
    if not necesarias.issubset(cols):
        raise ValueError(
            f"{path} no contiene columnas necesarias: {sorted(necesarias)}"
        )

    partes = []
    for i in range(parquet.metadata.num_row_groups):
        chunk = parquet.read_row_group(
            i, columns=[columna, "propina", "num_pasajeros"]
        ).to_pandas()
        chunk = chunk.dropna(subset=[columna])
        stats = (
            chunk.groupby(columna, observed=True)
            .agg(
                propina_sum=("propina", "sum"),
                pasajeros_sum=("num_pasajeros", "sum"),
                volumen_viajes=(columna, "size"),
            )
            .reset_index()
        )
        partes.append(stats)

    if not partes:
        return pd.DataFrame(columns=[columna, "Poder_Adquisitivo"])

    agregado = (
        pd.concat(partes, ignore_index=True)
        .groupby(columna, observed=True)
        .agg(
            propina_sum=("propina_sum", "sum"),
            pasajeros_sum=("pasajeros_sum", "sum"),
            volumen_viajes=("volumen_viajes", "sum"),
        )
        .reset_index()
    )

    agregado["propina_media"] = agregado["propina_sum"] / agregado["volumen_viajes"]
    agregado["pasajeros_medios"] = agregado["pasajeros_sum"] / agregado["volumen_viajes"]
    agregado["Poder_Adquisitivo"] = (
        normalize(agregado["propina_media"]) * 0.4
        + normalize(agregado["pasajeros_medios"]) * 0.3
        + normalize(agregado["volumen_viajes"]) * 0.3
    ) * 100
    return agregado[[columna, "Poder_Adquisitivo"]]

def generar_mapa_plotly():
    # 1. Cargar geometrías
    print("Cargando geometrías...")
    geo_barrios = obtener_geojson(GEOJSON_URL, GEOJSON_LOCAL_PATH)
    geo_zonas_tlc = obtener_geojson_zonas_tlc()
    if geo_zonas_tlc is None:
        print("⚠️ No habrá vista detallada TLC. Se mostrará solo barrios.")

    # 2. Calcular índice económico
    print("Calculando métricas por barrios y zonas (modo eficiente)...")
    stats_barrios = calcular_indice_economico_parquet(DATA_PATH, "origen_barrio")
    stats_zonas = calcular_indice_economico_parquet(DATA_PATH, "origen_id")

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
    botones = [
        dict(
            label="Ver por Barrios",
            method="restyle",
            args=[{
                "z": [stats_barrios["Poder_Adquisitivo"].tolist()],
                "locations": [stats_barrios["origen_barrio"].tolist()],
                "geojson": geo_barrios,
                "featureidkey": "properties.name"
            }]
        )
    ]
    if geo_zonas_tlc is not None:
        stats_zonas = stats_zonas.copy()
        stats_zonas["origen_id"] = stats_zonas["origen_id"].astype(int)
        botones.append(
            dict(
                label="Ver por Zonas TLC (Detallado)",
                method="restyle",
                args=[{
                    "z": [stats_zonas["Poder_Adquisitivo"].tolist()],
                    "locations": [stats_zonas["origen_id"].astype(str).tolist()],
                    "geojson": geo_zonas_tlc,
                    "featureidkey": "properties.LocationID"
                }]
            )
        )

    fig.update_layout(
        title={'text': "Análisis de Poder Adquisitivo NYC", 'x': 0.5, 'xanchor': 'center'},
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.5, xanchor="center", y=1.05,
                buttons=botones
            )
        ],
        margin={"r":0, "t":80, "l":0, "b":0}
    )

    # Guardar (sin fig.show para evitar problemas en entorno headless/Docker)
    MAP_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(MAP_OUTPUT_PATH)
    print(f"Mapa guardado en: {MAP_OUTPUT_PATH}")

if __name__ == "__main__":
    generar_mapa_plotly()
