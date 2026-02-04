COLUMNAS_YELLOW = {
    # Identificación / proveedor
    #"VendorID": "vendor_id",

    # Fechas
    "tpep_pickup_datetime": "fecha_inicio",
    "tpep_dropoff_datetime": "fecha_fin",

    # Zonas
    "PULocationID": "origen_id",
    "DOLocationID": "destino_id",

    # Viaje
    "passenger_count": "num_pasajeros",
    "trip_distance": "distancia",

    # Tarifas y precios
    "fare_amount": "tarifa_base",
    "extra": "extra",
    "mta_tax": "mta_tax",
    "tip_amount": "propina",
    "tolls_amount": "peajes",
    "ehail_fee": "ehail_fee",
    "improvement_surcharge": "recargo_mejora",
    "congestion_surcharge": "recargo_congestion",
    #"cbd_congestion_fee": "recargo_cbd",
    "total_amount": "precio_total",

    # Pago
    "payment_type": "tipo_pago",

    # Tarifas especiales
    "RatecodeID": "codigo_tarifa",

    # Otros
    #"store_and_fwd_flag": "store_and_fwd",
    "trip_type": "tipo_viaje",
}


COLUMNAS_FHVHV = {
    # Plataforma
    "hvfhs_license_num": "plataforma",
    "dispatching_base_num": "base_despacho",
    "originating_base_num": "base_origen",

    # Fechas
    "request_datetime": "fecha_solicitud",
    "on_scene_datetime": "fecha_llegada_conductor",
    "pickup_datetime": "fecha_inicio",
    "dropoff_datetime": "fecha_fin",

    # Zonas
    "PULocationID": "origen_id",
    "DOLocationID": "destino_id",

    # Viaje
    "trip_miles": "distancia",
    "trip_time": "duracion_seg",

    # Precios y pagos
    "base_passenger_fare": "tarifa_base",
    "tolls": "peajes",
    "bcf": "black_car_fund",
    "sales_tax": "impuesto_ventas",
    "congestion_surcharge": "recargo_congestion",
    "airport_fee": "recargo_aeropuerto",
    "cbd_congestion_fee": "recargo_cbd",
    "tips": "propina",
    "driver_pay": "pago_conductor",

    # Viajes compartidos
    "shared_request_flag": "solicitud_compartida",
    "shared_match_flag": "viaje_compartido",

    # Accesibilidad
    "access_a_ride_flag": "access_a_ride",
    "wav_request_flag": "wav_solicitado",
    "wav_match_flag": "wav_realizado",
}
