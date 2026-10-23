# CC5212-1 Procesamiento Masivo de Datos - Primavera 2026
# Proyecto: Analisis de caracteristicas musicales y similitud en Spotify (1.2M+ tracks)
# Grupo: Verde

import sys
from functools import reduce

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, BooleanType
)

# Columnas acusticas continuas para clustering y similitud
# Excluimos key y mode por ser categoricas/tonales
FEATURE_COLS = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo"
]

PERFIL_COLS = [
    "danceability", "energy", "loudness", "valence",
    "acousticness", "instrumentalness", "speechiness", "tempo"
]

Z_COLS = ["z_" + c for c in FEATURE_COLS]

K_CLUSTERS = 5
MAX_ITER = 20
TOLERANCIA = 1e-4
SEED = 42


def get_spark_session():
    spark = (
        SparkSession.builder
        .appName("SpotifyAnalysis_Verde")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def cargar_datos(spark, path_csv):
    # Esquema explicito para agilizar lectura y asegurar tipos
    schema = StructType([
        StructField("id", StringType()),
        StructField("name", StringType()),
        StructField("album", StringType()),
        StructField("album_id", StringType()),
        StructField("artists", StringType()),
        StructField("artist_ids", StringType()),
        StructField("track_number", IntegerType()),
        StructField("disc_number", IntegerType()),
        StructField("explicit", BooleanType()),
        StructField("danceability", DoubleType()),
        StructField("energy", DoubleType()),
        StructField("key", IntegerType()),
        StructField("loudness", DoubleType()),
        StructField("mode", IntegerType()),
        StructField("speechiness", DoubleType()),
        StructField("acousticness", DoubleType()),
        StructField("instrumentalness", DoubleType()),
        StructField("liveness", DoubleType()),
        StructField("valence", DoubleType()),
        StructField("tempo", DoubleType()),
        StructField("duration_ms", IntegerType()),
        StructField("time_signature", DoubleType()),
        StructField("year", IntegerType()),
        StructField("release_date", StringType()),
    ])

    df = (
        spark.read
        .option("header", True)
        .option("quote", '"')
        .option("escape", '"')
        .schema(schema)
        .csv(path_csv)
    )

    # Extraer anio de release_date y limpiar registros invalidos
    df = df.withColumn("release_year", F.substring(F.col("release_date"), 1, 4).cast("int"))
    df = df.filter(F.col("release_year").between(1920, 2026))
    df = df.dropna(subset=FEATURE_COLS + ["id", "name"])

    # Filtro basico de duracion para descartar ruidos menores a 30s o mayores a 30m
    df = df.filter(F.col("duration_ms").between(30000, 1800000))

    # Agrupar por decadas (ej. 1984 -> 1980)
    df = df.withColumn("decade", (F.floor(F.col("release_year") / 10) * 10).cast("int"))

    return df.cache()


def analisis_por_decadas(df, output_dir):
    print("\n>>> Analisis temporal por decadas")

    agregaciones = [
        F.count("*").alias("n_canciones"),
        F.round(F.avg(F.col("explicit").cast("double")) * 100, 2).alias("pct_explicit")
    ]
    for c in PERFIL_COLS:
        agregaciones.append(F.round(F.avg(c), 3).alias(f"avg_{c}"))

    resumen = df.groupBy("decade").agg(*agregaciones).orderBy("decade")
    resumen.show(20, truncate=False)

    if output_dir:
        resumen.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{output_dir}/resumen_decadas")
    return resumen


def estandarizar_features(df):
    # z-score: (x - media) / stddev en un solo pase
    exprs = []
    for c in FEATURE_COLS:
        exprs.extend([F.avg(c).alias(f"m_{c}"), F.stddev(c).alias(f"s_{c}")])
    stats = df.agg(*exprs).first()

    cols_z = []
    for c in FEATURE_COLS:
        media = stats[f"m_{c}"]
        std = stats[f"s_{c}"] or 1.0
        cols_z.append(((F.col(c) - F.lit(media)) / F.lit(std)).alias(f"z_{c}"))

    return df.select("*", *cols_z)


def calcular_distancia_cuadrada(centroide):
    terminos = [(F.col(z) - F.lit(float(val))) ** 2 for z, val in zip(Z_COLS, centroide)]
    return reduce(lambda a, b: a + b, terminos)


def asignar_clusters(df, centroides):
    # Calcula la distancia a cada centroide y toma el menor indice
    distancias = F.array(*[calcular_distancia_cuadrada(c) for c in centroides])
    df_temp = df.withColumn("_dists", distancias).withColumn("_dmin", F.array_min("_dists"))

    cond = F.when(F.col("_dists")[0] == F.col("_dmin"), 0)
    for k in range(1, len(centroides)):
        cond = cond.when(F.col("_dists")[k] == F.col("_dmin"), k)

    return df_temp.withColumn("cluster_implicito", cond.otherwise(0)).drop("_dists")


def kmeans_distribuido(df_z):
    # Tomar K puntos iniciales aleatorios
    muestra = df_z.orderBy(F.rand(SEED)).limit(K_CLUSTERS).select(*Z_COLS).collect()
    centroides = [[row[z] for z in Z_COLS] for row in muestra]

    costo_final = 0.0
    for it in range(1, MAX_ITER + 1):
        df_asignado = asignar_clusters(df_z, centroides)

        # Recalcular centroides promediando los puntos de cada cluster
        agg = (
            df_asignado.groupBy("cluster_implicito")
            .agg(
                F.sum("_dmin").alias("costo"),
                F.count("*").alias("count"),
                *[F.avg(z).alias(z) for z in Z_COLS]
            )
            .collect()
        )

        nuevos_centroides = [list(c) for c in centroides]
        costo_final = 0.0
        for r in agg:
            cid = r["cluster_implicito"]
            nuevos_centroides[cid] = [r[z] for z in Z_COLS]
            costo_final += r["costo"]

        # Medir cambio maximo en la posicion de los centroides
        delta = max(
            sum((c1 - c2) ** 2 for c1, c2 in zip(c_ant, c_pos)) ** 0.5
            for c_ant, c_pos in zip(centroides, nuevos_centroides)
        )
        centroides = nuevos_centroides
        print(f"Iteracion {it:2d} | Costo (WSSSE): {costo_final:,.2f} | Delta maximo: {delta:.6f}")

        if delta < TOLERANCIA:
            print(f"K-Means convergio tras {it} iteraciones.")
            break

    return centroides, costo_final


def clustering_analisis(df, output_dir):
    print(f"\n>>> Clustering K-Means (K={K_CLUSTERS})")
    df_z = estandarizar_features(df).cache()
    centroides, costo = kmeans_distribuido(df_z)

    df_clusterizado = asignar_clusters(df_z, centroides).drop("_dmin").cache()

    # Perfil medio por cluster
    print("\nPerfil promedio de cada cluster:")
    perfil = (
        df_clusterizado.groupBy("cluster_implicito")
        .agg(
            F.count("*").alias("n_canciones"),
            *[F.round(F.avg(c), 3).alias(f"avg_{c}") for c in PERFIL_COLS]
        )
        .orderBy("cluster_implicito")
    )
    perfil.show(truncate=False)

    # Distribucion por decada
    print("\nEvolucion de clusters a traves del tiempo:")
    evolucion = (
        df_clusterizado.groupBy("decade", "cluster_implicito")
        .count()
        .orderBy("decade", "cluster_implicito")
    )
    totales_decada = df_clusterizado.groupBy("decade").agg(F.count("*").alias("total"))
    pct_evolucion = (
        evolucion.join(totales_decada, "decade")
        .withColumn("pct", F.round(F.col("count") / F.col("total") * 100, 2))
        .select("decade", "cluster_implicito", "count", "pct")
        .orderBy("decade", "cluster_implicito")
    )
    pct_evolucion.show(50, truncate=False)

    if output_dir:
        perfil.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{output_dir}/perfil_clusters")
        pct_evolucion.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{output_dir}/evolucion_clusters")

    return df_clusterizado


def buscar_track_referencia(df_vec, query):
    """
    Busca una cancion de referencia:
    1. Si no hay query, retorna el primer registro disponible.
    2. Por ID exacto de Spotify.
    3. Por nombre de la cancion (insensible a mayusculas/minusculas).
    4. Por nombre de artista (insensible a mayusculas/minusculas).
    """
    if not query:
        return df_vec.first()

    # 1. Busqueda por ID exacto
    match = df_vec.filter(F.col("id") == query).first()
    if match:
        return match

    # 2. Busqueda por titulo de cancion
    match = df_vec.filter(F.lower(F.col("name")).contains(query.lower())).first()
    if match:
        return match

    # 3. Busqueda por artista
    match = df_vec.filter(F.lower(F.col("artists")).contains(query.lower())).first()
    if match:
        return match

    return None


def calcular_similitud(df_vec, ref, metrica="coseno", top_k=5):
    track_id = ref["id"]
    candidatos = df_vec.filter(F.col("id") != track_id)

    if metrica == "coseno":
        dot_product = reduce(lambda a, b: a + b, [F.col(z) * F.lit(float(ref[z])) for z in Z_COLS])
        similitud = dot_product / (F.col("_norma") * F.lit(float(ref["_norma"])))
        resultado = (
            candidatos.withColumn("score_similitud", F.round(similitud, 4))
            .orderBy(F.desc("score_similitud"))
        )
    else:  # euclidiana
        dist2 = reduce(lambda a, b: a + b, [(F.col(z) - F.lit(float(ref[z]))) ** 2 for z in Z_COLS])
        resultado = (
            candidatos.withColumn("distancia_euclidiana", F.round(F.sqrt(dist2), 4))
            .orderBy(F.asc("distancia_euclidiana"))
        )

    cols = ["id", "name", "artists", "decade", "cluster_implicito"]
    metric_col = "score_similitud" if metrica == "coseno" else "distancia_euclidiana"
    cols.append(metric_col)

    top = resultado.select(*cols).limit(top_k)
    top.show(top_k, truncate=False)
    return top


def main():
    if len(sys.argv) < 3:
        input_path = "spotify-sample-1k.csv"
        output_dir = "salida_spotify"
    else:
        input_path = sys.argv[1]
        output_dir = sys.argv[2]

    # Nombre de cancion, artista o ID opcional por argumento 3
    target_query = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"Procesando: {input_path}")
    print(f"Directorio de salida: {output_dir}")

    spark = get_spark_session()
    try:
        df = cargar_datos(spark, input_path)
        print(f"Total registros cargados y limpios: {df.count():,}")

        # 1. Analisis por decadas
        analisis_por_decadas(df, output_dir)

        # 2. Agrupamiento K-Means
        df_clusterizado = clustering_analisis(df, output_dir)

        # 3. Motor de similitud
        print("\n>>> Motor de similitud acustica")
        norma2 = reduce(lambda a, b: a + b, [F.col(z) ** 2 for z in Z_COLS])
        df_vec = df_clusterizado.withColumn("_norma", F.sqrt(norma2)).cache()

        ref = buscar_track_referencia(df_vec, target_query)
        if not ref:
            print(f"[!] No se encontro ninguna cancion que coincida con: '{target_query}'")
        else:
            print(f"\nCancion de referencia: '{ref['name']}' - {ref['artists']} (Decada {ref['decade']}, Cluster {ref['cluster_implicito']})")

            print("\nTop similares por Coseno:")
            calcular_similitud(df_vec, ref, metrica="coseno", top_k=5)

            print("\nTop similares por Distancia Euclidiana:")
            calcular_similitud(df_vec, ref, metrica="euclidiana", top_k=5)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
