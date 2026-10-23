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


def buscar_candidatos(df_vec, query, limite=5):
    """
    Busca hasta `limite` canciones candidatas:
    1. Por ID exacto de Spotify.
    2. Por coincidencia en artistas (ej. 'Queen').
    3. Por titulo de cancion.
    """
    q = query.strip().lower()

    # 1. Busqueda por ID exacto
    match_id = df_vec.filter(F.col("id") == query).limit(1).collect()
    if match_id:
        return match_id

    # 2. Busqueda por coincidencia en artistas (si el query coincide con un artista)
    match_artista = df_vec.filter(F.lower(F.col("artists")).contains(q)).limit(limite).collect()
    if match_artista:
        match_nombre_exacto = df_vec.filter(F.lower(F.col("name")) == q).limit(limite).collect()
        if match_nombre_exacto:
            return match_nombre_exacto
        return match_artista

    # 3. Busqueda por titulo de cancion (contiene texto)
    match_nombre = df_vec.filter(F.lower(F.col("name")).contains(q)).limit(limite).collect()
    if match_nombre:
        return match_nombre

    return []


def hay_terminal():
    """Detecta si hay una terminal de control (consola interactiva) disponible."""
    try:
        with open("/dev/tty", "r"):
            return True
    except Exception:
        return sys.stdin.isatty()


def leer_terminal(prompt=""):
    """
    Lee entrada del usuario directamente desde /dev/tty para evitar que
    Spark/Java intercepte o cierre el flujo de stdin.
    """
    try:
        with open("/dev/tty", "r") as tty_in, open("/dev/tty", "w") as tty_out:
            tty_out.write(prompt)
            tty_out.flush()
            line = tty_in.readline()
            return line.strip() if line else ""
    except Exception:
        try:
            return input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return ""


def seleccionar_track(candidatos, es_interactivo=True):
    """
    Permite seleccionar una cancion si hay multiples coincidencias.
    En modo interactivo muestra un menu de opciones (1 a N).
    """
    if not candidatos:
        return None
    if len(candidatos) == 1 or not es_interactivo:
        return candidatos[0]

    print("\nSe encontraron varias canciones:")
    for idx, c in enumerate(candidatos, 1):
        print(f"  [{idx}] '{c['name']}' - {c['artists']} ({c['release_year']}) [Album: {c['album']}]")

    eleccion = leer_terminal(f"Elige una cancion (1-{len(candidatos)}) [por defecto 1]: ")
    if eleccion.isdigit() and 1 <= int(eleccion) <= len(candidatos):
        return candidatos[int(eleccion) - 1]

    return candidatos[0]


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


def ejecutar_consulta_similitud(df_vec, ref):
    print(f"\nCancion de referencia: '{ref['name']}' - {ref['artists']} (Decada {ref['decade']}, Cluster {ref['cluster_implicito']})")
    print("\nTop 5 similares por Coseno:")
    calcular_similitud(df_vec, ref, metrica="coseno", top_k=5)
    print("\nTop 5 similares por Distancia Euclidiana:")
    calcular_similitud(df_vec, ref, metrica="euclidiana", top_k=5)


def modo_interactivo(df_vec):
    print("\n" + "=" * 60)
    print("MOTOR INTERACTIVO DE SIMILITUD ACUSTICA")
    print("Ingresa el nombre de una cancion o artista (o escribe 'salir' para terminar):")
    print("=" * 60)

    while True:
        query = leer_terminal("\nBuscar cancion o artista > ")
        if not query or query.lower() in ["salir", "exit", "quit", "q"]:
            print("Finalizando sesion interactiva.")
            break

        candidatos = buscar_candidatos(df_vec, query)
        if not candidatos:
            print(f"[!] No se encontro ninguna cancion para: '{query}'")
            continue

        ref = seleccionar_track(candidatos, es_interactivo=True)
        ejecutar_consulta_similitud(df_vec, ref)


def main():
    if len(sys.argv) < 3:
        input_path = "spotify-sample-1k.csv"
        output_dir = "salida_spotify"
    else:
        input_path = sys.argv[1]
        output_dir = sys.argv[2]

    # Si se pasa 'none', '-' o 'false', se desactiva la exportacion de archivos
    if output_dir and output_dir.lower() in ["none", "null", "no", "false", "-"]:
        output_dir = None

    # Nombre de cancion, artista o ID opcional por argumento 3
    target_query = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"Procesando: {input_path}")
    print(f"Directorio de salida: {output_dir if output_dir else '(Deshabilitado: no se generaran archivos)'}")

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

        if target_query:
            candidatos = buscar_candidatos(df_vec, target_query)
            if not candidatos:
                print(f"[!] No se encontro ninguna cancion que coincida con: '{target_query}'")
            else:
                ref = seleccionar_track(candidatos, es_interactivo=hay_terminal())
                ejecutar_consulta_similitud(df_vec, ref)
        elif hay_terminal():
            modo_interactivo(df_vec)
        else:
            # En ejecucion no interactiva sin argumentos, toma la primera por defecto
            ref = df_vec.first()
            ejecutar_consulta_similitud(df_vec, ref)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
