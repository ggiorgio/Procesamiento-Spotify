# Análisis de Patrones de Producción Musical y Similitud de Tracks a Gran Escala con Spotify 1.2M+

**CC5212-1 Procesamiento Masivo de Datos — Primavera 2026**  
*Departamento de Ciencias de la Computación (DCC) — Facultad de Ciencias Físicas y Matemáticas, Universidad de Chile*  
**Profesor:** Alberto Moya  
**Grupo:** Verde  

**Integrantes:**
* Barbara Gonzalez
* Ignacia Cejas
* Giorgio Bravo

---

## Descripción del Proyecto

El objetivo principal de este proyecto es analizar un catálogo masivo de más de 1.2 millones de canciones de Spotify mediante procesamiento distribuido sobre Apache Spark. El proyecto aborda la evolución histórica de patrones acústicos de producción musical (como el fenómeno del *Loudness War*), la segmentación de perfiles sonoros mediante clustering no supervisado a gran escala y la construcción de un motor de recomendación acústica basado en similitud vectorial.

### Ejes Principales de Análisis:
1. **Evolución Temporal y *Loudness War*:** Agrupación distribuida por décadas (1920 a 2020) calculando agregaciones estadísticas de volumen (`loudness`), energía (`energy`), bailabilidad (`danceability`), valencia musical (`valence`), acústica (`acousticness`) y porcentaje de contenido explícito (`% explicit`).
2. **Clustering Acústico K-Means Distribuido:** Implementación de Lloyd's K-Means optimizado de forma nativa en Spark SQL (estandarización z-score, cálculo matricial distribuido y minimización de costo WSSSE), sin dependencia de bibliotecas externas como NumPy o Spark MLlib.
3. **Motor de Similitud Acústica (Top-$K$):** Búsqueda vectorial distribuida sobre las 1.2M+ canciones en el espacio normalizado, evaluando tanto **Similitud Coseno** (vía producto punto y norma $L_2$) como **Distancia Euclidiana**.

---

## Dataset

* **Origen:** [Spotify 1.2M+ Songs Dataset (Kaggle)](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs) (por Rodolfo Figueroa).
* **Volumen:** 1.204.025 pistas con 24 columnas (metadatos y descriptores acústicos continuos).
* **Almacenamiento en Clúster:** Reside replicado en el sistema de archivos distribuido HDFS en `/uhadoop2026/verde/spotify/tracks_features.csv` junto al subconjunto de prueba `/uhadoop2026/verde/spotify/spotify-sample-1k.csv`.

> *Nota de arquitectura:* Por tamaño de archivo, los datos crudos residen exclusivamente en HDFS y no se versionan en Git.

---

## Estructura del Repositorio

```
procesamiento-spotify/
├── .gitignore          # Exclusión de datasets crudos (.csv, .zip) y logs temporales
├── README.md           # Informe técnico y guía de reproducción
└── script_spotify.py   # Pipeline distribuido en PySpark (Limpieza, K-Means y Similitud)
```

### Directorios de Salida Generados (HDFS y Local):
* `resumen_decadas/`: Promedios históricos de descriptores acústicos y % explicit por década.
* `perfil_clusters/`: Centroides y promedios característicos de cada uno de los K=5 clústeres sonoros.
* `evolucion_clusters/`: Distribución y porcentaje de presencia de cada clúster a lo largo del tiempo.

---

## Entorno de Ejecución

* **Servidor SSH:** `cm.dcc.uchile.cl:220` (nodo maestro `cluster-01`)
* **Directorio de Trabajo Local (LFS):** `~/projects/verde/`
* **Directorio Distribuido (HDFS):** `/uhadoop2026/verde/`
* **Workers:** 5 nodos (`cluster-01` a `cluster-05`, 32 núcleos en total)
* **Master Spark:** `spark://cluster-01:7077`
* **Particionado:** 64 particiones de shuffle adaptativas (`spark.sql.shuffle.partitions = 64`)

---

## Guía Paso a Paso para Reproducir el Proyecto

### Paso 1: Subir el script al clúster (desde la máquina local)
Desde la raíz del repositorio local en tu computador:
```bash
scp -P 220 script_spotify.py uhadoop@cm.dcc.uchile.cl:~/projects/verde/
```

### Paso 2: Conectarse al clúster por SSH
```bash
ssh -p 220 uhadoop@cm.dcc.uchile.cl
cd ~/projects/verde
```

### Paso 3: Carga inicial de datos en HDFS (Solo requerida la primera vez)
Si los datos aún no están en HDFS, se cargan desde el sistema de archivos local del servidor:
```bash
hdfs dfs -mkdir -p /uhadoop2026/verde/spotify/
hdfs dfs -put -f tracks_features.csv /uhadoop2026/verde/spotify/
hdfs dfs -put -f spotify-sample-1k.csv /uhadoop2026/verde/spotify/
```

### Paso 4: Prueba rápida de verificación con el Sample (~30 segundos)
Antes de procesar los 1.2M de registros, se valida el pipeline con la muestra de 1.000 filas:

1. Limpiar directorio de prueba previo en HDFS:
   ```bash
   hdfs dfs -rm -r -f /uhadoop2026/verde/salida_spotify_test/
   ```
2. Ejecutar el script:
   ```bash
   spark-submit --master spark://cluster-01:7077 script_spotify.py \
     /uhadoop2026/verde/spotify/spotify-sample-1k.csv \
     /uhadoop2026/verde/salida_spotify_test/
   ```

### Paso 5: Ejecución distribuida a gran escala (1.2M+ registros)
Para procesar el catálogo completo de 1.2 millones de canciones:

1. (Opcional) Iniciar una sesión en `screen` para asegurar que el trabajo no se cancele ante desconexiones de SSH:
   ```bash
   screen -S spotify_run
   ```

2. Limpiar directorio de salida final en HDFS:
   ```bash
   hdfs dfs -rm -r -f /uhadoop2026/verde/salida_spotify_full/
   ```

3. Lanzar el trabajo distribuido en Spark:
   ```bash
   spark-submit --master spark://cluster-01:7077 script_spotify.py \
     /uhadoop2026/verde/spotify/tracks_features.csv \
     /uhadoop2026/verde/salida_spotify_full/
   ```

4. (Opcional, si se utilizó screen) Controlar la sesión en segundo plano:
   * Desconectarse de la pantalla (*detach*): Presionar `Ctrl + A` y luego `D`.
   * Volver a conectarse a la sesión (*reattach*): `screen -r spotify_run`
   * Cerrar la sesión una vez concluido el trabajo: `exit`.

### Paso 6: Traer y revisar los resultados en el almacenamiento local
Una vez finalizado el trabajo en Spark, se pueden transferir los resultados desde HDFS al disco local del clúster:
```bash
cd ~/projects/verde
rm -rf salida_spotify_full
hdfs dfs -copyToLocal /uhadoop2026/verde/salida_spotify_full/ ./
```

Para inspeccionar las tablas de resultados consolidadas:
```bash
# 1. Resumen temporal de la evolucion acustica y Loudness War
cat salida_spotify_full/resumen_decadas/part-*.csv

# 2. Perfil sonoro medio de cada cluster acustico
cat salida_spotify_full/perfil_clusters/part-*.csv

# 3. Evolucion porcentual de los clusteres a lo largo de las decadas
cat salida_spotify_full/evolucion_clusters/part-*.csv
```

### Paso 7: Consulta personalizada de similitud para un Track específico
El script permite ingresar opcionalmente el `id` de cualquier canción en el tercer argumento para buscar sus K canciones más cercanas:
```bash
spark-submit --master spark://cluster-01:7077 script_spotify.py \
  /uhadoop2026/verde/spotify/tracks_features.csv \
  /uhadoop2026/verde/salida_spotify_full/ \
  4cOdK2wGLETKBW3PvgPWqT
```
