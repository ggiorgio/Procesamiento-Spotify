# Análisis de Patrones de Producción Musical y Similitud de Tracks a Gran Escala con Spotify 1.2M+

**CC5212-1 Procesamiento Masivo de Datos — Primavera 2026**  
*Departamento de Ciencias de la Computación (DCC) — Facultad de Ciencias Físicas y Matemáticas, Universidad de Chile*  
**Profesor:** Aidan Hogan / Alberto Moya  
**Grupo:** Verde  

---

## Descripción del Proyecto

El objetivo principal de este proyecto es analizar más de un millón de canciones de Spotify mediante procesamiento distribuido para descubrir cómo ha evolucionado la producción musical a lo largo de las décadas y construir un motor de similitud acústica masiva.

### Ejes de Investigación:
1. **Evolución Temporal y *Loudness War*:** Agrupación distribuida por época y década ($1920\text{s}$ a $2020\text{s}$) para computar métricas acústicas cuantitativas (`loudness`, `energy`, `danceability`, `valence`, `acousticness` y `% explicit`) en un solo pase mediante `aggregateByKey` en Apache Spark.
2. **Similitud de Tracks a Gran Escala ($K$-Nearest Songs):** Búsqueda distribuida en el espacio acústico multidimensional normalizado $[0, 1]$ para recuperar en paralelo las canciones más similares a cualquier tema de consulta sobre el catálogo completo de 1.2M de pistas.

---

## Dataset

* **Origen:** [Spotify 1.2M+ Songs Dataset (Kaggle)](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs) (por Rodolfo Figueroa).
* **Volumen:** $1.204.025$ pistas y 24 atributos (metadatos y descriptores acústicos continuos).
* **Almacenamiento:** Los datos se encuentran almacenados y replicados en el clúster distribuido en HDFS (`/uhadoop2026/verde/spotify/tracks_features.csv`).

> *Nota de arquitectura:* Los datos crudos residen exclusivamente en HDFS y no se versionan en Git.

---

## Estructura del Repositorio

```
procesamiento-spotify/
├── .gitignore          # Reglas para excluir datasets masivos y logs temporales
├── README.md           # Informe técnico y documentación del proyecto
├── SpotifyAnalyzer.py  # Script distribuido principal (PySpark)
└── resultados.txt      # Resultados y respuestas a las consultas obligatorias
```

---

## Entorno de Ejecución

* **Servidor SSH:** `cm.dcc.uchile.cl:220` (nodo maestro `cluster-01`)
* **Spark Master:** `spark://cluster-01:7077`
* **Directorio HDFS:** `/uhadoop2026/verde/spotify/`

### Comandos de Ejecución

```bash
# 1. Ejecutar job distribuido sobre el dataset completo en Spark
spark-submit --master spark://cluster-01:7077 SpotifyAnalyzer.py \
  hdfs://cm:9000/uhadoop2026/verde/spotify/tracks_features.csv \
  hdfs://cm:9000/uhadoop2026/verde/spotify-full-py/

# 2. Extraer resultados y generar reporte
hdfs dfs -cat /uhadoop2026/verde/spotify-full-py/part-* > resultados.txt
```

---

## Integrantes del Grupo

* Barbara Gonzalez
* Ignacia Cejas
* Giorgio Bravo
