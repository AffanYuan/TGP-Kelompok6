from pyspark.sql import SparkSession, functions as F, types as T

# ============================================================
# 0. KONFIGURASI DASAR
# ============================================================

# Nama bucket di MinIO (buat manual di console MinIO)
BUCKET = "data-tgp"

# Lokasi file di MinIO (harus sudah kamu upload)
SUPERSTORE_KEY = "raw/superstore/Sample - Superstore.csv"
TWCS_KEY       = "raw/tweets/twcs.csv"

superstore_path = f"s3a://{BUCKET}/{SUPERSTORE_KEY}"
tweets_path     = f"s3a://{BUCKET}/{TWCS_KEY}"

# PostgreSQL (lihat docker-compose kamu)
PG_URL = "jdbc:postgresql://postgres:5432/tgp"
PG_PROPERTIES = {
    "user": "admin",
    "password": "admin123",
    "driver": "org.postgresql.Driver",  # pastikan jar driver ada di Spark
}

# Kandidat nama kolom di TWCS (supaya fleksibel)
TWEET_TEXT_CANDIDATES = ["text", "tweet_text", "body"]
TWEET_CREATED_AT_CANDIDATES = ["created_at", "created", "tweet_created_at"]
COMPANY_COL_CANDIDATES = ["company", "brand", "handle"]


def pick_col(columns, candidates, what):
    """
    Ambil nama kolom pertama yang ketemu dari daftar kandidat.
    """
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(
        f"Tidak menemukan kolom {what}. Kandidat: {candidates}. "
        f"Kolom yang ada: {columns}"
    )


# ============================================================
# 1. SPARK SESSION + KONFIG MINIO (S3A)
# ============================================================

spark = (
    SparkSession.builder
    .appName("tgp_etl_final")
    # Konfigurasi MinIO via S3A
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
    .config("spark.hadoop.fs.s3a.access.key", "admin")
    .config("spark.hadoop.fs.s3a.secret.key", "admin12345")
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("INFO")

# ============================================================
# 2. BRONZE LAYER – BACA RAW DATA DARI MINIO (CSV)
# ============================================================

print("=== BACA SUPERSTORE DARI MINIO (CSV) ===")
superstore_raw = spark.read.csv(
    superstore_path,
    header=True,
    inferSchema=True
)

print("=== BACA TWCS DARI MINIO (CSV) ===")
tweets_raw = spark.read.csv(
    tweets_path,
    header=True,
    inferSchema=True
)

# Simpan sebagai Bronze Parquet (opsional tapi rapi)
print("=== SIMPAN BRONZE KE MINIO ===")
superstore_raw.write.mode("overwrite").parquet(f"s3a://{BUCKET}/bronze/superstore")
tweets_raw.write.mode("overwrite").parquet(f"s3a://{BUCKET}/bronze/twcs")

# ============================================================
# 3. SILVER LAYER – CLEANING & TYPING
# ============================================================

# ------------------------------------------------------------
# 3A. SUPERSTORE – CASTING & TAMBAH KOLON WAKTU
# ------------------------------------------------------------

print("=== PROSES SILVER: SUPERSTORE ===")

superstore_silver = (
    superstore_raw
    # Cast tanggal
    .withColumn("Order Date", F.to_date(F.col("Order Date"), "M/d/yyyy"))
    .withColumn("Ship Date", F.to_date(F.col("Ship Date"), "M/d/yyyy"))
    # Cast numerik
    .withColumn("Sales", F.col("Sales").cast(T.DoubleType()))
    .withColumn("Profit", F.col("Profit").cast(T.DoubleType()))
    .withColumn("Quantity", F.col("Quantity").cast(T.IntegerType()))
    .withColumn("Discount", F.col("Discount").cast(T.DoubleType()))
)

# Tambah kolom tahun/bulan dari Order Date (kalau ada)
if "Order Date" in superstore_silver.columns:
    superstore_silver = (
        superstore_silver
        .withColumn("order_year", F.year(F.col("Order Date")))
        .withColumn("order_month", F.month(F.col("Order Date")))
        .withColumn(
            "order_year_month",
            F.date_format(F.col("Order Date"), "yyyy-MM")
        )
    )

# Simpan Silver ke MinIO
superstore_silver.write.mode("overwrite").parquet(
    f"s3a://{BUCKET}/silver/superstore"
)

# Simpan Silver ke PostgreSQL sebagai tabel detail
superstore_silver.write.mode("overwrite").jdbc(
    url=PG_URL,
    table="superstore_clean",
    properties=PG_PROPERTIES
)

# ------------------------------------------------------------
# 3B. TWCS – CLEAN + SENTIMENT DASAR + WAKTU
# ------------------------------------------------------------

print("=== PROSES SILVER: TWCS ===")

cols_tweets = tweets_raw.columns
text_col = pick_col(cols_tweets, TWEET_TEXT_CANDIDATES, "text (isi tweet)")

# created_at opsional – kalau tidak ada, kita skip date parsing
created_col = None
try:
    created_col = pick_col(cols_tweets, TWEET_CREATED_AT_CANDIDATES, "created_at")
except ValueError:
    print("PERINGATAN: kolom created_at tidak ditemukan, skip parsing tanggal tweet.")

# company opsional
company_col = None
try:
    company_col = pick_col(cols_tweets, COMPANY_COL_CANDIDATES, "company/brand")
except ValueError:
    print("INFO: kolom company/brand tidak ada, skip agregasi per brand.")

tweets_base = tweets_raw.withColumn(
    "text_clean",
    F.lower(F.col(text_col).cast(T.StringType()))
)

if created_col is not None:
    # created_at di TWCS biasanya format: yyyy-MM-dd HH:mm:ss atau ISO
    tweets_base = (
        tweets_base
        .withColumn(
            "created_at_ts",
            F.to_timestamp(F.col(created_col))
        )
        .withColumn("tweet_date", F.to_date(F.col("created_at_ts")))
        .withColumn("tweet_year", F.year(F.col("tweet_date")))
        .withColumn("tweet_month", F.month(F.col("tweet_date")))
        .withColumn(
            "tweet_year_month",
            F.date_format(F.col("tweet_date"), "yyyy-MM")
        )
    )

# Kamus kecil untuk sentiment
POSITIVE_WORDS = [
    "good", "great", "awesome", "love", "excellent", "nice", "helpful",
    "fast", "thank you", "thanks", "happy", "satisfied"
]
NEGATIVE_WORDS = [
    "bad", "terrible", "awful", "hate", "worst", "slow", "angry",
    "disappointed", "broken", "issue", "problem", "not working", "late"
]

@F.udf(returnType=T.IntegerType())
def sentiment_score(text):
    if text is None:
        return 0
    t = text.lower()
    score = 0
    for w in POSITIVE_WORDS:
        if w in t:
            score += 1
    for w in NEGATIVE_WORDS:
        if w in t:
            score -= 1
    return score

@F.udf(returnType=T.StringType())
def sentiment_label(score):
    if score is None:
        return "neutral"
    if score > 0:
        return "positive"
    elif score < 0:
        return "negative"
    else:
        return "neutral"

tweets_silver = (
    tweets_base
    .withColumn("sentiment_score", sentiment_score(F.col("text_clean")))
    .withColumn("sentiment", sentiment_label(F.col("sentiment_score")))
)

# Simpan Silver ke MinIO
tweets_silver.write.mode("overwrite").parquet(
    f"s3a://{BUCKET}/silver/twcs_sentiment"
)

# Simpan Silver ke PostgreSQL
tweets_silver.write.mode("overwrite").jdbc(
    url=PG_URL,
    table="twcs_clean",
    properties=PG_PROPERTIES
)

# ============================================================
# 4. GOLD LAYER – AGGREGATE UNTUK DASHBOARD (TANPA LOGIKA BISNIS)
# ============================================================

print("=== GOLD: SUPERSTORE CATEGORY x MONTH ===")

# Agregat sales/profit by Category, Sub-Category, Month
group_cols = ["Category", "Sub-Category"]
if "order_year_month" in superstore_silver.columns:
    group_cols.append("order_year_month")

superstore_gold_category_month = (
    superstore_silver
    .groupBy(*[F.col(c) for c in group_cols])
    .agg(
        F.sum("Sales").alias("total_sales"),
        F.sum("Profit").alias("total_profit"),
        F.sum("Quantity").alias("total_quantity"),
        F.avg("Discount").alias("avg_discount")
    )
)

# Simpan ke MinIO
superstore_gold_category_month.write.mode("overwrite").parquet(
    f"s3a://{BUCKET}/gold/superstore_category_month"
)

# Simpan ke PostgreSQL
superstore_gold_category_month.write.mode("overwrite").jdbc(
    url=PG_URL,
    table="superstore_category_month",
    properties=PG_PROPERTIES
)

print("=== GOLD: TWCS SENTIMENT x MONTH (GLOBAL) ===")

if "tweet_year_month" in tweets_silver.columns:
    tweets_gold_sentiment_month = (
        tweets_silver
        .groupBy("tweet_year_month", "sentiment")
        .agg(F.count("*").alias("tweet_count"))
    )
else:
    # fallback: tanpa month, cuma sentiment total
    tweets_gold_sentiment_month = (
        tweets_silver
        .groupBy("sentiment")
        .agg(F.count("*").alias("tweet_count"))
    )

# Simpan ke MinIO
tweets_gold_sentiment_month.write.mode("overwrite").parquet(
    f"s3a://{BUCKET}/gold/twcs_sentiment_month"
)

# Simpan ke PostgreSQL
tweets_gold_sentiment_month.write.mode("overwrite").jdbc(
    url=PG_URL,
    table="twcs_sentiment_month",
    properties=PG_PROPERTIES
)

# ============================================================
# 5. (OPSIONAL) SIMPAN GOLD KE /warehouse DALAM CSV
#    -> Bisa kamu import manual ke Postgres kalau JDBC bermasalah
# ============================================================

print("=== SIMPAN GOLD KE FOLDER /warehouse (CSV, OPSIONAL) ===")

(
    superstore_gold_category_month
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", True)
    .csv("/warehouse/superstore_category_month_csv")
)

(
    tweets_gold_sentiment_month
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", True)
    .csv("/warehouse/twcs_sentiment_month_csv")
)

print("=== ETL SELESAI ===")
spark.stop()
