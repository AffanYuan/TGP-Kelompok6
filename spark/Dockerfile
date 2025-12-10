FROM docker.io/apache/spark:3.5.0

# Tambah library untuk S3A (MinIO)
RUN wget -P /opt/spark/jars \
  https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar && \
  wget -P /opt/spark/jars \
  https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.772/aws-java-sdk-bundle-1.12.772.jar && \
  # Tambah driver JDBC PostgreSQL
  wget -P /opt/spark/jars \
  https://repo1.maven.org/maven2/org/postgresql/postgresql/42.7.3/postgresql-42.7.3.jar
