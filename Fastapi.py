import findspark
findspark.init()

from pyspark import SparkContext, SparkConf, SQLContext

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import os

CLICKHOUSE_IP = "34.32.60.106"
CLICKHOUSE_PORT = 9000
CLICKHOUSE_USER = "default"
CLICKHOUSE_USER_PASSWORD = "1278"

#https://repo1.maven.org/maven2/com/github/housepower/clickhouse-native-jdbc/2.7.1/clickhouse-native-jdbc-2.7.1.jar
packages = [
    "com.github.housepower:clickhouse-spark-runtime-3.4_2.12:0.7.3",
    "com.clickhouse:clickhouse-jdbc:0.6.0-patch4",
    "com.clickhouse:clickhouse-http-client:0.6.0-patch4",
    "org.apache.httpcomponents.client5:httpclient5:5.3.1",
    "com.github.housepower:clickhouse-native-jdbc:2.7.1"
]
ram = 30
cpu = 22*3
# Define the application name and setup session
appName = "Connect To ClickHouse via PySpark"
spark = (SparkSession.builder
         .appName(appName)
         .config("spark.jars.packages", ",".join(packages))
         .config("spark.sql.catalog.clickhouse", "xenon.clickhouse.ClickHouseCatalog")
         .config("spark.sql.catalog.clickhouse.host", CLICKHOUSE_IP)
         .config("spark.sql.catalog.clickhouse.protocol", "http")
         .config("spark.sql.catalog.clickhouse.http_port", "8123")
         .config("spark.sql.catalog.clickhouse.user", CLICKHOUSE_USER)
         .config("spark.sql.catalog.clickhouse.password", CLICKHOUSE_USER_PASSWORD)
         .config("spark.sql.catalog.clickhouse.database", "default")
         #.config("spark.spark.clickhouse.write.compression.codec", "lz4")
         #.config("spark.clickhouse.read.compression.codec", "lz4")
         #.config("spark.clickhouse.write.format", "arrow")
         #    .config("spark.clickhouse.write.distributed.convertLocal", "true")
         #    .config("spark.clickhouse.write.repartitionNum", "1")
         #.config("spark.clickhouse.write.maxRetry", "1000")
         #    .config("spark.clickhouse.write.repartitionStrictly", "true")
         #    .config("spark.clickhouse.write.distributed.useClusterNodes", "false")
         #.config("spark.clickhouse.write.batchSize", "1000000")
         #.config("spark.sql.catalog.clickhouse.socket_timeout", "600000000")
         #  .config("spark.sql.catalog.clickhouse.connection_timeout", "600000000")
         #  .config("spark.sql.catalog.clickhouse.query_timeout", "600000000")
         #  .config("spark.clickhouse.options.socket_timeout", "600000000")
         #  .config("spark.clickhouse.options.connection_timeout", "600000000")
         #  .config("spark.clickhouse.options.query_timeout", "600000000")
         .config("spark.executor.memory", f"{ram}g")
         #.config("spark.executor.cores", "5")
         .config("spark.driver.maxResultSize", f"{ram}g")
         #.config("spark.driver.memory", f"{ram}g")
         #.config("spark.executor.memoryOverhead", f"{ram}g")
         #.config("spark.sql.debug.maxToStringFields", "100000")
         .getOrCreate()
         )
#SedonaRegistrator.registerAll(spark)
# spark.conf.set("spark.sql.catalog.clickhouse", "xenon.clickhouse.ClickHouseCatalog")
# spark.conf.set("spark.sql.catalog.clickhouse.host", "127.0.0.1")
# spark.conf.set("spark.sql.catalog.clickhouse.protocol", "http")
# spark.conf.set("spark.sql.catalog.clickhouse.http_port", "8123")
# spark.conf.set("spark.sql.catalog.clickhouse.user", "default")
# spark.conf.set("spark.sql.catalog.clickhouse.password", "")
# spark.conf.set("spark.sql.catalog.clickhouse.database", "default")
spark.sql("use clickhouse")

from ydata_profiling import ProfileReport
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import mlflow
import optuna


root_path = "/app"
#root_path = "."
#path_data = f'{root_path}/data'
your_mlflow_tracking_uri = f'{root_path}/mlruns'

mlflow.set_tracking_uri(your_mlflow_tracking_uri)

import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, UploadFile, File
import io
from pyspark.ml.pipeline import Pipeline

from optuna.integration.mlflow import MLflowCallback



app = FastAPI()

loaded_model = Pipeline.load("bestmodel")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content=await file.read()
    df = pd.read_csv(io.StringIO(content.decode('utf-8')))
    df = spark.createDataFrame(df)
    print(df.columns)
    predict_model=loaded_model.predict(df)
    print(type(predict_model))
    return {"predict": predict_model.tolist()}
