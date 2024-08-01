# create a fast api for downloading the file
import pandas as pd
from fastapi import FastAPI, UploadFile, File
import io
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import sklearn
sklearn.set_config(transform_output="pandas")
from sklearn.utils import compute_sample_weight
#
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import Ridge
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
import seaborn as sns


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import RobustScaler

from sklearn.cluster import KMeans

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC

import xgboost as xgb
from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from joblib import dump, load

from sklearn.impute import SimpleImputer

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import PowerTransformer

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from xgboost import plot_importance

from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt

from time import sleep

from sklearn.model_selection import cross_val_score

import optuna

from sklearn.metrics import fbeta_score, make_scorer

import mlflow

import json
from typing import List
from pydantic import BaseModel

from pyspark.ml import PipelineModel

#import findspark
#findspark.init()

from pyspark import SparkContext, SparkConf, SQLContext

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import os

CLICKHOUSE_IP = "34.32.47.131"
CLICKHOUSE_PORT = 9000
CLICKHOUSE_USER = "default"
CLICKHOUSE_USER_PASSWORD = "12341"

packages = [
    "com.github.housepower:clickhouse-spark-runtime-3.4_2.12:0.7.3",
    "com.clickhouse:clickhouse-jdbc:0.6.0-patch5",
    "com.clickhouse:clickhouse-http-client:0.6.0-patch5",
    "org.apache.httpcomponents.client5:httpclient5:5.3.1",
    "com.github.housepower:clickhouse-native-jdbc:2.7.1"
]
ram = 20
cpu = 22*3
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
         .config("spark.executor.memory", f"{ram}g")
         .config("spark.driver.maxResultSize", f"{ram}g")
         .getOrCreate()
         )
spark.sql("use clickhouse")


app = FastAPI()

loaded_model = PipelineModel.load("bestmodel")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content=await file.read()
    df = pd.read_csv(io.StringIO(content.decode('utf-8')))
    df = spark.createDataFrame(df)
    print(df.columns)
    predict_model=loaded_model.predict(df)
    print(type(predict_model))
    return {"predict": predict_model.tolist()}
