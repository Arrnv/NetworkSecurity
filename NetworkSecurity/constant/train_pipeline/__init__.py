import os
import sys
import numpy as np # type: ignore
import pandas as pd# type: ignore

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")


"""
Data ingestion related constant starts with DATA_INGESTION VAR NAME
"""

DATA_INGESTION_COLLECTION_NAME: str = "Phising_data"
DATA_INGESTION_DATABASE_NAME: str = "Network_security"
DATA_INGESTION_DIR_NAME: str = "data_ingested"
DATA_INGESTION_FEATURE_STORE_DIR: str = "featurea_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.25

"""
defining common const variable for training pipeline
"""
TRAGET_COLLUMN = "Result"
PIPELINE_NAME:str = "Network_security"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "phisingData.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME:str = "test.csv"

"""
data validation realated constant
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validation"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRAFT_REPORT_DIR: str = "draft_report"
DATA_VALIDATION_DRAFT_REPORT_FILE_NAME:str = "report.yaml"