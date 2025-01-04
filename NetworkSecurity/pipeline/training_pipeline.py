import os
import sys

from NetworkSecurity.exception.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging
from NetworkSecurity.constant.train_pipeline import TRAINING_BUCKET_NAME
from NetworkSecurity.cloud.s3_syncer import S3Sync


from NetworkSecurity.component.Data_ingestion import DataIngestion
from NetworkSecurity.component.data_validation import DataValidation
from NetworkSecurity.component.data_transformation import DataTransformation
from NetworkSecurity.component.model_trainer import ModelTrainer
from NetworkSecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)
from NetworkSecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from NetworkSecurity.constant.train_pipeline import SAVED_MODEL_DIR


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config=TrainingPipelineConfig()
        self.s3_sync = S3Sync()
        
    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start Data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.InitateDataIngestion()
            logging.info(f"Data ingestion completed: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e ,sys)
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            data_validation_config=DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Initiate the data Validation")
            data_validation=DataValidation(data_ingestion_artifact,data_validation_config)
            data_validation_artifact=data_validation.initiate_data_validation()
            logging.info("data Validation Completed")
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact):
        try:
            data_transformation_config=DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("data Transformation started")
            data_transformation=DataTransformation(data_validation_artifact,data_transformation_config)
            data_transformation_artifact=data_transformation.InitateDataTransformation()
            logging.info("data Transformation completed")
            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logging.info("Model traning started")
            model_trainer_config=ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,
                                       data_transformation_artifacts=data_transformation_artifact)
            model_trainer_artifact=model_trainer.initate_model_trainer()
            logging.info("model training complete")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def sync_artifacts_folder_to_s3_bucket(self):
        try:
            aws_bucket_url =f"s3://{TRAINING_BUCKET_NAME}/Artifacts/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.artifact_dir, aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def sync_saved_model_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_model/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder = self.training_pipeline_config.model_dir,aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            self.sync_artifacts_folder_to_s3_bucket()
            self.sync_saved_model_dir_to_s3()
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)