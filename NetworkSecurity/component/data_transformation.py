import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from NetworkSecurity.constant.train_pipeline import TRAGET_COLLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from NetworkSecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)
from NetworkSecurity.entity.config_entity import DataTransformationConfig
from NetworkSecurity.exception.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging
from NetworkSecurity.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, data_validation_artifact:DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig
                 ):
        try:
            self.data_validation_artifact:DataValidationArtifact = data_validation_artifact
            self.data_transformation_config:DataTransformationConfig = data_transformation_config
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    @staticmethod
    def read_data(file_path)-> pd.DataFrame:
        try:
           return pd.read_csv(file_path)
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    
    def get_data_transformer_object(cls)->Pipeline:
            """
            It initialises a KNNImputer object with the parameters specified in the training_pipeline.py file
            and returns a Pipeline object with the KNNImputer object as the first step.

            Args:
            cls: DataTransformation

            Returns:
            A Pipeline object
            """
            logging.info(
                "Entered get_data_trnasformer_object method of Trnasformation class"
            )
            try:
                imputer:KNNImputer=KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
                logging.info(
                        f"Initialise KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}"
                    )
                processor:Pipeline=Pipeline([("imputer",imputer)])
                
                return processor
            except Exception as e:
                raise NetworkSecurityException(e,sys)
            
    
    def InitateDataTransformation(self)-> DataTransformationArtifact:
        logging.info("Entered data transformation")
        try:
            logging.info(f"starting data transformation")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_training_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_testing_file_path)
            

            input_feature_train_df = train_df.drop(columns=[TRAGET_COLLUMN], axis=1)
            target_feature_train_df = train_df[TRAGET_COLLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1,0)
            
            input_feature_test_df = test_df.drop(columns=[TRAGET_COLLUMN], axis=1)
            target_feature_test_df = test_df[TRAGET_COLLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1,0)
            preprocess = self.get_data_transformer_object()
            preprocess_obj = preprocess.fit(input_feature_train_df)
            transformed_input_train_feature = preprocess_obj.transform(input_feature_train_df)
            transformed_input_test_feature = preprocess_obj.transform(input_feature_test_df)

            train_arr = np.c_[transformed_input_train_feature, np.array(input_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(input_feature_test_df)]
            
            #save numpy array data
            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr,)
            save_object( self.data_transformation_config.transformed_object_file_path, preprocess_obj,)

            save_object( "final_model/preprocessor.pkl", preprocess_obj,)


            #preparing artifacts

            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)