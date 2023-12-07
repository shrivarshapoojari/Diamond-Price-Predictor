from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import sys
import os
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformation_object(self):
        try:
            logging.info("DataTransformation Initiated")
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols=['cut','color','clarity']
            numerical_cols=['carat','depth','table','x','y','z']

             # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            logging.info("Data Transformation pipeline initiated")


            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
                ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            logging.info("Data Transformation Complete")

            return preprocessor
        except Exception as e:
            logging.info("Exception occured in data transformed")
            raise CustomException(e,sys)

    
    
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)
            logging.info("Train data read")
            logging.info("Obtaining preprocessor obj")
            preprocessing_obj =self.get_data_transformation_object()

            target_column='price'
            drop_column=[target_column,'id']


            input_feature_train_df=train_df.drop(columns=drop_column,axis=1)
            taget_feature_train_df=train_df[target_column]
            input_feature_test_df=test_df.drop(columns=drop_column,axis=1)
            taget_feature_test_df=test_df[target_column]

            ## Data Transformation
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            train_arr=np. c_[input_feature_train_arr,np.array(taget_feature_train_df)]
            test_arr= np.c_[input_feature_test_arr,np.array(taget_feature_test_df)]
            logging.info("Preprocessing data complete")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)
        


    
    

