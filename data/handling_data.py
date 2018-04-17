import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import Imputer,StandardScaler
from sklearn.model_selection import train_test_split
from data.utils import FeatureEngineering,DataFrameSelector 
from spec_files.categorical_encoder import CategoricalEncoder


class HandlingData:  
    """
        This class handles the data. There are two methods: prepare_data and get_data.
        
    """
    def __init__(self,nom_du_fichier='ortho.csv',target="Discharge"):
        self.nom_du_fichier = nom_du_fichier
        self.target = target

    def prepare_data(self,data):
        """
            Prepares the data for Machine Learning algorithm. 
            First, we make the Feature Engineering, and then the selection of revelant features for solving 
            the machine learning problem. And then, the mean of the data, finally the feature scaling.
        """
        num_attrs = ['BMI','AGE_AT_ADMIT','Gender','Female','PreOpNarcotic','PreOpInsulin',
                              'PreOpDMMeds','PreOpBloodThinner','degre_dx','med_cond',]
        cat_attrs = ['RawDx', 'Side'] 
        
        #A numerical pipeline for transfoming the numerical features
        num_pipeline = Pipeline([
            ('feature_engineering', FeatureEngineering()),# FeatureEngineering allows the creation the new features. 
                                                          # It exists in the module utils 
            ('selector', DataFrameSelector(num_attrs)), # DataFrameSelector allows the selection of attributes. it exists in the module utils 
            ('imputer', Imputer(strategy="mean")),
            ('std_scaler', StandardScaler()),
        ])
        
        
        #Categorical pipeline for transforming textual features
        cat_pipeline = Pipeline([
            ('selector', DataFrameSelector(cat_attrs)),
            ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),# CategoricalEncoder exists 
        ])
        
        #Union both pipelines
        full_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline),
            ("cat_pipeline", cat_pipeline),
        ])
        return full_pipeline.fit_transform(data)
    
    def get_data(self):
        """
            Get the data prepared for ML
        """
        #Extraction the data in cvs format
        data = pd.read_csv("data/" + self.nom_du_fichier,sep=';',decimal=',')
        
        #Features predictors and labels
        X = data.drop(self.target, axis=1)
        y = data[self.target]
        
        #Get the data prepared to Machine Learning Algorithm
        self.X_prepared = self.prepare_data(X)
        self.y_data = y
        
        #Splitting data
        self.X_train_prepared, self.X_test_prepared, self.y_train, self.y_test = train_test_split(self.X_prepared,y,test_size=0.2, random_state=42)
    
    def main(self):
        self.get_data()