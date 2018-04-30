from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import Imputer,StandardScaler
from utils.categorical_encoder import CategoricalEncoder


def prepare_data(data):
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
            ('feature_engineering', PreProcessing()),
            ('selector', DataFrameSelector(num_attrs)),
            ('imputer', Imputer(strategy="mean")),
            ('std_scaler', StandardScaler()),
        ])


        #Categorical pipeline for transforming textual features
        cat_pipeline = Pipeline([
            ('selector', DataFrameSelector(cat_attrs)),
            ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
        ])

        #Union both pipelines
        full_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline),
            ("cat_pipeline", cat_pipeline),
        ])
        return full_pipeline.fit_transform(data)



class PreProcessing(BaseEstimator, TransformerMixin):
    """
        This a class transformer for making new features or feature engineering
        The created features are: "degre_dx" and "med_cond"
        degre_dx --> determines the degree of severity of the patient's disease taking into account the diagnoses made beforehand
        med_cond --> it allows to determine the medical conditions of the patient while taking into account the diseases suffered in the past
    """
    def __init_(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = self.create_degree_dx(X)
        X = self.create_medical_conditions(X)

        return X



    def create_degree_dx(self,X):
        X['degre_dx'] = 1

        X.loc[X.Dx1.isnull(), 'degre_dx'] = 0

        X.loc[(X.Dx1.isnull() == False) &
              (X.Dx2.isnull() == False), 'degre_dx'] = 2

        X.loc[(X.Dx1.isnull() == False) &
              (X.Dx2.isnull() == False) &
              (X.Dx3.isnull() == False), 'degre_dx'] = 3

        return X

    def create_medical_conditions(self, X):
        X_med_cond = X.loc[:,'PreOpHgb':'Depression']

        cols_2 = ['PreOpHgb', 'PreOpGlucose', 'pulm circ', 'other neuro', 'chronic pulm',]
        cols_3 = ['PreOpCr','Paralysis','renal failure','liver failure',]


        for col in X_med_cond.columns:
            val = 0
            X_med_cond[col].loc[X_med_cond[col].isnull()] = val

            if col in cols_2:
                val = 2
            elif col in cols_3:
                val = 3
            else:
                val = 1

            X_med_cond[col].loc[X_med_cond[col] != 0] = val


        X['med_cond'] = X_med_cond.sum(axis=1)

        return X




class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
        Selection of acceptables attributes for the machine learning
    """
    def __init__(self, attr_names):
        self.attr_names = attr_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.attr_names]


#class
