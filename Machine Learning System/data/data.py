import pandas as pd

from sklearn.model_selection import train_test_split
from utils.preprocessing import prepare_data
from imblearn.over_sampling import SMOTE


class HandlingData:
    """
        This class handles the data. There are two methods: prepare_data and get_data.

    """
    def __init__(self,nom_du_fichier='ortho.csv',target="Discharge"):
        self.nom_du_fichier = nom_du_fichier
        self.target = target

    def get_data(self):
        """
            Get the data prepared for ML
        """
        #Extraction the data in cvs format
        data = pd.read_csv('Data/ortho.csv',sep=';',decimal=',')

        #Features predictors and labels
        X = data.drop(self.target, axis=1)
        y = data[self.target]

        #Get the data prepared to Machine Learning Algorithm
        X_prepared = prepare_data(X)

        #Splitting data
        X_train_prepared, self.X_test_prepared, y_train, self.y_test = train_test_split(X_prepared,
                                                                                        y,test_size=0.2,
                                                                                        random_state=42)
        sm = SMOTE(random_state=42)
        self.X_train_resampled,self.y_train_resampled = sm.fit_sample(X_train_prepared,y_train)
        self.X_resampled, self.y_resampled = sm.fit_sample(X_prepared, y)

    def main(self):
        self.get_data()
