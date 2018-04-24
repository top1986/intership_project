import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier,
                              GradientBoostingClassifier,AdaBoostClassifier)
from mlxtend.classifier import StackingClassifier

from sklearn.metrics import (f1_score,precision_score,recall_score, classification_report,
                             precision_recall_curve,roc_auc_score,accuracy_score)

from sklearn.externals import joblib
import pickle

PATH_SAVE_MODEL = "spec_files/models/"
PATH_SAVE_PREDCITIONS = "spec_files/models"

class BuildingModel:
    """
        Manipulates everything related to the construction of a model. Training model, evaluation's model
        and selection and saving of the best model.

    """
    def __init__(self, X_data, X_train, X_test, y_data, y_train, y_test):

        self.X_data, self.X_train, self.X_test = X_data, X_train, X_test,
        self.y_data, self.y_train, self.y_test =  y_data, y_train, y_test

    def models_to_train(self):
        """
            Appends the differents type of models to train and returns a list of the tuples with name and an
            instance of each model
        """
        models=[]
        models.append(('GradientBoostingClassifier',
                       GradientBoostingClassifier(n_estimators=500, learning_rate=1,
                                                  max_depth=1, random_state=0)))#gradient boosting with differents
                                                                                #hyperparameters for fine-tune the
                                                                                #model
        models.append(('AdaBoostClassifier',
                       AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=500,
                                          algorithm="SAMME.R", learning_rate=1)))#Applying an adaptive boosting
                                                                                   #classifier on an tree classifier
                                                                                   #wich badly ovefits the data


        models.append(('StackingClassifier',
                      StackingClassifier(classifiers = [LogisticRegression(),
                                                       KNeighborsClassifier(),
                                                       RandomForestClassifier(),],
                                        meta_classifier = LogisticRegression(), use_probas=True)
                      ))

        return models


    def evaluate_best_model(self, model):
        """
            evaluating differents scores, precision, recall and classification report of the best model.
            Returns the results in a dcitionnary
        """

        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)

        results = {"name": model.__class__.__name__,
                    "roc_auc_score": roc_auc_score(self.y_test, y_prob[:,1]),
                    "precision_score": precision_score(self.y_test,y_pred),
                    "recall_score": recall_score(self.y_test,y_pred),
                    "accuracy_score": accuracy_score(self.y_test,y_pred),
                    "classification_report": classification_report(self.y_test,y_pred),
                }

        return results


    def select_best_model(self):
        """
            returns the best model basing on accuracy score.
        """
        scores = []
        models = []

        for name,model in self.models_to_train():

            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            score = accuracy_score(self.y_test, y_pred)

            models.append(model)
            scores.append(score)

        ind = scores.index(max(scores))

        return models[ind]

    def save_evaluation(self, model):
        with open(PATH_SAVE_MODEL + 'evaluation', 'wb') as evaluation:
            mon_pickler = pickle.Pickler(evaluation)
            mon_pickler.dump(self.evaluate_best_model(model))

    def save_model(self,model):

        model.fit(self.X_data, self.y_data)

        joblib.dump(model, PATH_SAVE_MODEL + "best_model.sav")



    def save(self):
        model = self.select_best_model()
        self.save_evaluation(model)
        self.save_model(model)


    def main(self):
        self.save()
