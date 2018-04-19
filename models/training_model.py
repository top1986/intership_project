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




class BuildingModel:
    """
        
    """
    def __init__(self, X_train, X_test, y_train, y_test):
        
        self.X_train, self.X_test = X_train, X_test, 
        self.y_train, self.y_test =  y_train, y_test
        
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
         
    
    def train_and_eval_models(self):
        """
            Training, evaluating and displaying the models
        """
        
        results = ""
        
        for name,model in self.models_to_train(): 
        
            results += name + ": \n"
            model.fit(self.X_train, self.y_train)
            
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)
            
            results += "\t \t roc_auc_score: {} \n".format(roc_auc_score(self.y_test, y_prob[:,1]))
            results +="\t \t precision_score: {} \n".format(precision_score(self.y_test,y_pred))
            results +="\t \t recall_score : {} \n".format(recall_score(self.y_test,y_pred)) 
            results +="\t \t accuracy_score : {} \n".format(accuracy_score(self.y_test,y_pred))
            results +="\t \t score_model_on_train: {} \n".format(model.score(self.X_train, self.y_train))
            results +="\t \t score_model_on_testset : {} \n".format(model.score(self.X_test,self.y_test))
            results +="\t \t classification_report : {} \n".format(classification_report(self.y_test,y_pred))
            
        print (results)
               
   
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

        best_ind = scores.index(max(scores))
        
        return models[best_ind], scores[best_ind]
        
        
    
    def plot_precision_recall_vs_threshold(self,precisions,recalls,thresholds): 
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision") 
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall") 
        plt.xlabel("Threshold")
        plt.legend(loc="center left")
        plt.ylim([0, 1])
    
    def main(self):
        self.train_model()
        