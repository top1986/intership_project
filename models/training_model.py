import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer,StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score,confusion_matrix,classification_report,
                             f1_score,precision_score,recall_score,precision_recall_curve,
                             roc_auc_score,)

class TrainingModel:
    def __init__(self, X_data, X_train, X_test, y_data ,y_train, y_test):
        
        self.X_data, self.X_train, self.X_test = X_data, X_train, X_test, 
        self.y_data, self.y_train, self.y_test = y_data,  y_train, y_test
        
        kfold = KFold(n_splits=5, random_state=42)
        self.cv = kfold
        
    def models_to_train(self):
        models=[]
        models.append(('SGDClassifier', SGDClassifier(loss="modified_huber")))
        models.append(('RandomForestClassifier', RandomForestClassifier()))
        models.append(('LogisticRegression', LogisticRegression()))
        models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
        models.append(('KNeighborsClassifier', KNeighborsClassifier(n_neighbors=5)))
        models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
        models.append(('GaussianNB', GaussianNB()))
        models.append(('SVC', SVC(kernel='linear', probability=True, class_weight='balanced')))
        
        return models
        
    def non_probabilistic_evaluating_scores(self):
        """
            This function is a simple evaluation of each model.
            Also Evaluation is done with and without cross validation.
            
        """
        results = {}
        
        for (name,model) in self.models_to_train():
            model.fit(self.X_train,self.y_train)
            y_pred = model.predict(self.X_test)
            
            y_pred_cv = cross_val_predict(model,self.X_data, self.y_data,cv=self.cv)
            
            scores = cross_val_score(model, self.X_data, self.y_data, cv=self.cv, scoring="accuracy").mean()
            
            results[name] = [accuracy_score(self.y_test, y_pred),
                            accuracy_score(self.y_data,y_pred_cv),
                            scores,]

        df = pd.DataFrame(results, index=['accuracy_score_without_cv',
                                          'accuracy_score_cv_predict',
                                          'cross_val_score_with_accuracy'])
        return df    
    
    def probabilistic_evaluating_scores(self):
        """
            function that makes a probabilistic evaluation of each model.
            Evaluation is done with and without cross validation
        """
        results = {}
        
        for (name,model) in self.models_to_train():
            model.fit(self.X_train,self.y_train)
            y_pred_prob = model.predict_proba(self.X_test)
            
            y_pred_prob_cv = cross_val_predict(model,self.X_data, self.y_data, cv=self.cv,method="predict_proba")
            
            scores_prob = cross_val_score(model, self.X_data, self.y_data, cv=self.cv, scoring="roc_auc").mean()
                
            results[name] = [roc_auc_score(self.y_test, y_pred_prob[:,1]),
                             roc_auc_score(self.y_data, y_pred_prob_cv[:,1]),
                            scores_prob,]
            
        return pd.DataFrame(results, index=['roc_auc_score_without_cv',
                                            'roc_auc_score_cv_predict',
                                            'cross_val_score_with_roc_auc'])
    
    def perfomance_measures(self,):
        results = {}
        
        for (name,model) in self.models_to_train():
            
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            
            y_pred_cv = cross_val_predict(model,self.X_data, self.y_data, cv=self.cv,)
                
            results[name] = [confusion_matrix(self.y_test, y_pred),
                             precision_score(self.y_test,y_pred),
                             recall_score(self.y_test,y_pred),
                             f1_score(self.y_test,y_pred),
                             confusion_matrix(self.y_data, y_pred_cv),
                             precision_score(self.y_data,y_pred_cv),
                             recall_score(self.y_data,y_pred_cv),
                             f1_score(self.y_data,y_pred_cv),]
            
        df = pd.DataFrame(results, index=['confusion_matrix', 'precision_score', 
                                          'recall_score', 'f1_score',
                                          'confusion_matrix_cv', 'precision_score_cv', 
                                          'recall_score_cv', 'f1_score_cv',])
        
        return df
         
    
    def train_model(self, X, y):
        self.return_home = (y==1)
        self.do_not_return_home = (y==0)
               
   
    def select_best_model(self,X,y):
        pass
    
    def plot_precision_recall_vs_threshold(self,precisions,recalls,thresholds): 
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision") 
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall") 
        plt.xlabel("Threshold")
        plt.legend(loc="center left")
        plt.ylim([0, 1])
    
    def main(self):
        self.train_model()
        