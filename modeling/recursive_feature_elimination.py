import pandas as pd
import numpy as np
import seaborn as sns
import os
import bson
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import shap
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


class rfe():
    def __init__(self,X_train,y_train,X_test,y_test,categorical_features):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.categorical_features = categorical_features


         
    def run_rfe(self):
        feature_list = self.X_train.columns.to_numpy()
        self.num_features_list = []
        self.actual_features_list = []
        self.train_auc_score_list = []
        self.test_auc_score_list = []
        params = {
            'objective': 'binary',
            'metric': 'binary_error',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }

        num_round = 100
        step_size = 1
        while len(feature_list)>2:
            if len(feature_list)>=200:
                step_size=30
            elif (len(feature_list)>=100) & (len(feature_list)<200):
                step_size=15
            elif (len(feature_list)>=50) & (len(feature_list)<100):
                step_size=10
            elif (len(feature_list)>=20) & (len(feature_list)<50):
                    step_size=3
            elif (len(feature_list)>=1) & (len(feature_list)<20):
                    step_size=1
                    

            self.categorical_features = list(set(self.categorical_features).intersection(set(feature_list)))
            train_data = lgb.Dataset(self.X_train[feature_list], label=self.y_train, categorical_feature=self.categorical_features)
            test_data = lgb.Dataset(self.X_test[feature_list], label=self.y_test, categorical_feature=self.categorical_features)
            bst = lgb.train(params, train_data, num_round, valid_sets=[test_data])
            y_train_pred = bst.predict(self.X_train[feature_list], num_iteration=bst.best_iteration)
            y_test_pred = bst.predict(self.X_test[feature_list], num_iteration=bst.best_iteration)
            self.num_features_list.append(str(len(feature_list)))
            self.actual_features_list.append(feature_list)    
            self.train_auc_score_list.append(roc_auc_score(self.y_train, y_train_pred))
            self.test_auc_score_list.append(roc_auc_score(self.y_test, y_test_pred))
            ini_len = len(feature_list)
            feature_list = feature_list[bst.feature_importance(importance_type='gain').argsort()[::-1]][:-step_size]
        fig,ax = plt.subplots(1,1,figsize=(16,8))
        ax.scatter(self.num_features_list,self.train_auc_score_list,label="train")
        ax.scatter(self.num_features_list,self.test_auc_score_list,label="test")
        plt.title("AUC Score Train and Test on using n features")
        plt.xlabel("Number of features used")
        plt.ylabel("AUC Score")
        plt.legend()
        df = pd.DataFrame({"num_features_list":self.num_features_list,"train_auc_score_list":self.train_auc_score_list,"test_auc_score_list":self.test_auc_score_list})
        return df,fig 

    def get_top_k_feature_list(self,wanted_features):
        num_features_list = np.array(self.num_features_list) 
        return self.actual_features_list[np.argwhere(num_features_list==str(wanted_features))[0][0]]