# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 12:18:04 2017

@author: Chetan
"""

import pandas as pd 
import numpy  as np

#Importing Data
claimants = pd.read_csv("C:\\Users\\vinay goud\Documents\\DATA SCIENCE\\Codes With Examples - Python-20191016\Logistic Regression\\claimants.csv",sep=",")

claimants.columns

#removing CASENUM 

del claimants['CASENUM']

claimants

claimants.head(10)# to see top 10 observations

#Imputating the missing values with most repeated values in that column              
#claimants = claimants.apply(lambda x:x.fillna(x.value_counts().index[0]))

#Model building 
import numpy as np
import statsmodels.formula.api as sm
logit_model = sm.logit('ATTORNEY~CLMAGE+LOSS+CLMINSUR+CLMSEX+SEATBELT',data = claimants).fit()

#summary
logit_model.summary()

logit_model.params


#Odds Ratio

(np.exp(logit_model.params))

from sklearn.metrics import confusion_matrix
import scipy
from sklearn import linear_model

predict=logit_model.predict(pd.DataFrame(claimants[['CLMAGE','LOSS','CLMINSUR','CLMSEX','SEATBELT']]))

predict1=logit_model.predict(pd.DataFrame(claimants[['CLMAGE','LOSS','CLMINSUR','CLMSEX','SEATBELT']]))

pcnf_matrix = confusion_matrix(claimants['ATTORNEY'],predict > 0.5 )
pcnf_matrix
from sklearn.metrics import accuracy_score 
Accuracy_Score = accuracy_score(claimants['ATTORNEY'],predict > 0.5)

Accuracy_Score

#Accuracy = sum(diag(pcnf_matrix))
##
#cnf_matrixrint(np.exp(logit_model.params))

# Model Accuracy 
Accuracy<-sum(diag(confusion)/sum(confusion))
Accuracy



#from sklearn.metrics import confusion_matrix

 #from sklearn.metrics import classification_report
#print(classification_report(logit_model,predict))

accuracy = (435+506)/(435+250+149+506)
accuracy

