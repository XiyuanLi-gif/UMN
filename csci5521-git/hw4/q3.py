# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 02:02:00 2020

@author: lxy
"""

import numpy as np
import pandas as pd
import math
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
## prepare for the data
from sklearn import datasets
boston=datasets.load_boston()
response = boston.target 
## boston50
t50=np.median(response)
y50list=[]
for i in response:
    if i < t50:
        y50list=y50list+[-1]
    else:
        y50list=y50list+[1] 


#print(r50)
#y50=pd.DataFrame({'y':y50list})
#r=pd.DataFrame({'r':response})
#r50=pd.concat([r,y50],axis=1)
        
#boston25
t25=np.percentile(response,25)
y25list=[]
for i in response:
    if i < t25:
        y25list=y25list+[-1]
    else:
        y25list=y25list+[1]    
#y25=pd.DataFrame({'y':y25list})
#r=pd.DataFrame({'r':response})
#r25=pd.concat([r,y25],axis=1)

#n=0
#for i in y25list:
#    if i == 0:
#        n+=1
        
#print(r25)
#print(n)
digits=datasets.load_digits()
#a=np.concatenate((digits.data[0:0,:],digits.data[0:10],axix=0)
#print(a])
#now we have datasets r50,r25,digits


class MySVM2:
    def __init__(self,d,m):
        self.d=d
        self.m=m
        self.w0=0.0
        self.w=np.ones(self.d)
        self.l=0.0001
    
    def sigmoid(self,x):
        if x>=0:
            m=np.exp(-x)
            return 1/(1+m)
        else:
            m=np.exp(x)
            return m/(1+m)
    
    def fit(self,X,y):
        t_max=500   ##maximum iterater times
        n=X.shape[0]
        batch_num=math.floor(n/self.m)
        i = np.arange(n)
        np.random.shuffle(i)
        for j in range(0,t_max):
            for k in range(0,batch_num):
                r = i[k*self.m:(k+1)*self.m]
                X_array = np.take(X, r, axis=0)
                Y_array = np.take(y, r)
                grad0,gradw = self.SGD(X_array, Y_array)
                self.w = self.w - self.l * gradw
                self.w0 = self.w0 - self.l * grad0
                
    def SGD(self, x, y):     #helper function of fit
        grad0=0.0
        gradw = np.zeros(self.d) 
        for i in range(0,self.m):
            x_t=x[i]
            y_t=y[i]
            e = np.dot(self.w, x_t) + self.w0
            if 1.0-e*y_t > 0.0:
                grad0 += -y_t       #g1
                gradw += -y_t*x_t   #f1
            else:
                grad0 += 0.0        #g2
                gradw += 0.0        #f2
        return grad0, gradw

    def predict(self,X):
        prelist=np.empty([X.shape[0],1])
        for k in range(X.shape[0]):
            score=self.w0 + np.dot(self.w, X[k])
            if score>0:
                prelist[k,0]=1
            else:
                prelist[k,0]=-1
        return prelist


def my_accuracy(model,X,y,k):   ##helper function of my_cross_val
    row=len(X)
    size_fold=int(row/k)
    scorelist=[]
    for i in range(0,k):
        start=int(i*size_fold)
        end=int(size_fold+start)
        X_test=X[start:end]
        y_test=y[start:end]
        if i==0:
            X_train=X[end+1:-1]
            y_train=y[end+1:-1]
        elif i==k-1:
            X_train=X[0:end-1]
            y_train=y[0:end-1]
        else:
            X_train=np.concatenate((X[0:start-1],X[end+1:-1]),axis=0)
            y_train=np.concatenate((y[0:start-1],y[end+1:-1]),axis=0)
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        scorelist+=[comp(y_pred,y_test)]
    scores=np.array(scorelist)
    error_rates=1-scores
    mean=np.mean(error_rates)
    std=np.std(error_rates)
    return error_rates,mean,std

def comp(y1,y2):           ##helper function of my_cross_val
    n=0
    for i in range(0,len(y1)):
        if(y1[i]==y2[i]):
            n+=1
    rt=n/float(len(y1))
    return rt
    
    
def get_params(self, deep=True):
	return {"d": self.d, "m": self.m}


def q3():
    m=40
    mod1=MySVM2(boston.data.shape[1],m)
    score1=my_accuracy(mod1,boston.data,y50list,5)
    print("Error rates for MySVM2 with m = 40 for Boston 50:")
    n=1
    for i in score1[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",score1[1])
    print("Standard Deviation: ",score1[2])
    m=200
    mod2=MySVM2(boston.data.shape[1],m)
    score2=my_accuracy(mod2,boston.data,y50list,5)
    print("Error rates for MySVM2 with m = 200 for Boston 50:")
    n=1
    for i in score2[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",score2[1])
    print("Standard Deviation: ",score2[2])
    m=10e6
    mod3=MySVM2(boston.data.shape[1],m)
    score3=my_accuracy(mod3,boston.data,y50list,5)
    print("Error rates for MySVM2 with m = n for Boston 50:")
    n=1
    for i in score3[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",score3[1])
    print("Standard Deviation: ",score3[2])
    mod4=LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial',max_iter=5000)
    mod4.fit(boston.data, y50list)
    score4 = cross_val_score(mod4, boston.data, y50list, cv=5, scoring = "accuracy")
    print("Error rates for LogisticRegression with Boston50:")
    n=1
    for i in score4:
       print("Fold ",n,":",1-i)
       n+=1
    print("Mean: ",np.mean(1-score4))
    print("Standard Deviation: ",np.std(1-score4))
    m=40
    mod5=MySVM2(boston.data.shape[1],m)
    score5=my_accuracy(mod5,boston.data,y25list,5)
    print("Error rates for MySVM2 with m = 40 for Boston 25:")
    n=1
    for i in score5[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",score5[1])
    print("Standard Deviation: ",score5[2])
    m=200
    mod6=MySVM2(boston.data.shape[1],m)
    score6=my_accuracy(mod6,boston.data,y25list,5)
    print("Error rates for MySVM2 with m = 200 for Boston 25:")
    n=1
    for i in score6[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",score6[1])
    print("Standard Deviation: ",score6[2])
    m=10e6
    mod7=MySVM2(boston.data.shape[1],m)
    score7=my_accuracy(mod7,boston.data,y25list,5)
    print("Error rates for MySVM2 with m = n for Boston 25:")
    n=1
    for i in score7[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",score7[1])
    print("Standard Deviation: ",score7[2])
    mod8=LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial',max_iter=5000)
    mod8.fit(boston.data, y25list)
    score8 = cross_val_score(mod8, boston.data, y25list, cv=5, scoring = "accuracy")
    print("Error rates for LogisticRegression with Boston25:")
    n=1
    for i in score8:
       print("Fold ",n,":",1-i)
       n+=1
    print("Mean: ",np.mean(1-score8))
    print("Standard Deviation: ",np.std(1-score8))
 
q3()