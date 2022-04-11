# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 04:23:16 2020

@author: lxy
"""

import numpy as np
import pandas as pd
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
        y50list=y50list+[0]
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
        y25list=y25list+[0]
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

class MyLogisticReg2:
    def __init__(self,d):
        self.d=d
        self.w=np.ones(1+self.d)
        self.delta=1e-11
    
    def sigmoid(self,x):
        z = 1 / (1 + np.exp(-x))
        return z
    
    def fit(self,X,y):
        """
        训练模型
        ----------
       X : 自变量数据，形状为（m,n）,m:数据的个数，n:自变量（或者特征）的个数
       y : 因变量数据，形状为（m,1）
       learning_rate : 训练速度，自己随意设置，可以为0.1，0.01，0.001，等任何数字
       iters : 迭代次数，自己随意设置，可以为100，1000，10000，等任何数字
        ----------
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        t_max=1000
        size=0.00001
        self.W = np.zeros((n_features,1))
        self.b = 0
            
        cost_list = []
        for i in range(t_max):
            a = self.sigmoid(np.dot(X,self.W)+self.b)
            cost = -1/n_samples * np.sum(y*np.log(a) + (1-y)*np.log(1-a))
            cost_list.append(cost)
            if i % 100 == 0:
                print(f"Cost at iteration {i}:{cost}")
            
            dW = np.dot(X.T,(a-y))/n_samples
            db = np.sum(a-y)/n_samples
            
            self.W = self.W - size*dW
            self.b = self.b - size*db
            
        return self.W,self.b,cost_list
    
            
    def grad(self, x, y):     
        grad = np.ones(1 + self.d) 
        e = np.dot(self.w[1:self.d + 1], x) + self.w[0]
        grad *= self.sigmoid(e) - y
        for i in range(1 + self.d):
            if i == 0:
                continue
            grad[i] *= x[i-1]
        return grad
    
    def predict(self,X):
        prelist=np.empty([X.shape[0],1])
        a_pred=np.transpose(np.ones(X.shape[0]))
        X_new=np.column_stack((a_pred,X))
        for k in range(X.shape[0]):
            z=np.matmul(X_new[k],np.transpose(self.w))
            p=self.sigmoid(z)
            if p>0.5:
                prelist[k,0]=1
            else:
                prelist[k,0]=0
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
    
def my_cross_val(method,X,y,k):    ##main part
    if method==LinearSVC:
        model=LinearSVC(max_iter=2000)
        return my_accuracy(model,X,y,k)
    elif method==SVC:
        model=SVC(gamma='scale', C=10)  
        return my_accuracy(model,X,y,k)
    elif method==LogisticRegression:
        model=LogisticRegression( penalty='l2', solver='lbfgs', multi_class='multinomial',max_iter=5000)
        return my_accuracy(model,X,y,k)
    elif method=="MyLogisticReg2":
        d=X.shape[1]
        model=MyLogisticReg2(d)
        return my_accuracy(model,X,y,k)
    else:
        return "Errormessage"
    
def get_params(self, deep=True):
	return {"k": self.k, "d": self.dim, "diag": self.diagonal}

def q3():
    mod1=my_cross_val("MyLogisticReg2",boston.data,y50list,5)
    print("Error rates for MyLogisticReg2 with Boston 50:")
    n=1
    for i in mod1[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod1[1])
    print("Standard Deviation: ",mod1[2])
    mod2=my_cross_val("MyLogisticReg2",boston.data,y25list,5)
    print("Error rates for MyLogisticReg2 with Boston 25:")
    n=1
    for i in mod2[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod2[1])
    print("Standard Deviation: ",mod2[2])
    mod3=my_cross_val(LogisticRegression,boston.data,y50list,5)
    print("Error rates for LogisticRegression with Boston50:")
    n=1
    for i in mod3[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod3[1])
    print("Standard Deviation: ",mod3[2])
    mod4=my_cross_val(LogisticRegression,boston.data,y25list,5)
    print("Error rates for LogisticRegression with Boston25:")
    n=1
    for i in mod4[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod4[1])
    print("Standard Deviation: ",mod4[2])

    
q3()