# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 00:02:12 2020

@author: lxy
"""

import numpy as np
import pandas as pd
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
#hw1_problem3
#(i)
#(a)
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


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
    else:
        return "Errormessage"

def q3i():                  ##main part
    mod1=my_cross_val(LinearSVC,boston.data,y50list,10)
    print("Error rates for LinearSVC with Boston50:")
    n=1
    for i in mod1[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod1[1])
    print("Standard Deviation: ",mod1[2])
    mod2=my_cross_val(LinearSVC,boston.data,y25list,10)
    print("Error rates for LinearSVC with Boston25:")
    n=1
    for i in mod2[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod2[1])
    print("Standard Deviation: ",mod2[2])
    mod3=my_cross_val(LinearSVC,digits.data,digits.target,10)
    print("Error rates for LinearSVC with Digits:")
    n=1
    for i in mod3[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod3[1])
    print("Standard Deviation: ",mod3[2])
    mod4=my_cross_val(SVC,boston.data,y50list,10)
    print("Error rates for SVC with Boston50:")
    n=1
    for i in mod4[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod4[1])
    print("Standard Deviation: ",mod4[2])
    mod5=my_cross_val(SVC,boston.data,y25list,10)
    print("Error rates for SVC with Boston25:")
    n=1
    for i in mod5[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod5[1])
    print("Standard Deviation: ",mod5[2])
    mod6=my_cross_val(SVC,digits.data,digits.target,10)
    print("Error rates for SVC with Digits:")
    n=1
    for i in mod6[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod6[1])
    print("Standard Deviation: ",mod6[2])
    mod7=my_cross_val(LogisticRegression,boston.data,y50list,10)
    print("Error rates for LogisticRegression with Boston50:")
    n=1
    for i in mod7[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod7[1])
    print("Standard Deviation: ",mod7[2])
    mod8=my_cross_val(LogisticRegression,boston.data,y25list,10)
    print("Error rates for LogisticRegression with Boston25:")
    n=1
    for i in mod8[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod8[1])
    print("Standard Deviation: ",mod8[2])
    mod9=my_cross_val(LogisticRegression,digits.data,digits.target,10)
    print("Error rates for LogisticRegression with Digits:")
    n=1
    for i in mod9[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod9[1])
    print("Standard Deviation: ",mod9[2])
q3i()


#from sklearn.datasets import load_iris
#iris=load_iris()
#model=LinearSVC(max_iter=2000)
#score=cross_val_score(model, boston.data, y50list, scoring='accuracy', cv=10)
#print(score)
#print(np.mean(score))
#print(1-np.mean(score))
#print(digits.target)

#(ii)
from random import shuffle
import random

def my_train(model,X,y,pi,k):           ##helper function for my_train_test
    row=len(X)
    num=int(pi*row)
    scorelist=[]
    for i in range(0,k):
        Rdata=np.column_stack((X,y))
        shuffle(Rdata)
        trainset=Rdata[:num]
        testset=Rdata[num:]
        X_train=np.delete(trainset,-1,axis=1)
        y_train=trainset[:,-1]
        X_test=np.delete(testset,-1,axis=1)
        y_test=testset[:,-1]
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        scorelist+=[comp(y_pred,y_test)]
    scores=np.array(scorelist)
    error_rates=1-scores
    mean=np.mean(error_rates)
    std=np.std(error_rates)
    return error_rates,mean,std

def my_train_test(method,X,y,pi,k):    ##main part
    if method==LinearSVC:
        model=LinearSVC(max_iter=2000)
        return my_train(model,X,y,pi,k)
    elif method==SVC:
        model=SVC(gamma='scale', C=10)  
        return my_train(model,X,y,pi,k)
    elif method==LogisticRegression:
        model=LogisticRegression( penalty='l2', solver='lbfgs', multi_class='multinomial',max_iter=5000)
        return my_train(model,X,y,pi,k)
    else:
        return "Errormessage"
     
def q3ii():                     ##main part
    mod1=my_train_test(LinearSVC,boston.data,y50list,0.75,10)
    print("Error rates for LinearSVC with Boston50:")
    n=1
    for i in mod1[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod1[1])
    print("Standard Deviation: ",mod1[2])
    mod2=my_train_test(LinearSVC,boston.data,y25list,0.75,10)
    print("Error rates for LinearSVC with Boston25:")
    n=1
    for i in mod2[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod2[1])
    print("Standard Deviation: ",mod2[2])
    mod3=my_train_test(LinearSVC,digits.data,digits.target,0.75,10)
    print("Error rates for LinearSVC with Digits:")
    n=1
    for i in mod3[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod3[1])
    print("Standard Deviation: ",mod3[2])
    mod4=my_train_test(SVC,boston.data,y50list,0.75,10)
    print("Error rates for SVC with Boston50:")
    n=1
    for i in mod4[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod4[1])
    print("Standard Deviation: ",mod4[2])
    mod5=my_train_test(SVC,boston.data,y25list,0.75,10)
    print("Error rates for SVC with Boston25:")
    n=1
    for i in mod5[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod5[1])
    print("Standard Deviation: ",mod5[2])
    mod6=my_train_test(SVC,digits.data,digits.target,0.75,10)
    print("Error rates for SVC with Digits:")
    n=1
    for i in mod6[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod6[1])
    print("Standard Deviation: ",mod6[2])
    mod7=my_train_test(LogisticRegression,boston.data,y50list,0.75,10)
    print("Error rates for LogisticRegression with Boston50:")
    n=1
    for i in mod7[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod7[1])
    print("Standard Deviation: ",mod7[2])
    mod8=my_train_test(LogisticRegression,boston.data,y25list,0.75,10)
    print("Error rates for LogisticRegression with Boston25:")
    n=1
    for i in mod8[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod8[1])
    print("Standard Deviation: ",mod8[2])
    mod9=my_train_test(LogisticRegression,digits.data,digits.target,0.75,10)
    print("Error rates for LogisticRegression with Digits:")
    n=1
    for i in mod9[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod9[1])
    print("Standard Deviation: ",mod9[2])
#q3ii()
    
#4
def rand_proj(X,d):             ##main part
    col=np.shape(X)[1]
    G=np.random.normal(loc=0, scale=1, size=(col,d))
    G_M=np.mat(G)
    X1_bar=X*G_M
    return X1_bar

def quad_proj(X):             ##main part
    row=np.shape(X)[0]
    col=np.shape(X)[1]
    Xnew = np.empty(shape=[row,int(2*col+col*(col-1)/2)])
    for k in range(0,row):
        array_1=X[k]
        array_2=[]
        for i in array_1:
            array_2.append(i*i)
        array_3=[]
        for i in range(0,col):
            for j in range(0,col):
                if j>i:
                    array_3.append(array_1[i]*array_1[j])
                else:
                    continue
        array_1=list(array_1)
        array_t=array_1+array_2+array_3
        array_t=np.array(array_t)
        Xnew[k]=array_t
    return Xnew

def q4():              ##main part
    X1bar=rand_proj(digits.data,32)
    mod1=my_cross_val(LinearSVC,X1bar,digits.target,10)
    print("Error rates for LinearSVC with X1:")
    n=1
    for i in mod1[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod1[1])
    print("Standard Deviation: ",mod1[2])
    X2bar=quad_proj(digits.data)
    mod2=my_cross_val(LinearSVC,X2bar,digits.target,10)
    print("Error rates for LinearSVC with X2:")
    n=1
    for i in mod2[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod2[1])
    print("Standard Deviation: ",mod2[2])
    X1bar=rand_proj(digits.data,32)
    mod3=my_cross_val(SVC,X1bar,digits.target,10)
    print("Error rates for SVC with X1:")
    n=1
    for i in mod3[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod3[1])
    print("Standard Deviation: ",mod3[2])
    X2bar=quad_proj(digits.data)
    mod4=my_cross_val(SVC,X2bar,digits.target,10)
    print("Error rates for SVC with X2:")
    n=1
    for i in mod4[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod4[1])
    print("Standard Deviation: ",mod4[2])
    X1bar=rand_proj(digits.data,32)
    mod5=my_cross_val(LogisticRegression,X1bar,digits.target,10)
    print("Error rates for LogisticRegression with X1:")
    n=1
    for i in mod5[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod5[1])
    print("Standard Deviation: ",mod5[2])
    X2bar=quad_proj(digits.data)
    mod6=my_cross_val(LogisticRegression,X2bar,digits.target,10)
    print("Error rates for LogisticRegression with X2:")
    n=1
    for i in mod6[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod6[1])
    print("Standard Deviation: ",mod6[2])
q4()