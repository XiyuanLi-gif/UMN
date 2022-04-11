# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 21:03:53 2020

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

# Hw2 Q3

class MultiGaussClassify:
    def __init__(self,k,d,diag=False):
        self.k=k
        self.d=d
        self.diag=diag
        self.prior=np.array([1/self.k]*self.k)
        self.mean=np.array([np.full((self.d), 0.0)]*self.k)
        self.covar=np.array([np.full((self.d,self.d), 0.0)]*self.k)
            
        
    def fit(self,X,y):
        self.lable,self.lable_counts=np.unique(y,return_counts=True)
        self.prior=self.lable_counts/len(y)
        n=0
        for i in self.lable:
            subMat = X[np.where(y == i)][0]
            self.mean[n] = subMat.sum(axis=0)
            self.mean[n] /= self.lable_counts[n]
            ## mean=sum(xt)/N
            n+=1
        m=0
        for j in self.lable:
            ## covariance:: sum(xt-m)(xt-m)/N
            subMat = X[np.where(y == j)][0]
            for s in range(0,subMat.shape[0]):    
                diff=subMat[s]-self.mean[m]
                self.covar[m]+=(np.matmul(diff,np.transpose(diff)))/(self.lable_counts[m])
                s+=1
            self.covar[m]+=np.eye(X.shape[1])*1e-6
            if self.diag==True:  ## diagonal convariance matrix
                self.diagonal=np.diag(self.covar[m])
                self.covar[m]=np.diag(self.diagonal)
            m+=1
            
        
    def predict(self,X):
        prelist=np.array(np.full((X.shape[0]),0.0))
        for i in range(0,X.shape[0]):
            prelist2=np.array(np.full((self.k),0.0))
            for j in range(0,self.k):
                ##print(self.covar[j])
                ## - (1/2)*log(determinat(estimated_covariance))
                p1=(-(1/2))*np.log(np.linalg.det(self.covar[j]))
                diff2=X[i]-self.mean[j]
                ## -  (1/2)*((x-mean)^T)*((estimated_covariance)^-1)*(x-mean)
                p2=(-(1/2))*np.matmul(np.matmul(np.transpose(diff2),np.linalg.inv(self.covar[j])),diff2)
                ##log(p(Ci))
                p3=np.log(self.prior[j])      
                prelist2[j]=p1+p2+p3
            prelist[i]=self.lable[np.argmax(prelist2)]
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
    elif method=="multigaussclassify":
        kk=len(np.unique(y))
        dd=X.shape[1]
        model=MultiGaussClassify(kk,dd,False)
        return my_accuracy(model,X,y,k)
    elif method=="multigaussdiagclassify":
         kk=len(np.unique(y))
         dd=X.shape[1]
         model=MultiGaussClassify(kk,dd,True)
         return my_accuracy(model,X,y,k)
    else:
        return "Errormessage"
    
def get_params(self, deep=True):
	return {"k": self.k, "d": self.dim, "diag": self.diagonal}
    
def hw2q3():
    mod1=my_cross_val("multigaussclassify",boston.data,y50list,5)
    print("Error rates for MultiGaussClassify with full covariance matrix on Boston50:")
    n=1
    for i in mod1[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod1[1])
    print("Standard Deviation: ",mod1[2])
    mod2=my_cross_val("multigaussclassify",boston.data,y25list,5)
    print("Error rates for MultiGaussClassify with full covariance matrix on Boston25:")
    n=1
    for i in mod2[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod2[1])
    print("Standard Deviation: ",mod2[2])
    mod3=my_cross_val("multigaussclassify",digits.data,digits.target,5)
    print("Error rates for MultiGaussClassify with full covariance matrix on Digits:")
    n=1
    for i in mod3[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod3[1])
    print("Standard Deviation: ",mod3[2])
    mod4=my_cross_val("multigaussdiagclassify",boston.data,y50list,5)
    print("Error rates for MultiGaussClassify with diagonal covariance matrix on Boston50:")
    n=1
    for i in mod4[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod4[1])
    print("Standard Deviation: ",mod4[2])
    mod5=my_cross_val("multigaussdiagclassify",boston.data,y25list,5)
    print("Error rates for MultiGaussClassify with diagonal covariance matrix on Boston25:")
    n=1
    for i in mod5[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod5[1])
    print("Standard Deviation: ",mod5[2])
    mod6=my_cross_val("multigaussdiagclassify",digits.data,digits.target,5)
    print("Error rates for MultiGaussClassify with diagonal covariance matrix on Digits:")
    n=1
    for i in mod6[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod6[1])
    print("Standard Deviation: ",mod6[2])
    mod7=my_cross_val(LogisticRegression,boston.data,y50list,5)
    print("Error rates for LogisticRegression with Boston50:")
    n=1
    for i in mod7[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod7[1])
    print("Standard Deviation: ",mod7[2])
    mod8=my_cross_val(LogisticRegression,boston.data,y25list,5)
    print("Error rates for LogisticRegression with Boston25:")
    n=1
    for i in mod8[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod8[1])
    print("Standard Deviation: ",mod8[2])
    mod9=my_cross_val(LogisticRegression,digits.data,digits.target,5)
    print("Error rates for LogisticRegression with Digits:")
    n=1
    for i in mod9[0]:
       print("Fold ",n,":",i,)
       n+=1
    print("Mean: ",mod9[1])
    print("Standard Deviation: ",mod9[2])
    
hw2q3()

#print(np.array([1/2]*2))

#print(np.array([]))

#print(np.array([np.full((3, 3), 0.0)]*2))
#print(np.array([np.full((3), 0.0)]*1))
#print("abc")
#a=np.array(np.zeros((1,3)))
#b=np.array([[[1, 2, 3],
#      [4, 5, 6],
#       [7, 8, 9]],
#        [[1, 2, 3],
##       [4, 5, 6],
 #      [7, 8, 9]]])
#a[ 1, : , :]=np.diag(np.diag(b[1,:,:]))
#print(b[1])
#print("abc")
#print(b[1,:,:])
#x = [1,1,1,2,2,2,5,25,1,1]

#x,y=np.array(np.unique(x, return_counts=True))
#print(x)
#print(y)

#a=np.array(np.full((3),0.0))
#b=np.zeros(3)
#print(a)
#print(b)
