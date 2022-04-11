# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 06:13:37 2021

@author: lxy
"""

import numpy as np
def Generatecoin(p):
    i=0
    r=7
    while i!=1:
        toss1=np.random.binomial(1,p,2)
        toss=[]
        for i in toss1:
            toss+=[i]
        if toss==[0,1]:
            r=1
            i=1
        elif toss==[1,0]:
            r=0
            i=1
        else:
            i=0
    return r

def coin1in2(p):
    sample=[]
    i=0
    while i<10: 
        r=Generatecoin(p)
        sample+=[r]
        i=i+1
    return sample
def coin1in3(p):
    sample=[]
    subsample=[]
    i=0
    while i<10:
        subsample=coin1in2(p)
        if subsample[0:2]==[1,1]:
            x='A'
            sample+=[x]
            i=i+1
        elif subsample[0:2]==[1,0]:
            x='B'
            sample+=[x]
            i=i+1
        elif subsample[0:2]==[0,1]:
            x='C'
            sample+=[x]
            i=i+1
        else:
            i=i
    for i in range(len(sample)):
        if sample[i]=='A':
            sample[i]=1
        elif sample[i]=='B' or sample[i] =='C':
            sample[i]=0
    return sample

#a=coin1in2(0.3)
b=coin1in3(0.3)
print(b)
