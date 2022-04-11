# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 23:01:41 2020

@author: lxy
"""
import numpy as np
#print(np.array([1/2]*2))

#print(np.array([]))

#print(np.array([np.full((3, 3), 0.0)]*2))
#print(np.array([np.full((3), 0.0)]*1))
#print("abc")
#a=np.array(np.zeros((1,3)))
b=np.array([[[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]],
        [[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]]])
#a[ 1, : , :]=np.diag(np.diag(b[1,:,:]))
print(b[1])
print("abc")
print(b[1,:,:])
#x = [1,1,1,2,2,2,5,25,1,1]

#x,y=np.array(np.unique(x, return_counts=True))
#print(x)
#print(y)

#a=np.array(np.full((3),0.0))
#b=np.zeros(3)
#print(a)
#print(b)