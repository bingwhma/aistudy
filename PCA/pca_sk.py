# coding=utf-8
from sklearn.decomposition import PCA  
import numpy as np

dataMat=np.array([
    [84,65,61,72,79,81],
    [64,77,77,76,55,70],
    [65,67,63,49,57,67],
    [74,80,69,75,63,74],
    [84,74,70,80,74,82],
    ])


pca_sk = PCA(n_components=3) 
newMat = pca_sk.fit_transform(dataMat) 

print(newMat)

newMat = pca_sk.fit(dataMat) 
print(newMat)