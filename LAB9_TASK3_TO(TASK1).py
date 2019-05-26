# -*- coding: utf-8 -*-
"""
Created on Sat May 25 12:00:33 2019

@author: Administrator
"""

#调库
import pandas as pd
import numpy as np
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#导入数据-DataFrames
filename = 'E:/大学课程/AI程序设计/实验部分/实验9 聚类-关联-异常/实验课聚类关联分析/house-votes-84.csv'
data_origin = pd.read_csv(filename, engine = 'python')
k = 2
threshold = 2
iteration = 500

data_origin_matrix = data_origin.values
label_lst = []
for item in data_origin_matrix:
    if item[0]=='republican':
        label_lst.append(0)#共和党为0
    else:
        label_lst.append(1)#民主党为1
  
#形成matrix      
data_product=data_origin.drop('A',axis=1)
data_product_matrix = data_product.values
pri_X = data_product_matrix

#把string全都变成int
lst_extern = []

for item in pri_X:
    lst_essen = []
    for member in item:
        if member == 'n':
            lst_essen.append(-1)
        elif member == 'y':
            lst_essen.append(1)
        else:
            lst_essen.append(0)
    lst_extern.append(lst_essen)

X = np.mat(lst_extern)

#降维
pca = PCA(n_components = 2)
reduced_X = pca.fit_transform(X)

#
X_lst = []
Y_lst = []

for item in reduced_X:
    X_lst.append(item[0])
    Y_lst.append(item[1])
    
lst1 = []
lst2 = []
for item in reduced_X:
    lst1.append(item[0])
    lst2.append(item[1])

data = np.array(reduced_X)
DF = pd.DataFrame(data,index = range(435))

clf = KMeans(n_clusters=2)
model = KMeans(n_clusters = k, max_iter  = iteration)
data_fit = model.fit(DF)
y_pred = clf.fit_predict(reduced_X)

DF_result = pd.concat([DF, pd.Series(model.labels_, index = DF.index)], axis = 1)  #每个样本对应的类别
DF_result.columns = list(DF.columns) + [u'聚类类别'] #重命名表头

DF_norm = []
for i in range(2):
    norm_tmp = DF_result[[0,1]][DF_result['聚类类别']==i]-model.cluster_centers_[i]
    norm_tmp = norm_tmp.apply(np.linalg.norm,axis = 1)
    DF_norm.append(norm_tmp/norm_tmp.median())

DF_norm = pd.concat(DF_norm)

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
DF_norm[DF_norm <= threshold].plot(style = 'go') #正常点

discrete_points = DF_norm[DF_norm > threshold] #离群点
discrete_points.plot(style = 'ro')


for i in range(len(discrete_points)): #离群点做标记
    id = discrete_points.index[i]
    n = discrete_points.iloc[i]
    plt.annotate('(%s, %0.2f)'%(id, n), xy = (id, n), xytext = (id, n))


plt.xlabel('编号')
plt.ylabel('相对距离')
plt.show()

DF_norm_0 = []
for i in range(1):
    norm_tmp = DF_result[[0,1]][DF_result['聚类类别']==i]-model.cluster_centers_[i]
    norm_tmp = norm_tmp.apply(np.linalg.norm,axis = 1)
    DF_norm_0.append(norm_tmp/norm_tmp.median())

DF_norm_0 = pd.concat(DF_norm_0)

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
DF_norm_0[DF_norm_0 <= threshold].plot(style = 'go') #正常点

discrete_points = DF_norm_0[DF_norm_0 > threshold] #离群点
discrete_points.plot(style = 'ro')


for i in range(len(discrete_points)): #离群点做标记
    id = discrete_points.index[i]
    n = discrete_points.iloc[i]
    plt.annotate('(%s, %0.2f)'%(id, n), xy = (id, n), xytext = (id, n))


plt.xlabel('编号')
plt.ylabel('相对距离')
plt.show()

DF_norm_1 = []
for i in range(1,2):
    norm_tmp = DF_result[[0,1]][DF_result['聚类类别']==i]-model.cluster_centers_[i]
    norm_tmp = norm_tmp.apply(np.linalg.norm,axis = 1)
    DF_norm_1.append(norm_tmp/norm_tmp.median())

DF_norm_1 = pd.concat(DF_norm_1)

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
DF_norm_1[DF_norm_1 <= threshold].plot(style = 'go') #正常点

discrete_points = DF_norm_1[DF_norm_1 > threshold] #离群点
discrete_points.plot(style = 'ro')


for i in range(len(discrete_points)): #离群点做标记
    id = discrete_points.index[i]
    n = discrete_points.iloc[i]
    plt.annotate('(%s, %0.2f)'%(id, n), xy = (id, n), xytext = (id, n))


plt.xlabel('编号')
plt.ylabel('相对距离')
plt.show()
