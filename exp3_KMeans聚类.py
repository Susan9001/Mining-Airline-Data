#-*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
from functools import reduce

def getDistance(xlist1:list, xlist2:list):
    '''获取xlist1和xlist2欧氏距离'''
    return np.sqrt(np.sum(np.power([xlist1[i] - xlist2[i] for i in range(len(xlist1))], 2)))

def getRandCenter(data:pd.DataFrame, k:int):
    '''随机生成k个簇的中心'''
    feature_count = data.shape[1] # 列数
    centers = np.zeros((k, feature_count))
    for i in range(feature_count):
        # 选取第i列的最大最小，作为范围
        data_min, data_max = np.min(data[:,i]), np.max(data[:, i])
        # 生成随机
        centers[:, i] = data_min + np.random.rand(k) * (data_max - data_min)
    return centers

def clusterByKMeans(data, k=3) :
    '''kmeans聚类'''
    m = data.shape[0]
    res_cluster = np.mat(np.zeros((m, 2)))  # 用于存放该样本属于哪类及质心距离
    # res_cluster第一列存放该数据所属的中心点，第二列是该数据到中心点的距离
    centers = getRandCenter(data, k)
    hasChanges = True  # 聚类是否已经收敛，若收敛则结束

    while hasChanges:
        hasChanges = False
        for i in range(m):  # 把每一个数据点划分到离它最近的中心点
            min_dis = np.inf # min_distance
            min_index = -1
            for j in range(k):
                curr_min = getDistance(centers[j, :], data[i, :])
                if curr_min < min_dis: # 更新最小距离
                    min_dis = curr_min
                    min_index = j
            if res_cluster[i, 0] != min_index:
                hasChanges = True  # 若变了，则要继续迭代
            res_cluster[i, :] = min_index, min_dis ** 2  # 第i个数据点的分配情况存入字典
        # print(centers) # 每次迭代的中心

        for cent in range(k) :  # 重新计算中心点
            ptsInClust = data[np.nonzero(res_cluster[:, 0].A == cent)[0]]
            centers[cent, :] = np.mean(ptsInClust, axis=0)  # 算出这些数据的中心点

    return centers, res_cluster

if __name__ == '__main__':
    filepath = './data/air_LRFMC.csv'
    respath = './data/kmeans_result.csv'
    data = pd.read_csv(filepath)
    mData = data.as_matrix()
    centers, res_cluster = clusterByKMeans(mData)
    print("-----------centers------------")
    print(centers)
    print("-----------result------------")
    print(res_cluster)
    # 保存
    cluster_list = [int(res_cluster[i,0]) for i in range(len(res_cluster))]
    data = pd.concat([data, pd.Series(cluster_list, index=data.index)], axis=1)
    data.to_csv(respath)



