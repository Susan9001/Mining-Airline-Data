#-*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def getLabels(data, freq = 0.06):
    '''
    以F==freq为界限，分两类，列为label并去除F这一列
    :param freq: 阈值
    :return: 加了label之后
    '''
    data['F'] = data['F'].apply(lambda x : 1 if x >= freq else 0)
    x, y= data[list("LRMC")], data['F']
    return (x, y)

def getDistance(xlist1:list, xlist2:list):
    '''获取xlist1和xlist2欧氏距离'''
    return np.sqrt(np.sum(np.power([xlist1[i] - xlist2[i] for i in range(len(xlist1))], 2)))

def getKnnClassify(x_test:pd.DataFrame, x_train:pd.DataFrame, y_train:pd.Series, k=4, y_test=None):
    '''
    用K临近分类，根据LRMC这4个指标，把数据分为F指标下的两类
    :param x: LRMC这4列
    :return: y_predict即每一行的分类结果
    '''
    # y_predict = pd.DataFrame(index=x_test.index)
    y_predict = []
    j = -1
    for i_test, row_test in x_test.iterrows():
        lrmc_list = list(row_test) # 一行的lrmc指标
        d_list = [] # 欧式距离
        ytrain_list = list(y_train)
        i = 0
        for i_train, row_train in x_train.iterrows():
            curr_d = getDistance(lrmc_list, list(row_train))
            d_list.append((y_train.iloc[i], curr_d)) # d_list: (class, distance)
            i += 1
        # 选择K近邻
        d_list = sorted(d_list,key =lambda x: x[1])[:k]
        # 获取概率
        count_dict = dict().fromkeys(set(y_train), 0)
        for tup in d_list:
            count_dict[tup[0]] += 1
        # 添加概率最大者
        y_predict.append(max(count_dict.items(), key=lambda x:x[1])[0])
        j+=1

    return pd.Series(y_predict, x_test.index) # 一定要series!


if __name__ == '__main__':
    filepath = './data/air_LRFMC.csv'
    resfile = './data/knn_pred_'
    trufile = './data/knn_true_'
    bothfile = './data/knn_both_'
    data = pd.read_csv(filepath).sample(2000)
    x, y = getLabels(data)
    # 分测试集和验证集
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.1, random_state=211)

    best_k = 2
    best_score = np.inf
    best_pred = pd.Series()
    for k in range(2, 9):
        y_predict = getKnnClassify(x_test, x_train, y_train, k, y_test)
        # 计算平方差损失，并更新最佳
        curr_score = mean_squared_error(y_test, y_predict)
        print("k = %d, MSE损失%f" % (k, curr_score))
        if (curr_score < best_score):
            best_score, best_k, best_pred = curr_score, k, y_predict

    # 写结果
    print("best k = %d, 损失%f" % (best_k, best_score))
    pd.DataFrame(pd.concat([x_test, y_test], axis=1)).to_csv(trufile + str(best_k) +".csv")
    pd.DataFrame(pd.concat([x_test,best_pred], axis=1)).to_csv(resfile + str(best_k) +".csv")
    pd.DataFrame(pd.concat([best_pred, y_test], axis=1)).to_csv(bothfile + str(best_k) +".csv")

    # 再次读出来的时候：
    # pred_true = pd.read_csv(bothfile, index_col=0)
    # print(mean_squared_error(pred_true['0'], pred_true['F']))




