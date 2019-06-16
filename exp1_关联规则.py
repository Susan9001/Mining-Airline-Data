#-*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def get_typeRules(data, keys, k):
    '''把5列指标给离散化，即把每一列处于某个区间的数归为一类、另一区间又另一类
    :param k: 每一列都划分k个区间
    :return: 每个区间的最小值，以及该区间的样本个数
    '''
    result = pd.DataFrame()
    for i in range(len(keys)):
        # 调用k-means算法，进行聚类离散化
        print(u'正在进行“%s”的聚类...' % keys[i])
        kmodel = KMeans(n_clusters = k)
        kmodel.fit(data[[keys[i]]].as_matrix()) # 训练模型

        r1 = pd.DataFrame(kmodel.cluster_centers_, columns = [keys[i]]) # 聚类中心
        r2 = pd.Series(kmodel.labels_).value_counts() # 分类统计
        #print(kmodel.cluster_centers_)
        #print(kmodel.labels_)
        r2 = pd.DataFrame(r2, columns = [keys[i]+'n']) # 转为DataFrame，记录各个类别的数目
        r = (pd.concat([r1, r2], axis = 1)).sort_values(keys[i]) # 匹配聚类中心和类别数目
        r.index = [1, 2, 3, 4]

        r[keys[i]] = pd.rolling_mean(r[keys[i]], 2) # 用来计算相邻2列的均值，以此作为边界点。
        r[keys[i]][1] = 0.0 # 这两句代码将原来的聚类中心改为边界点。
        result = result.append(r.T)
    print(result.head())
    return result

def get_everyKind(data, tIndex, keys=list("LRFMC")):
    '''
    给每个LRFMC的每个指标都离散化为XN，X为LRFMC, N为1234区间号
    :param data:  原始LRFMC一共5列数据
    :param tIndex: LRFMC五行、各自程度为1,2,3,4的区间
    :return: 每行类似于L1,R2,F2,M0,C1...这样, L1的1表示L指标它根据区间分为1
    '''
    # 转换规则为字典
    rule_dict = {}
    for index, row in tIndex.iterrows() :
        rulelist = list(row)
        rule_dict[rulelist[0]] = rulelist[1 :].copy()
        rule_dict[rulelist[0]].append(1.01) # 最大
    # 分
    res_list = []
    for index, row in data.iterrows():
        tmp_lsit = []
        for key in keys:
            n = len(rule_dict[key]) # 含1
            for i in range(n - 1):
                if row[key] > rule_dict[key][i] and row[key] <= rule_dict[key][i + 1]:
                    tmp_lsit.append(key + str(i))
                    break
        if len(tmp_lsit) == len(keys):
            res_list.append(",".join(tmp_lsit))
    return res_list

# 连接函数，用于实现L_{k-1}到C_k的连接
def connect_string(x, ms):
    x = list(map(lambda i : sorted(i.split(ms)), x))
    l = len(x[0])
    r = []
    for i in range(len(x)):
        for j in range(i, len(x)) :
            if x[i][:l - 1] == x[j][:l - 1] and x[i][l - 1] != x[j][l - 1]:
                r.append(x[i][:l - 1] + sorted([x[j][l - 1], x[i][l - 1]]))
    return r

# 寻找关联规则的函数
def get_rules(d, support, confidence, ms='--') :
    result = pd.DataFrame(index=['support', 'confidence'])  # 定义输出结果
    support_series = 1.0 * d.sum() / len(d)  # 支持度序列
    column = list(support_series[support_series > support].index)  # 1-频繁的下标
    k = 0

    while len(column) > 1:
        k += 1
        print('\n正在进行第%d次搜索...' % k)
        column = connect_string(column, ms)
        print('数目：%d...' % len(column))
        sf = lambda i: d[i].prod(axis=1, numeric_only=True)  # 新一批支持度的计算函数

        d_2 = pd.DataFrame(list(map(sf, column)), index=[ms.join(i) for i in column]).T
        support_series_2 = 1.0 * d_2[[ms.join(i) for i in column]].sum() / len(d)  # 计算连接后的支持度
        column = list(support_series_2[support_series_2 > support].index)  # 新一轮支持度筛选
        support_series = support_series.append(support_series_2)
        column2 = [] # 备选的关联

        for i in column:
            i_list = i.split(ms)
            for j in range(len(i_list)):
                column2.append(i_list[:j] + i_list[j + 1 :] + i_list[j :j + 1])

        cofidence_series = pd.Series(index=[ms.join(i) for i in column2])  # 对于每个序列的置信度
        for i in column2:  # 计算置信度
            cofidence_series[ms.join(i)] = support_series[ms.join(sorted(i))] / support_series[ms.join(i[:len(i) - 1])]

        for i in cofidence_series[cofidence_series > confidence].index:  # 置信度筛选
            result[i] = 0.0
            result[i]['confidence'] = cofidence_series[i]
            result[i]['support'] = support_series[ms.join(sorted(i.split(ms)))]

    result = (result.T.sort_values(['confidence', 'support'], ascending=False))
    return result

def get_buyOrNotMatrix(data):
    '''一类(如A1,B2,C1...)为一列，每行顾客是否属于此类为0/1'''
    print(u'\n转换原始数据至0-1矩阵...')
    ct = lambda x : pd.Series(1, index = x[pd.notnull(x)])
    b = list(map(ct, data.as_matrix()))
    data = pd.DataFrame(b).fillna(0)
    return data

if __name__=='__main__':
    filepath = './data/air_LRFMC.csv'
    rulepath = './data/rules.csv'
    tagpath = './data/dis.txt'
    data = pd.read_csv(filepath).iloc[:,1:] # 去除索引号
    keys = list("LRFMC")
    # k = 4 # 每一类划分4个区间
    #rules = get_typeRules(data, keys, k)
    #rules.to_csv(rulepath)

    # rules = pd.read_csv(rulepath)
    # # data = data.sample(2000) # 不能全用，不然很慢的
    # tag_lsit = get_everyKind(data, rules)
    # with open(tagpath, 'w', encoding='utf-8') as file:
    #     file.write("\n".join(tag_lsit))

    # 开始进行关联分析
    appath = './data/association_matrix.csv'
    res_p = './data/association_result_'
    # data = get_buyOrNotMatrix(pd.read_csv(tagpath)) # 0-1矩阵
    # data.to_csv(appath)
    for i in range(6):
        support = 0.06 # 最小支持度
        confidence = 0.6 + 0.05 * i # 最小置信度
        data = pd.read_csv(appath).iloc[:,1:] # 去除index
        result = get_rules(data, support, confidence)
        print('\n结果为：')
        print(result)
        result.to_csv(res_p + ("%.2f" % confidence) + ".txt")





