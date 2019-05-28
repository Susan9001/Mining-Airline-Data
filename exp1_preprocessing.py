#-*- coding: utf-8 -*-
#数据清洗，过滤掉不符合规则的数据

import pandas as pd

def clean_data(data):
    '''
    去掉na及一些可能出错、没太多意义的数据
    :param datafile: 原始数据文件路径
    :return: 清洗完的数据
    '''
    # 去掉na
    data = data.dropna()
    # 只保留票价非零的，或者平均折扣率与总飞行公里数同时为0的记录。
    index1 = data['SUM_YR_1'] != 0
    index2 = data['SUM_YR_2'] != 0
    index3 = (data['SEG_KM_SUM'] == 0) & (data['avg_discount'] == 0)
    data = data[index1 | index2 | index3]
    return data

def simplify_data(data):
    '''
    选择比较有用的6个属性: FFP_DATE、LOAD_TIME、FLIGHT_COUNT、AVG_DISCOUNT、SEG_KM_SUM、LAST_TO_END
    分别是：第一次飞行日期，观察窗口结束时间，飞行次数，平均折扣，总飞行公里数，（到观察结束为止）多少未飞行
    并且把日期转换为
    :param datafile: input path
    :param okfile: output filepath
    :return: output
    '''
    valid_keys = ["FFP_DATE", "LOAD_TIME", "FLIGHT_COUNT", "avg_discount", "SEG_KM_SUM", "LAST_TO_END"]
    data = data[valid_keys]
    data.to_csv(okfile, encoding='utf-8')
    return data

#%%
def proc_indecator(data):
    '''
    获取以下5个对于客户的评价指标，并进行正则化
    客户关系时长L、消费时间间隔R、消费频率F、飞行里程M和折扣系数的平均值C五个指
    :param datafile: input path
    :return: output
    '''
    res_data = pd.DataFrame()
    # L: 乘客入会了的月份数，代表新/老程度
    ffp_time = pd.to_datetime(data["FFP_DATE"], format="%Y/%m/%d")
    load_time = pd.to_datetime(data["LOAD_TIME"], format="%Y/%m/%d")
    res_data["L"] = (load_time - ffp_time).apply(lambda x: int(int(((str(x)).split()[0])) / 30))
    # R: 客户最后一次坐飞机至今的月份数
    res_data["R"] = data['LAST_TO_END']
    # F: 超过你坐飞机的次数
    res_data["F"] = data['FLIGHT_COUNT']
    # M: 总里程数
    res_data["M"] = data['SEG_KM_SUM']
    # C: 平均折扣率
    res_data["C"] = data['avg_discount']
    return res_data
#%%

if __name__ == '__main__':
    originfile = './data/air_data.csv' # 航空原始数据,第一行为属性标签
    #cleanedfile = './data/air_data_cleaned.csv' # 数据清洗后保存的文件
    okfile = './data/air_ok.csv' # 取了6个特征之后
    ind_file = './data/air_LRFMC.csv'

    data = pd.read_csv(okfile, encoding='utf-8')
    data_with_indicator = proc_indecator(data)
    data_with_indicator.to_csv(ind_file)

