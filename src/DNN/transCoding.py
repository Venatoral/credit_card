from os import path
from typing import Tuple
import pandas as pd
import torch
import numpy as np
from typing import Tuple
from imblearn.combine import SMOTEENN
import torch
import os
from sklearn.model_selection import train_test_split

################进行编码并且将数据集划分为结果集和测试集###################

# 普通硬编码规则
transCodingDict = {
    'M': 0,
    'F': 1,
    'N': 0,
    'Y': 1,
    'Working': 0,
    'Commercial associate': 1,
    'Pensioner': 2,
    'State servant': 3,
    'Student': 4,
    'Higher education': 0,
    'Secondary / secondary special': 1,
    'Incomplete higher': 2,
    'Lower secondary': 3,
    'Academic degree': 4,
    'Married': 0,
    'Single / not married': 1,
    'Civil marriage': 2,
    'Separated': 3,
    'Widow': 4,
    'House / apartment': 0,
    'With parents': 1,
    'Municipal apartment': 2,
    'Rented apartment': 3,
    'Office apartment': 4,
    'Co-op apartment': 5,
    'Laborers': 0,
    'Core staff': 1,
    'Sales staff': 2,
    'Managers': 3,
    'Drivers': 4,
    'Medicine staff': 5,
    'Security staff': 6,
    'Accountants': 7,
    'High skill tech staff': 8,
    'Cleaning staff': 9,
    'Private service staff': 10,
    'Cooking staff': 11,
    'Low-skill Laborers': 12,
    'Secretaries': 13,
    'Waiters/barmen staff': 14,
    'HR staff': 15,
    'Realty agents': 16,
    'IT staff': 17,
    'Unknown': 18,
    'more': 3,
    # 不知道为什么有部分的数字为字符串形式，此处需要转换为相应的数字
    '0': 0,
    '1.0': 1,
    '1': 1,
    '2.0': 2,
    '2': 2,
    '3.0': 3,
    '3': 3
}


def readFromFile(pathPrefix: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    X_b = pd.read_csv(pathPrefix+'X_b.csv')
    Y_b = pd.read_csv(pathPrefix+'Y_b.csv')
    X_test = pd.read_csv(pathPrefix+'X_test.csv')
    Y_test = pd.read_csv(pathPrefix+'Y_test.csv')
    return torch.tensor(Y_b.values).t()[0], torch.tensor(X_b.values),  torch.tensor(Y_test.values).t()[0], torch.tensor(X_test.values), X_b.shape[1]


def saveToFile(pathPrefix: str, dataDict: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    for key in dataDict.keys():
        dataDict[key].to_csv(pathPrefix+key+'.csv', index=False)

# 使用欠采样数据


def underSamplingTransCoding() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if os.path.exists('../../data/UNDERSAMPLING/X_b.csv'):
        return readFromFile('../../data/UNDERSAMPLING/')
    global absolutePath
    merged_data = pd.read_csv('../../data/undersampling.csv')
    merged_data = merged_data.drop(['Unnamed: 0', 'ID'], axis=1)
    merged_data.dropna(axis=0, how='any', inplace=True)
    merged_data = merged_data.replace(transCodingDict)
    result = merged_data['target']
    data = merged_data.drop(['target'], axis=1)
    X_b, X_test, Y_b, Y_test = train_test_split(
        data, result, test_size=0.1)
    saveToFile('../../data/UNDERSAMPLING/',
               {'X_b': X_b, 'Y_b': Y_b, 'X_test': X_test, 'Y_test': Y_test})
    return torch.tensor(Y_b.values), torch.tensor(X_b.values),  torch.tensor(Y_test.values), torch.tensor(X_test.values), X_b.shape[1]

##SMOTE+ENN##


def returnDealedData() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if os.path.exists('../../data/SMOTEENN/X_b.csv'):
        return readFromFile('../../data/SMOTEENN/')
    # 此处是用的数据已经经过特征工程处理
    credit = pd.read_csv('../../data/featureEngineering.csv')
    credit = pd.get_dummies(credit)
    feat_cols = credit.columns.to_list()
    feat_cols.remove('ID')
    feat_cols.remove('target')
    X, Y = credit[feat_cols], credit['target']
    X = X.drop(['Unnamed: 0'], axis=1)
    Y = Y.astype(int)
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    # X_b, Y_b = SMOTEENN().fit_resample(X_train, Y_train)
    X, Y = SMOTEENN().fit_resample(X, Y)
    X_b, X_test, Y_b, Y_test = train_test_split(X, Y, test_size=0.1)
    saveToFile('../../data/SMOTEENN/',
               {'X_b': X_b, 'Y_b': Y_b, 'X_test': X_test, 'Y_test': Y_test})
    return torch.tensor(Y_b.values), torch.tensor(X_b.values),  torch.tensor(Y_test.values), torch.tensor(X_test.values), X_b.shape[1]


##TEST##
if __name__ == '__main__':
    print(underSamplingTransCoding())
    print('===========================')
    print(returnDealedData())
    print('===========================')
