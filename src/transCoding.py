from typing import Tuple
import pandas as pd
import torch

################进行编码并且将数据集划分为结果集和特征集###################

## TODO: 进行特征工程，筛选特征

# 此处我将处理好的数据集改名为了merged_data.csv
absolutePath = '../data/merged_data.csv'

# 编码规则如下:
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
    'IT staff': 17
}


def transCoding() -> Tuple[torch.Tensor, torch.Tensor,int]:
    havaShowed = []
    global absolutePath
    merged_data = pd.read_csv(absolutePath)
    merged_data = merged_data.drop(['reputation', 'Unnamed: 0', 'ID'], axis=1)
    merged_data.dropna(axis=0, how='any', inplace=True)
    merged_data = merged_data.replace(transCodingDict)
    tensor_data = torch.tensor(merged_data.values)
    resultTensor = tensor_data[:,-1]
    print('resultTensor:',resultTensor)
    dataTensor = tensor_data[:,:-1]
    print('dataTensor:',dataTensor)
    print('n_features:',dataTensor.shape[1])
    return resultTensor,dataTensor,dataTensor.shape[1]




if __name__ == '__main__':
    transCoding()
