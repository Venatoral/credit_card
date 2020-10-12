# coding = utf-8
# igore all warnings
import warnings

import pandas as pd
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')
# datasets
datasets_name = '../data/credit.csv'


# read and clean up the datasets
def load_credit() -> pd.DataFrame:
    df = pd.read_csv(datasets_name)
    # delete useless column
    df = df.drop(columns='Unnamed: 0', axis=1)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # 处理二分变量, FLAG, Gender...
    df['CODE_GENDER'].replace(['F', 'M'], [0, 1], inplace=True)
    df['FLAG_OWN_CAR'].replace(['N', 'Y'], [0, 1], inplace=True)
    df['FLAG_OWN_REALTY'].replace(['N', 'Y'], [0, 1], inplace=True)
    # 通过 dep_value 得到target 之后丢弃 dep_value列
    df.drop(columns='dep_value', axis=1, inplace=True)
    return df


import itertools
import numpy as np


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
if __name__ == '__main__':
    credit_data = load_credit()
    credit_data = preprocess(credit_data)
    rf = RandomForestClassifier(n_estimators=50)
    credit_data = pd.get_dummies(credit_data)
    print(credit_data.columns)
    X = credit_data[[
        'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
        'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'FLAG_MOBIL',
        'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'CNT_FAM_MEMBERS',
        'NAME_INCOME_TYPE_Commercial associate',
        'NAME_INCOME_TYPE_Pensioner', 'NAME_INCOME_TYPE_State servant',
        'NAME_INCOME_TYPE_Student', 'NAME_INCOME_TYPE_Working',
        'NAME_EDUCATION_TYPE_Academic degree',
        'NAME_EDUCATION_TYPE_Higher education',
        'NAME_EDUCATION_TYPE_Incomplete higher',
        'NAME_EDUCATION_TYPE_Lower secondary',
        'NAME_EDUCATION_TYPE_Secondary / secondary special',
        'NAME_FAMILY_STATUS_Civil marriage', 'NAME_FAMILY_STATUS_Married',
        'NAME_FAMILY_STATUS_Separated',
        'NAME_FAMILY_STATUS_Single / not married', 'NAME_FAMILY_STATUS_Widow',
        'NAME_HOUSING_TYPE_Co-op apartment',
        'NAME_HOUSING_TYPE_House / apartment',
        'NAME_HOUSING_TYPE_Municipal apartment',
        'NAME_HOUSING_TYPE_Office apartment',
        'NAME_HOUSING_TYPE_Rented apartment', 'NAME_HOUSING_TYPE_With parents',
        'OCCUPATION_TYPE_Accountants', 'OCCUPATION_TYPE_Cleaning staff',
        'OCCUPATION_TYPE_Cooking staff', 'OCCUPATION_TYPE_Core staff',
        'OCCUPATION_TYPE_Drivers', 'OCCUPATION_TYPE_HR staff',
        'OCCUPATION_TYPE_High skill tech staff', 'OCCUPATION_TYPE_IT staff',
        'OCCUPATION_TYPE_Laborers', 'OCCUPATION_TYPE_Low-skill Laborers',
        'OCCUPATION_TYPE_Managers', 'OCCUPATION_TYPE_Medicine staff',
        'OCCUPATION_TYPE_Private service staff',
        'OCCUPATION_TYPE_Realty agents', 'OCCUPATION_TYPE_Sales staff',
        'OCCUPATION_TYPE_Secretaries', 'OCCUPATION_TYPE_Security staff',
        'OCCUPATION_TYPE_Waiters/barmen staff'
    ]]
    Y = credit_data['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.3)
    print(rf.get_params().keys())
    parameters = {}
    grid = GridSearchCV(estimator=rf, param_grid=parameters, n_jobs=-1, refit=True, cv=5)
    grid.fit(X_train, Y_train)
    est = grid.best_estimator_
    print(grid.best_score_)

