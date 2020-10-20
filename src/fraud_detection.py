# coding = utf-8
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import warnings
import pandas as pd
from matplotlib import pyplot as plt
from xgboost import XGBClassifier

# igore all warnings
warnings.filterwarnings('ignore')
# datasets
datasets_name = '../data/credit.csv'


# read and clean up the datasets
def load_credit() -> pd.DataFrame:
    df = pd.read_csv(datasets_name)
    # delete useless column
    df = df.drop(columns='Unnamed: 0', axis=1)
    return df


def preprocess(credit: pd.DataFrame) -> pd.DataFrame:
    # 处理二分变量, FLAG, Gender...
    credit['month_on_book'] = -credit['begin_month']
    # 将年龄和工龄转换为年
    credit['DAYS_BIRTH'] = -credit['DAYS_BIRTH'] / 365
    credit['DAYS_EMPLOYED'] = credit['DAYS_EMPLOYED'] / 365
    credit.rename(columns={'DAYS_BIRTH': 'age',
                           'DAYS_EMPLOYED': 'work_year'}, inplace=True)
    credit.drop(columns='begin_month', axis=1, inplace=True)
    # 将工资按照 'k'为单位
    credit['AMT_INCOME_TOTAL'] = credit['AMT_INCOME_TOTAL'] / 1000
    # 处理flag类二分数据
    dic = {'Y': 1, 'N': 0}
    credit['FLAG_OWN_CAR'] = credit['FLAG_OWN_CAR'].replace(dic)
    credit['FLAG_OWN_REALTY'] = credit['FLAG_OWN_REALTY'].replace(dic)
    credit['CODE_GENDER'] = credit['CODE_GENDER'].replace({'M': 1, 'F': 0})
    # 对于缺失的工作column，由于其缺失值 11323过大，我们对其该column进行舍弃
    credit['OCCUPATION_TYPE'].fillna('unknown', inplace=True)
    # 使用插值法，将异常值替换为去掉异常值之后的平均值
    credit.loc[credit['work_year'] > 0, 'work_year'] = credit[credit['work_year'] < 0]['work_year'].max()
    credit['work_year'] = -credit['work_year']
    # 丢弃 children column
    credit.drop(columns='CNT_CHILDREN', inplace=True)
    # 将3个及其以上的家庭CNT_FAM_MEMBERS设为more
    credit.loc[credit['CNT_FAM_MEMBERS'] >= 3, 'CNT_FAM_MEMBERS'] = 'more'
    # 我们将用户划分为 拥有/无 住房和公寓（1/0），从而将上述多类型变量转换为二分类 HOUSING_STATUS
    credit['housing-status'] = credit['NAME_HOUSING_TYPE'].apply(
        lambda x: 1 if x == 'House / apartment' else 0
    )
    credit.drop(columns='NAME_HOUSING_TYPE', inplace=True)
    # 划分婚姻状况
    credit['marriage_status'] = credit['NAME_FAMILY_STATUS'].apply(
        lambda x: 1 if (x == 'Married' or x == 'Civil marriage') else 0
    )
    credit.drop(columns='NAME_FAMILY_STATUS', inplace=True)
    # 无量纲化
    scl = StandardScaler()
    rbs = RobustScaler()
    # 年龄和帐龄几乎无离群值，正常处理
    scl_data = scl.fit_transform(credit[['age', 'month_on_book']])
    # 离群值较多使用 RobustScaler 处理
    rbs_data = rbs.fit_transform(credit[['work_year', 'AMT_INCOME_TOTAL']])
    credit[['age', 'month_on_book']] = scl_data
    credit[['work_year', 'AMT_INCOME_TOTAL']] = rbs_data
    # 丢弃部分列
    credit = credit.drop(columns=['FLAG_EMAIL', 'FLAG_WORK_PHONE', 'housing-status'])
    return credit


# 画出混淆矩阵
def draw_cm(cm, classes, title='Confused Matrix'):
    indices = range(len(cm))
    plt.title(title)
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    cm = cm / cm.sum(axis=0)[:, ]
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            plt.text(i, j, format(cm[i][j], '.2f'),
                     horizontalalignment='center')


# 画学习曲线
def draw_learn_curve(model, x, y):
    train_sizes, train_score, test_score = learning_curve(
        model, x, y, train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1], cv=10, scoring='roc_auc'
    )
    train_mean = np.mean(train_score, axis=1)
    test_mean = np.mean(test_score, axis=1)
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='training')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='testing')
    plt.legend(loc='best')
    plt.xlabel('traing examples')
    plt.ylabel('Score')
    plt.show()


# 平衡数据集，根据over_sampling 采取不同措施
def balance(x, y, over_sampling: bool = False):
    return SMOTEENN().fit_resample(x, y) if over_sampling else RandomUnderSampler().fit_resample(x, y)


def xgb_train(x, y):
    xg = XGBClassifier()
    gs_xg = GridSearchCV(xg)


if __name__ == '__main__':
    credit_df = load_credit()
    credit_df: pd.DataFrame = preprocess(credit_df)
    credit_df = pd.get_dummies(credit_df)
    feat_col = credit_df.columns.to_list()
    feat_col.remove('ID')
    feat_col.remove('target')
    X = credit_df[feat_col]
    Y = credit_df['target']
    Y = Y.astype(int)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=10086)
    # 平衡数据集
    X_b, Y_b = balance(X_train, Y_train, over_sampling=False)
