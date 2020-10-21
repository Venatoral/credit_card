# **表格字段说明**

<br/>

### **[数据集来源](https://www.kaggle.com/rikdifos/credit-card-approval-prediction)**

<br/>

## **application_record.csv**

| 字段名              | 解释           | 备注                                           |
|---------------------|----------------|------------------------------------------------|
| ID                  | 客户号         |是一串纯数字字符串                                                |
| CODE_GENDER         | 性别           |M:男性 F:女性                                                |
| FLAG_OWN_CAR        | 是否有车       |Y:有 N:无                                                |
| FLAG_OWN_REALTY     | 是否有房产     |Y:有 N:无                                                  |
| CNT_CHILDREN        | 孩子个数       |INT                                                |
| AMT_INCOME_TOTAL    | 年收入         |INT                                                |
| NAME_INCOME_TYPE    | 收入类别       | <p>Working : 打工收入<br/>Commercial associate : 经商收入<br/>Pensioner : 养老金<br/>State servant : 公务员收入<br/>Student : 学生零花钱</p>
| NAME_EDUCATION_TYPE | 教育程度       |<p>Higher education : 受过高等教育(本科以上)<br/>Secondary / secondary special : 高中/大专<br/>Incomplete higher : 受过高等教育，未毕业(本科辍学)<br/>Lower secondary : 初中<br/>Academic degree : 至少本科</p>                                                |
| NAME_FAMILY_STATUS  | 婚姻状态       | <p>Married : 已结婚<br/>Single / not married : 单身/未结婚<br/>Civil marriage : 已订婚<br/>Separated : 离异<br/>Widow : 丧偶</p>                                                |
| NAME_HOUSING_TYPE   | 居住方式       |<p>House / apartment : 住在自己的房屋,公寓<br/>With parents : 和父母一起居住<br/>Municipal apartment : 住在市政公寓<br/>Rented apartment : 租房子住<br/>Office apartment : 办公公寓/员工宿舍<br/>Co-op apartment : 住房合作社</p>                                                |
| DAYS_BIRTH          | 生日           | 0为当日，日期向前计算，比如-28为28天前出生     |
| DAYS_EMPLOYED       | 开始工作日期   | 0为当日，日期向前计算，比如-28为28天前开始工作 |
| FLAG_MOBIL          | 是否有手机     |Y:有 N:无                                                 |
| FLAG_WORK_PHONE     | 是否有工作电话 |Y:有 N:无                                                 |
| FLAG_PHONE          | 是否有电话     |Y:有 N:无                                                 |
| FLAG_EMAIL          | 是否有 email   |Y:有 N:无                                                 |
| OCCUPATION_TYPE     | 职业           |<p>Laborers : 打工者<br/>Core staff : 企业高层员工<br/>Sales staff : 销售业员工<br/>Managers : 经理<br/>Drivers : 司机<br/>Security staff : 安保人员<br/>Medicine staff : 医疗工作者<br/>Accountants : 会计<br/>High skill tech staff : 高级技工<br/>Cleaning staff : 清洁行业工人<br/>Private service staff : 私人服务职工<br/>Cooking staff : 餐饮业从业员<br/>Low-skill Laborers : 低技能体力劳动者<br/>Secretaries : 秘书<br/>Waiters/barmen staff : 服务员<br/>HR staff : 企业人事部员工<br/>Realty agents : 房地产经纪人<br/>IT staff : IT行业从业人事</p>                                                |
| CNT_FAM_MEMBERS     | 家庭人数       |INT                                                |

## **credit_record.csv**

| 字段名         | 解释     | 备注                                                                                                                                 |
|----------------|----------|--------------------------------------------------------------------------------------------------------------------------------------|
| ID             | 客户号   |                                                                                                                                      |
| MONTHS_BALANCE | 记录月份 | 已抽取数据月份为起点，向前倒退，0为当月，-1为前一个月，依次类推                                                                      |
| STATUS         | 状态     | 0:1-29 天逾期<br/> 1:30-59 天逾期 <br/>2:60-89 天逾期<br/> 3:90-119 天逾期<br/> 4:120-149 天逾期<br/> 5:150天以上逾期或坏账、核销<br/> C: 当月已还清<br/> X: 当月无借款<br/> |


## **项目目录结构**

```
├─.DS_Store  
├─introduction.ipynb  //说明文件
├─README.md  
├─src  
|  ├─trainedModel  //各个模型训练好之后导出的外部存储文件
|  |      ├─dnn.pt  
|  |      ├─lightgbm.pickle  
|  |      ├─lr.pickle  
|  |      ├─rf.pickle  
|  |      ├─svc.pickle  
|  |      └xg.pickle  
|  ├─TraditionalAlgorithm  //传统机器学习算法
|  |          ├─.DS_Store 
|  |          ├─fraud_detection.ipynb //各个算法的ipy
|  |          ├─fraud_detection.py //各个算法的py
|  |          ├─ml_detection.ipynb //数据统计和处理的ipy
|  |          ├─xb_fraud_detection.ipynb //xgboost
|  |          ├─.ipynb_checkpoints
|  |          |         └fraud_detection-checkpoint.ipynb //运行结果
|  ├─DNN  //DNN多层神经网络
|  |  ├─annealingTuning.py  //退火超参数寻优
|  |  ├─bpNeuralNetworks.py //bp神经网络
|  |  ├─confusionMatrix.py //混淆矩阵
|  |  ├─originalDataInfo.py //原数据信息统计
|  |  ├─transCoding.py //编码
├─data //数据
|  ├─credit.csv //原始数据的合并
|  ├─featureEngineering.csv //特征工程之后的数据
|  ├─undersampling.csv //欠采样数据
|  ├─UNDERSAMPLING //欠采样数据的DataFrames直接导出
|  |       ├─X_b.csv
|  |       ├─X_test.csv
|  |       ├─Y_b.csv
|  |       └Y_test.csv
|  ├─SMOTEENN //SMOTE+ENN的DataFrames直接导出
|  |    ├─X_b.csv
|  |    ├─X_test.csv
|  |    ├─Y_b.csv
|  |    └Y_test.csv
|  ├─ORIGIN //原始数据
|  |   ├─application_record.csv
|  |   └credit_record.csv
```
