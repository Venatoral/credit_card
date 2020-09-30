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
* **NAME_EDUCATION_TYPE : 受教育程度**                                                |
| NAME_EDUCATION_TYPE | 教育程度       |<p>Higher education : 受过高等教育(本科以上)<br/>Secondary / secondary special : 高中/大专<br/>Incomplete higher : 受过高等教育，未毕业(本科辍学)<br/>Lower secondary : 初中<br/>Lower secondary : 初中</p>                                                |
| NAME_FAMILY_STATUS  | 婚姻状态       | <p>Married : 已结婚<br/>Single / not married : 单身/未结婚<br/>Civil marriage : 已订婚<br/>Separated : 离异<br/>Widow : 丧偶</p>                                                |
| NAME_HOUSING_TYPE   | 居住方式       |<p>House / apartment : 住在自己的房屋,公寓<br/>With parents : 和父母一起居住<br/>Municipal apartment : 住在市政公寓<br/>Rented apartment : 租房子住<br/>Office apartment : 办公公寓/员工宿舍</p>                                                |
| DAYS_BIRTH          | 生日           | 0为当日，日期向前计算，比如-28为28天前出生     |
| DAYS_EMPLOYED       | 开始工作日期   | 0为当日，日期向前计算，比如-28为28天前开始工作 |
| FLAG_MOBIL          | 是否有手机     |Y:有 N:无                                                 |
| FLAG_WORK_PHONE     | 是否有工作电话 |Y:有 N:无                                                 |
| FLAG_PHONE          | 是否有电话     |Y:有 N:无                                                 |
| FLAG_EMAIL          | 是否有 email   |Y:有 N:无                                                 |
| OCCUPATION_TYPE     | 职业           |<p>Laborers : 打工者<br/>Core staff : 企业高层员工<br/>Sales staff : 销售业员工<br/>Managers : 经理<br/>Drivers : 司机</p>                                                |
| CNT_FAM_MEMBERS     | 家庭人数       |INT                                                |

## **credit_record.csv**

| 字段名         | 解释     | 备注                                                                                                                                 |
|----------------|----------|--------------------------------------------------------------------------------------------------------------------------------------|
| ID             | 客户号   |                                                                                                                                      |
| MONTHS_BALANCE | 记录月份 | 已抽取数据月份为起点，向前倒退，0为当月，-1为前一个月，依次类推                                                                      |
| STATUS         | 状态     | 0:1-29 天逾期 1:30-59 天逾期 2:60-89 天逾期 3:90-119 天逾期 4:120-149 天逾期 5:150天以上逾期或坏账、核销 C: 当月已还清 X: 当月无借款 |
