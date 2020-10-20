# coding = utf-8
import numpy as np
import pandas as pd
from sklearn import preprocessing

# datasets
appdata_file_name = '../data/ORIGIN/application_record.csv'
record_file_name = '../data/ORIGIN/credit_record.csv'


# read and clean up the datasets
def load_datasets():
    # application_record.csv contains appliers personal information.(feature)
    app_data = pd.read_csv(appdata_file_name).drop_duplicates()
    # credit_record.csv records users' behaviors of credit card.
    record_data = pd.read_csv(record_file_name)
    # merge datasets
    credit_data = pd.merge(left=app_data, right=record_data, on='ID')


if __name__ == '__main__':
    load_datasets()
