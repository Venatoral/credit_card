# coding = utf-8
import numpy as np
import pandas as pd

# datasets
record_file_name = '../data/application_record.csv'
credit_file_name = '../data/credit_record.csv'
if __name__ == '__main__':
    # read datasets into DataFrame using pandas
    credit_df = pd.read_csv(credit_file_name)
    record_df = pd.read_csv(record_file_name)
    # show info about dataFrame
    print(credit_df.dropna().info())
    print(record_df.dropna().info())
