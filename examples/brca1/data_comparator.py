'''
This script is used for csv field value and its probability
Input --> *.csv
Output --> feature distribution
'''

import utilities as Utils

print("-------------------------------Training--------------------")
df = Utils.read_from_file("dataset\\training_data.csv")
Utils.get_data_col_info(df)

print("--------------------Testing vs Training--------------------")

print("-------------------------------Testing---------------------")
df = Utils.read_from_file("dataset\\test_data.csv")
Utils.get_data_col_info(df)