import inspect
import os.path

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder


def read_from_file(filepath):
    if os.path.isfile(filepath) is False:
        raise ValueError(form_error_msg("Invalid parameter filepath."))
    if filepath.endswith(".csv") is False:
        raise ValueError(form_error_msg("Invalid file extension."))
    return pd.read_csv(filepath)


def drop_column(data, column):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    if column is False:
        raise ValueError(form_error_msg("Invalid parameter col_list."))
    del data[column]


def sum_given_columns(data, columns):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    if is_valid_list(columns) is False:
        raise ValueError(form_error_msg("Invalid parameter cols."))
    count = 0
    for col in columns:
        count += data[col]
    return count.to_numpy()


def get_match_prefix_pattern_columns(data, column_prefix):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    names_list = get_data_frame_col_names_list(data)
    if is_valid_list(names_list) is False:
        raise ValueError(form_error_msg("Invalid inside parameter names_list."))
    matched_list = []
    for name in names_list:
        if name is False:
            continue
        if name.startswith(column_prefix):
            matched_list.append(name)
    return matched_list


def sum_prefix_pattern_columns(data, column_prefix):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    matched_list = get_match_prefix_pattern_columns(data, column_prefix)
    return sum_given_columns(data, matched_list)


def split_data(data, fraction):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    if fraction <= 0 or fraction >= 1:
        raise ValueError(form_error_msg("Invalid parameter test_proportion."))
    return data.sample(frac=fraction)


def remove_data(data, indexs):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    data.drop(indexs)


def select_data_from_values(data, column, value_list):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    if column is False:
        raise ValueError(form_error_msg("Invalid parameter column."))
    if column not in data.columns:
        raise ValueError(form_error_msg("data doesn't contains parameter column."))
    if is_valid_list(value_list) is False:
        raise ValueError(form_error_msg("Invalid parameter value_list."))
    return data.loc[data[column].isin(value_list)]


def map_one_col_data(data, column, map_func):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    if column is False:
        raise ValueError(form_error_msg("Invalid parameter column."))
    if column not in data.columns:
        raise ValueError(form_error_msg("data doesn't contains parameter column."))
    if callable(map_func) is False:
        raise ValueError(form_error_msg("Invalid parameter map_func."))
    data[column] = data[column].map(map_func)
    return data


def get_function_name():
    currentframe = inspect.currentframe()
    return inspect.getframeinfo(currentframe).function


def form_error_msg(error_msg):
    return get_function_name() + ":" + error_msg


def get_data_frame_row_count(data_frame):
    if is_data_frame(data_frame) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    return data_frame.shape[0]


def is_col_exist(data, column):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    if column is False:
        raise ValueError(form_error_msg("Invalid parameter column."))
    return  column in data.columns


def get_data_frame_col_count(data_frame):
    if is_data_frame(data_frame) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    if len(data_frame) == 1:
        return 1
    return data_frame.shape[1]


def get_data_frame_col_names(data_frame):
    if is_data_frame(data_frame) is False:
        raise ValueError(form_error_msg("Invalid parameter data_frame."))
    return data_frame.columns


def get_data_frame_col_names_list(data_frame):
    if is_data_frame(data_frame) is False:
        raise ValueError(form_error_msg("Invalid parameter data_frame."))
    return list(data_frame.columns.values)


def get_col_values(data, column):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    if column is False:
        raise ValueError(form_error_msg("Invalid parameter column."))
    return set(data[column].unique())

def get_col_values_range(data, column):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    if column is False:
        raise ValueError(form_error_msg("Invalid parameter column."))
    return data[column].unique()

def get_data_col_info(data):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    col_list = get_data_frame_col_names_list(data)
    for col in col_list:
        print(col)
        merge_print_enum_two_series(data[col].value_counts(normalize=False),data[col].value_counts(normalize=True)*100)



def get_data_all_col_percent(data):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    col_list = get_data_frame_col_names_list(data)
    for col in col_list:
        print("-------get_col_item_occurrence-------")
        print(col)
        get_col_item_percent(data,col)

def get_col_item_percent(data,column):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    if column is False:
        raise ValueError(form_error_msg("Invalid parameter column."))
    if is_col_exist(data,column) is False:
        raise ValueError(form_error_msg("Invalid parameter column."))
    return print_enum_series(data[column].value_counts(normalize=True)*100)

def get_data_all_col_occurence(data):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    col_list = get_data_frame_col_names_list(data)
    for col in col_list:
        print("-------get_col_item_occurrence-------")
        print(col)
        get_col_item_occurrence(data,col)


def get_col_item_occurrence(data,column):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    if column is False:
        raise ValueError(form_error_msg("Invalid parameter column."))
    if is_col_exist(data,column) is False:
        raise ValueError(form_error_msg("Invalid parameter column."))
    return print_enum_series(data[column].value_counts(normalize=False)*100)


def merge_print_enum_two_series(seriesA, seriesB):
    if is_series(seriesA) is False or is_series(seriesB) is False:
        raise ValueError(form_error_msg("Invalid parameter series."))
    if seriesA.size != seriesB.size:
        raise ValueError(form_error_msg("incompatible series."))
    dictA = seriesA.to_dict()
    dictB = seriesB.to_dict()
    for key in dictA:
        print('index: ', key, 'number: ', dictA[key], 'percentage: ', dictB[key])



def print_enum_series(series):
    if is_series(series) is False:
        raise ValueError(form_error_msg("Invalid parameter series."))
    for i, v in series.items():
        print('index: ', i, 'value: ', v)

def filter_col_data(data, cols_array):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    if is_valid_list(cols_array) is False:
        raise ValueError(form_error_msg("Invalid parameter cols_array."))
    return data.loc[:, cols_array]


def is_data_frame(data):
    return isinstance(data, pd.DataFrame)


def is_series(data):
    return isinstance(data, pd.Series)


def is_integer(var):
    return isinstance(var, int)


def is_valid_list(array):
    return isinstance(array, list) and array


def one_hot_encode_cols(data, cols):
    if is_data_frame(data) is False:
        raise ValueError(form_error_msg("Invalid parameter data."))
    if is_valid_list(cols) is False:
        raise ValueError(form_error_msg("Invalid parameter cols."))
    transformer = make_column_transformer((OneHotEncoder(), cols), remainder="passthrough")
    return transformer.fit_transform(data)


def print_series_info(series):
    if is_series(series) is False:
        raise ValueError(form_error_msg("Invalid parameter series."))
    print(sorted(series.unique()))


def print_data_frame_info(df):
    if is_data_frame(df) is False:
        raise ValueError(form_error_msg("Invalid parameter df."))
    for col in df:
        print("----------" + col + "----------")
        print_series_info(df[col])
