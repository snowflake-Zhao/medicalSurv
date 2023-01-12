import configparser
import json

import utilities as Utils
import os.path

parser = configparser.ConfigParser()
parser.read("..\database\SEER\\attributeSchema.ini")
sections = parser.sections()

# get all column ranges
training_df = Utils.read_from_file("dataset\\training_data.csv")
testing_df = Utils.read_from_file("dataset\\test_data.csv")
df = training_df.append(testing_df)
col_names_list = Utils.get_data_frame_col_names_list(df)

json_file = "dataset\\api.json"
if (os.path.isfile(json_file)):
    os.remove(json_file)
schema = {}
schema_list = []
if Utils.is_valid_list(col_names_list) and len(col_names_list) > 0:
    for i in range(len(col_names_list)):
        col = col_names_list[i]
        if col not in sections:
            continue
        for key in parser[col]:
            key_schema = {}
            if key == "type":
                if parser[col]["type"] == "select":
                    values_range = Utils.get_col_values_range(df, col)
                    key_schema["name"] = col
                    key_schema["type"] = "select"
                    key_schema["range"] = values_range.tolist()
                    schema_list.append(key_schema)
                elif parser[col]["type"] == "input":
                    max_val = int(max(Utils.get_col_values_range(df, col)))
                    min_val = int(min(Utils.get_col_values_range(df, col)))
                    key_schema["name"] = col
                    key_schema["type"] = "input"
                    key_schema["range"] = [min_val,max_val]
                    schema_list.append(key_schema)
    schema["schema"] = schema_list
    with open(json_file, 'w') as f:
        json.dump(schema, f)

else:
    Utils.form_error_msg("invalid col_names_list")
