import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sksurv.datasets import load_gbsg2
from sksurv.ensemble import RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder



df_train = pd.read_csv("G:\Project\medicalSurv\pycox\datasets\dataset\\training_data.csv")
df_test = pd.read_csv("G:\Project\medicalSurv\pycox\datasets\dataset\\testing_data.csv",cache_dates=False)

df_train = df_train.loc[df_train["RX Summ--Surg Prim Site (1998+)"].isin([33,56])]

def adjust_tumor_size(df):
    return df/10

def age_recode(df):
    if df >= 15 and df<=19:
        return "15-19 years"
    elif df >= 20 and df<=24:
        return "20-24 years"
    elif df >= 25 and df<=29:
        return "25-29 years"
    elif df >= 30 and df<=34:
        return "30-34 years"
    elif df >= 35 and df<=39:
        return "35-39 years"
    elif df >= 40 and df<=44:
        return "40-44 years"
    elif df >= 45 and df<=49:
        return "45-49 years"
    elif df >= 50 and df<=54:
        return "50-54 years"
    elif df >= 55 and df<=59:
        return "55-59 years"
    elif df >= 60 and df<=64:
        return "60-64 years"
    elif df >= 65 and df<=69:
        return "65-69 years"
    elif df >= 70 and df<=74:
        return "70-74 years"
    elif df >= 75 and df<=79:
        return "75-79 years"
    elif df >= 80 and df<=84:
        return "80-84 years"
    elif df >=85:
        return "85+ years"
    else:
        raise ValueError("Invalid parameter map_func.")

df_test["Age recode"] = df_test["Age"].apply(age_recode)
df_train["CS tumor size (2004-2015)"] = df_train["CS tumor size (2004-2015)"].apply(adjust_tumor_size)
df_test = df_test.drop("Age",axis=1)

df = df_test.append(df_train)

def encode_event(df):
    if(df == "Dead"):
        return True
    else:
        return False

df = pd.get_dummies(df, prefix=["Site recode ICD-O-3/WHO 2008", "RX Summ--Surg Prim Site (1998+)", "Radiation recode",
                                "Chemotherapy recode (yes, no/unk)",
                                "Derived AJCC T, 7th ed (2010-2015)",
                                "Derived AJCC N, 7th ed (2010-2015)",
                                "Derived AJCC M, 7th ed (2010-2015)", "ICD-O-3 Hist/behav",
                                "Age recode", "Sex",
                                "Laterality"],
                    columns=["Site recode ICD-O-3/WHO 2008", "RX Summ--Surg Prim Site (1998+)", "Radiation recode",
                             "Chemotherapy recode (yes, no/unk)",
                             "Derived AJCC T, 7th ed (2010-2015)",
                             "Derived AJCC N, 7th ed (2010-2015)",
                             "Derived AJCC M, 7th ed (2010-2015)", "ICD-O-3 Hist/behav",
                             "Age recode", "Sex",
                             "Laterality"])
df["End Calc Vital Status (Adjusted)"] = df_test["End Calc Vital Status (Adjusted)"].apply(encode_event)

df_train = df[100:]
df_test = df[:100]

df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

get_target = lambda df: (df['Number of Intervals (Calculated)'].values.astype(np.float16), df['End Calc Vital Status (Adjusted)'].values.astype(bool))
zip_arrays = lambda arr1,arr2:list(zip(arr1,arr2))
def list_to_nparr(arr):
    out = np.empty(len(arr), dtype=[('cens', '?'), ('time', '<f8')])
    out[:] = arr
    return out



durations_train, events_train = get_target(df_train)
durations_val, events_val = get_target(df_val)
durations_test, events_test = get_target(df_test)

y_train = list_to_nparr(zip_arrays(events_train,durations_train))
y_val = list_to_nparr(zip_arrays(events_val,durations_val))
y_test = list_to_nparr(zip_arrays(events_test,durations_test))

x_train = df_train.drop("Number of Intervals (Calculated)",axis=1).drop("End Calc Vital Status (Adjusted)",axis=1)
x_val = df_val.drop("Number of Intervals (Calculated)",axis=1).drop("End Calc Vital Status (Adjusted)",axis=1)
x_test = df_test.drop("Number of Intervals (Calculated)",axis=1).drop("End Calc Vital Status (Adjusted)",axis=1)


random_state = 20

X_train, X_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size=0.25, random_state=random_state)

rsf = RandomSurvivalForest(n_estimators=959,
                           min_samples_split=10,
                           min_samples_leaf=15,
                           n_jobs=-1,
                           random_state=random_state)
rsf.fit(X_train, y_train)
ci_score = rsf.score(X_test, y_test)
print(ci_score)
