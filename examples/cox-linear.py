import pandas as pd
from lifelines import CoxPHFitter
import utilities as Utils




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
        return 1
    else:
        return 0

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
df["End Calc Vital Status (Adjusted)"] = df["End Calc Vital Status (Adjusted)"].apply(encode_event)

df_train = df[100:]
df_test = df[:100]
Utils.drop_column(df, 'Site recode ICD-O-3/WHO 2008_Lung and Bronchus')
Utils.drop_column(df, 'Derived AJCC M, 7th ed (2010-2015)_M0')


cph = CoxPHFitter(penalizer=0.01)
cph.fit(df, duration_col='Number of Intervals (Calculated)', event_col='End Calc Vital Status (Adjusted)',
        show_progress=True, step_size=0.80)
'''
Concordance = 0.64
Partial AIC = 85456.71
log-likelihood ratio test = -5612.76 on 90 df
-log2(p) of ll-ratio test = -0.00
'''
cph.print_summary()