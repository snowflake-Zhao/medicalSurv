import pandas as pd
from lifelines import CoxPHFitter



df = Utils.read_from_file("data/breast.csv")
df = Utils.filter_col_data(df, ["Age recode with <1 year olds", "Marital status at diagnosis", "Grade (thru 2017)",
                                "ICD-O-3 Hist/behav",
                                "Breast - Adjusted AJCC 6th T (1988-2015)", "Breast - Adjusted AJCC 6th N (1988-2015)",
                                "Breast - Adjusted AJCC 6th M (1988-2015)", "CS Tumor Size/Ext Eval (2004-2015)",
                                "CS Reg Node Eval (2004-2015)", "CS Mets Eval (2004-2015)",
                                "Laterality", "Breast Subtype (2010+)",
                                "RX Summ--Surg Prim Site (1998+)", "Radiation recode",
                                "Chemotherapy recode (yes, no/unk)",
                                "End Calc Vital Status (Adjusted)", "Number of Intervals (Calculated)"])

# take a look of the data info
Utils.print_data_frame_info(df)

# according to https://seer.cancer.gov/icd-o-3/sitetype.icdo3.20220429.pdf
duct_carcinoma_array = ['8500/3: Infiltrating duct carcinoma, NOS', '8501/3: Comedocarcinoma, NOS',
                        '8502/3: Secretory carcinoma of breast',
                        '8503/3: Intraductal papillary adenocarcinoma with invasion',
                        '8504/3: Intracystic carcinoma, NOS', '8507/3: Ductal carcinoma, micropapillary']
# according to https://seer.cancer.gov/icd-o-3/sitetype.icdo3.20220429.pdf
lobular_and_other_ductal_array = ['8520/3: Lobular carcinoma, NOS', '8521/3: Infiltrating ductular carcinoma',
                                  '8522/3: Infiltrating duct and lobular carcinoma',
                                  '8523/3: Infiltrating duct mixed with other types of carcinoma',
                                  '8524/3: Infiltrating lobular mixed with other types of carcinoma',
                                  '8525/3: Polymorphous low grade adenocarcinoma']
duct_lobular_array = duct_carcinoma_array + lobular_and_other_ductal_array

# filter the ICD-O-3 Hist/behav whose type is DUCT CARCINOM and LOBULAR AND OTHER DUCTAL CA
df = Utils.select_data_from_values(df, "ICD-O-3 Hist/behav", duct_lobular_array)

# map "RX Summ--Surg Prim Site (1998+)" according to map_breast_surg_type
df = Utils.map_one_col_data(df, "RX Summ--Surg Prim Site (1998+)", br_utils.map_breast_surg_type)

# map "End Calc Vital Status (Adjusted)" according to map_event_code
df = Utils.map_one_col_data(df, "End Calc Vital Status (Adjusted)", br_utils.map_event_code)

# take a look of the data info again
print("------------------After filtering and Mapping------------------")
Utils.print_data_frame_info(df)
df = pd.get_dummies(df, prefix=["Age recode with <1 year olds", "Marital status at diagnosis", "Grade (thru 2017)",
                                "ICD-O-3 Hist/behav",
                                "Breast - Adjusted AJCC 6th T (1988-2015)",
                                "Breast - Adjusted AJCC 6th N (1988-2015)",
                                "Breast - Adjusted AJCC 6th M (1988-2015)", "CS Tumor Size/Ext Eval (2004-2015)",
                                "CS Reg Node Eval (2004-2015)", "CS Mets Eval (2004-2015)",
                                "Laterality", "Breast Subtype (2010+)",
                                "RX Summ--Surg Prim Site (1998+)", "Radiation recode",
                                "Chemotherapy recode (yes, no/unk)"],
                    columns=["Age recode with <1 year olds", "Marital status at diagnosis", "Grade (thru 2017)",
                             "ICD-O-3 Hist/behav",
                             "Breast - Adjusted AJCC 6th T (1988-2015)",
                             "Breast - Adjusted AJCC 6th N (1988-2015)",
                             "Breast - Adjusted AJCC 6th M (1988-2015)", "CS Tumor Size/Ext Eval (2004-2015)",
                             "CS Reg Node Eval (2004-2015)", "CS Mets Eval (2004-2015)",
                             "Laterality", "Breast Subtype (2010+)",
                             "RX Summ--Surg Prim Site (1998+)", "Radiation recode",
                             "Chemotherapy recode (yes, no/unk)"])
data = df

print("-----------Training Data-----------")
print("-----------The row number-----------")
print(Utils.get_data_frame_row_count(data))
print("-----------The col number-----------")
print(Utils.get_data_frame_col_count(data))
print("-----------The column names are-----------")
print(Utils.get_data_frame_col_names(data))
print("-----------The null value summary-----------")
print(data.isnull().sum())

# ConvergenceWarning: Column(s) ['ICD-O-3 Hist/behav_8525/3: Polymorphous low grade adenocarcinoma', 'Breast - Adjusted AJCC 6th M (1988-2015)_M0', 'CS Mets Eval (2004-2015)_6', 'Laterality_Bilateral, single primary', 'Laterality_Only one side - side unspecified', 'Laterality_Paired site, but no information concerning laterality', 'RX Summ--Surg Prim Site (1998+)_Local tumor destruction'] have very low variance. This may harm convergence. 1) Are you using formula's? Did you mean to add '-1' to the end. 2) Try dropping this redundant column before fitting if convergence fails.
# Utils.drop_column(training_data, 'ICD-O-3 Hist/behav_8525/3: Polymorphous low grade adenocarcinoma')
Utils.drop_column(data, 'Breast - Adjusted AJCC 6th M (1988-2015)_M0')
# Utils.drop_column(training_data, 'Laterality_Bilateral, single primary')
# Utils.drop_column(training_data, 'CS Mets Eval (2004-2015)_6')
# Utils.drop_column(training_data, 'Laterality_Only one side - side unspecified')
# Utils.drop_column(training_data, 'Laterality_Paired site, but no information concerning laterality')
# Utils.drop_column(training_data,'RX Summ--Surg Prim Site (1998+)_Local tumor destruction')

cph = CoxPHFitter(penalizer=0.004, l1_ratio=0.005)
cph.fit(data, duration_col='Number of Intervals (Calculated)', event_col='End Calc Vital Status (Adjusted)',
        show_progress=True, step_size=0.96)
'''
Concordance = 0.64
Partial AIC = 85456.71
log-likelihood ratio test = -5612.76 on 90 df
-log2(p) of ll-ratio test = -0.00
'''
cph.print_summary()