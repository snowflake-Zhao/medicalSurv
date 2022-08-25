#!/usr/bin/env python
# coding: utf-8

# # Cox-PH and DeepSurv
#
# In this notebook we will train the [Cox-PH method](http://jmlr.org/papers/volume20/18-424/18-424.pdf), also known as [DeepSurv](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1).
# We will use the METABRIC data sets as an example
#
# A more detailed introduction to the `pycox` package can be found in [this notebook](https://nbviewer.jupyter.org/github/havakv/pycox/blob/master/examples/01_introduction.ipynb) about the `LogisticHazard` method.
#
# The main benefit Cox-CC (and the other Cox methods) has over Logistic-Hazard is that it is a continuous-time method, meaning we do not need to discretize the time scale.

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchtuples as tt

from pycox.evaluation import EvalSurv
from pycox.models import CoxPH

# In[2]:


## Uncomment to install `sklearn-pandas`
# ! pip install sklearn-pandas


# In[3]:


np.random.seed(1234)
_ = torch.manual_seed(123)

# ## Dataset
#
# We load the METABRIC data set and split in train, test and validation.

# In[4]:

# Step 1 data collection
df_train = pd.read_csv("G:\Project\pycox-master\pycox\datasets\dataset\\training_data.csv")
df_test = pd.read_csv("G:\Project\pycox-master\pycox\datasets\dataset\\testing_data.csv")

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
df_train = df[100:]
df_test = df[:100]

df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)



# In[5]:


df_train.head()

# ## Feature transforms
# We have 9 covariates, in addition to the durations and event indicators.
#
# We will standardize the 5 numerical covariates, and leave the binary variables as is. As variables needs to be of type `'float32'`, as this is required by pytorch.

# In[6]:


# cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
# cols_leave = ['x4', 'x5', 'x6', 'x7']
#
# standardize = [([col], StandardScaler()) for col in cols_standardize]
# leave = [(col, None) for col in cols_leave]
#
# x_mapper = DataFrameMapper()

# In[7]:



# We need no label transforms

# In[8]:


def encode_event(df):
    if(df == "Dead"):
        return 1
    else:
        return 0

df_test["End Calc Vital Status (Adjusted)"] = df_test["End Calc Vital Status (Adjusted)"].apply(encode_event)
df_train["End Calc Vital Status (Adjusted)"] = df_train["End Calc Vital Status (Adjusted)"].apply(encode_event)
df_val["End Calc Vital Status (Adjusted)"] = df_val["End Calc Vital Status (Adjusted)"].apply(encode_event)
get_target = lambda df: (df['Number of Intervals (Calculated)'].values, df['End Calc Vital Status (Adjusted)'].values)
y_train = get_target(df_train)
y_val = get_target(df_val)
durations_test, events_test = get_target(df_test)
x_train = df_train.drop("Number of Intervals (Calculated)",axis=1).drop("End Calc Vital Status (Adjusted)",axis=1)
x_val = df_val.drop("Number of Intervals (Calculated)",axis=1).drop("End Calc Vital Status (Adjusted)",axis=1)
x_test = df_test.drop("Number of Intervals (Calculated)",axis=1).drop("End Calc Vital Status (Adjusted)",axis=1)

val = x_val, y_val

# ## Neural net
#
# We create a simple MLP with two hidden layers, ReLU activations, batch norm and dropout.
# Here, we just use the `torchtuples.practical.MLPVanilla` net to do this.
#
# Note that we set `out_features` to 1, and that we have not `output_bias`.

# In[9]:


in_features = x_train.shape[1]
num_nodes = [32, 32]
out_features = 1
batch_norm = True
dropout = 0.661
output_bias = False

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                              dropout, output_bias=output_bias)

# ## Training the model
#
# To train the model we need to define an optimizer. You can choose any `torch.optim` optimizer, but here we instead use one from `tt.optim` as it has some added functionality.
# We use the `Adam` optimizer, but instead of choosing a learning rate, we will use the scheme proposed by [Smith 2017](https://arxiv.org/pdf/1506.01186.pdf) to find a suitable learning rate with `model.lr_finder`. See [this post](https://towardsdatascience.com/finding-good-learning-rate-and-the-one-cycle-policy-7159fe1db5d6) for an explanation.

# In[10]:


model = CoxPH(net, tt.optim.Adam)

# In[11]:


batch_size = 256
lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=10)
_ = lrfinder.plot()

# In[12]:


lrfinder.get_best_lr()

# Often, this learning rate is a little high, so we instead set it manually to 0.01

# In[13]:


model.optimizer.set_lr(0.01)

# We include the `EarlyStopping` callback to stop training when the validation loss stops improving. After training, this callback will also load the best performing model in terms of validation loss.

# In[14]:


epochs = 512
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True

# In[15]:

log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                val_data=val, val_batch_size=batch_size)

# In[16]:


_ = log.plot()

# We can get the partial log-likelihood

# In[17]:


# model.partial_log_likelihood(*val).mean()

# ## Prediction
#
# For evaluation we first need to obtain survival estimates for the test set.
# This can be done with `model.predict_surv` which returns an array of survival estimates, or with `model.predict_surv_df` which returns the survival estimates as a dataframe.
#
# However, as `CoxPH` is semi-parametric, we first need to get the non-parametric baseline hazard estimates with `compute_baseline_hazards`.
#
# Note that for large datasets the `sample` argument can be used to estimate the baseline hazard on a subset.

# In[18]:


_ = model.compute_baseline_hazards()

# In[19]:


surv = model.predict_surv_df(x_test)

# In[20]:


surv.iloc[:, :5].plot()
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

# ## Evaluation
#
# We can use the `EvalSurv` class for evaluation the concordance, brier score and binomial log-likelihood. Setting `censor_surv='km'` means that we estimate the censoring distribution by Kaplan-Meier on the test set.

# In[21]:


ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')

# In[22]:


print(ev.concordance_td())

# In[23]:


time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
_ = ev.brier_score(time_grid).plot()

# In[24]:


ev.integrated_brier_score(time_grid)

# In[25]:


ev.integrated_nbll(time_grid)

# In[ ]:
