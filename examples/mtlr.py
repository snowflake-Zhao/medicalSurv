#!/usr/bin/env python
# coding: utf-8

# # MTLR (N-MTLR) Example 
# 
# In this notebook, we will present a simple example of the neural network version of the `MTLR` method described in 
# [this](https://papers.nips.cc/paper/4210-learning-patient-specific-cancer-survival-distributions-as-a-sequence-of-dependent-regressors) and [this paper](https://arxiv.org/pdf/1801.05512.pdf).
# 
# For a more verbose introduction to `pycox` see [this notebook](https://nbviewer.jupyter.org/github/havakv/pycox/blob/master/examples/01_introduction.ipynb).

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 

import torch # For building the networks 
import torchtuples as tt # Some useful functions

from pycox.datasets import metabric
from pycox.models import MTLR
from pycox.evaluation import EvalSurv


# In[2]:


## Uncomment to install `sklearn-pandas`
# ! pip install sklearn-pandas


# In[3]:


np.random.seed(1234)
_ = torch.manual_seed(123)


# ## Dataset
# 
# We load the METABRIC data set as a pandas DataFrame and split the data in in train, test and validation.
# 
# The `duration` column gives the observed times and the `event` column contains indicators of whether the observation is an event (1) or a censored observation (0).

# In[4]:


df_train = metabric.read_df()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)


# In[5]:


df_train.head()


# ## Feature transforms
# 
# The METABRIC dataset has  9 covariates: `x0, ..., x8`.
# We will standardize the 5 numerical covariates, and leave the binary covariates as is.
# Note that PyTorch require variables of type `'float32'`.
# 
# We like using the `sklearn_pandas.DataFrameMapper` to make feature mappers.

# In[6]:


cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)


# In[7]:


x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')


# ## Label transforms
# 
# The survival methods require individual label transforms, so we have included a proposed `label_transform` for each method.
# In this case `label_transform` is just a shorthand for the class `pycox.preprocessing.label_transforms.LabTransDiscreteTime`.
# 
# The `MTLR` is a discrete-time method, meaning it requires discretization of the event times to be applied to continuous-time data.
# We let `num_durations` define the size of this (equidistant) discretization grid, meaning our network will have `num_durations` output nodes.

# In[8]:


num_durations = 10
labtrans = MTLR.label_transform(num_durations)
get_target = lambda df: (df['duration'].values, df['event'].values)
y_train = labtrans.fit_transform(*get_target(df_train))
y_val = labtrans.transform(*get_target(df_val))

train = (x_train, y_train)
val = (x_val, y_val)

# We don't need to transform the test labels
durations_test, events_test = get_target(df_test)


# In[9]:


type(labtrans)


# ## Neural net
# 
# We make a neural net with `torch`.
# For simple network structures, we can use the `MLPVanilla` provided by `torchtuples`.
# For building more advanced network architectures, see for example [the tutorials by PyTroch](https://pytorch.org/tutorials/).
# 
# The following net is an MLP with two hidden layers (with 32 nodes each), ReLU activations, and `num_nodes` output nodes.
# We also have batch normalization and dropout between the layers.

# In[10]:


in_features = x_train.shape[1]
num_nodes = [32, 32]
out_features = labtrans.out_features
batch_norm = True
dropout = 0.1

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)


# If you instead want to build this network with `torch` you can uncomment the following code.
# It is essentially equivalent to the `MLPVanilla`, but without the `torch.nn.init.kaiming_normal_` weight initialization.

# In[11]:


# net = torch.nn.Sequential(
#     torch.nn.Linear(in_features, 32),
#     torch.nn.ReLU(),
#     torch.nn.BatchNorm1d(32),
#     torch.nn.Dropout(0.1),
    
#     torch.nn.Linear(32, 32),
#     torch.nn.ReLU(),
#     torch.nn.BatchNorm1d(32),
#     torch.nn.Dropout(0.1),
    
#     torch.nn.Linear(32, out_features)
# )


# ## Training the model
# 
# To train the model we need to define an optimizer. You can choose any `torch.optim` optimizer, but here we instead use one from `tt.optim` as it has some added functionality.
# We use the `Adam` optimizer, but instead of choosing a learning rate, we will use the scheme proposed by [Smith 2017](https://arxiv.org/pdf/1506.01186.pdf) to find a suitable learning rate with `model.lr_finder`. See [this post](https://towardsdatascience.com/finding-good-learning-rate-and-the-one-cycle-policy-7159fe1db5d6) for an explanation.
# 
# We also set `duration_index` which connects the output nodes of the network the the discretization times. This is only useful for prediction and does not affect the training procedure.

# In[12]:


model = MTLR(net, tt.optim.Adam, duration_index=labtrans.cuts)


# In[13]:


batch_size = 256
lr_finder = model.lr_finder(x_train, y_train, batch_size, tolerance=6)
_ = lr_finder.plot()


# In[14]:


lr_finder.get_best_lr()


# Often, this learning rate is a little high, so we instead set it manually to 0.01

# In[15]:


model.optimizer.set_lr(0.01)


# We include the `EarlyStopping` callback to stop training when the validation loss stops improving. After training, this callback will also load the best performing model in terms of validation loss.

# In[16]:


epochs = 100
callbacks = [tt.callbacks.EarlyStopping()]
log = model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=val)


# In[17]:


_ = log.plot()


# ## Prediction
# 
# For evaluation we first need to obtain survival estimates for the test set.
# This can be done with `model.predict_surv` which returns an array of survival estimates, or with `model.predict_surv_df` which returns the survival estimates as a dataframe.

# In[18]:


surv = model.predict_surv_df(x_test)


# We can plot the survival estimates for the first 5 individuals.
# Note that the time scale is correct because we have set `model.duration_index` to be the grid points.
# We have, however, only defined the survival estimates at the 10 times in our discretization grid, so, the survival estimates is a step function

# In[19]:


surv.iloc[:, :5].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')


# It is, therefore, often beneficial to interpolate the survival estimates, see [this paper](https://arxiv.org/abs/1910.06724) for a discussion.
# Linear interpolation (constant density interpolation) can be performed with the `interpolate` method. We also need to choose how many points we want to replace each grid point with. Her we will use 10.

# In[20]:


surv = model.interpolate(10).predict_surv_df(x_test)


# In[21]:


surv.iloc[:, :5].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')


# ## Evaluation
# 
# The `EvalSurv` class contains some useful evaluation criteria for time-to-event prediction.
# We set `censor_surv = 'km'` to state that we want to use Kaplan-Meier for estimating the censoring distribution.
# 

# In[22]:


ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')


# #### Concordance
# 
# We start with the event-time concordance by [Antolini et al. 2005](https://onlinelibrary.wiley.com/doi/10.1002/sim.2427).

# In[23]:


ev.concordance_td('antolini')


# #### Brier Score
# 
# We can plot the the [IPCW Brier score](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0258%2819990915/30%2918%3A17/18%3C2529%3A%3AAID-SIM274%3E3.0.CO%3B2-5) for a given set of times.
# Here we just use 100 time-points between the min and max duration in the test set.
# Note that the score becomes unstable for the highest times. It is therefore common to disregard the rightmost part of the graph.

# In[24]:


time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
ev.brier_score(time_grid).plot()
plt.ylabel('Brier score')
_ = plt.xlabel('Time')


# #### Negative binomial log-likelihood
# 
# In a similar manner, we can plot the the [IPCW negative binomial log-likelihood](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0258%2819990915/30%2918%3A17/18%3C2529%3A%3AAID-SIM274%3E3.0.CO%3B2-5).

# In[25]:


ev.nbll(time_grid).plot()
plt.ylabel('NBLL')
_ = plt.xlabel('Time')


# #### Integrated scores
# 
# The two time-dependent scores above can be integrated over time to produce a single score [Graf et al. 1999](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0258%2819990915/30%2918%3A17/18%3C2529%3A%3AAID-SIM274%3E3.0.CO%3B2-5). In practice this is done by numerical integration over a defined `time_grid`.

# In[26]:


ev.integrated_brier_score(time_grid) 


# In[27]:


ev.integrated_nbll(time_grid) 


# In[ ]:




