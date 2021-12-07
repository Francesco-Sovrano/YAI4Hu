#!/usr/bin/env python
# coding: utf-8

# # Developing a model for heart disease prediction using PyCaret

# Author: Jason Bentley  
# Date: June 2020  
# 
# **Background**
# 
# In a developed professional data science environment, you might have access to AWS, GCP, Azure or other platforms or software with tools to perform experiment set-up, tracking and logging. But what if you just want to get a pilot up and running fast or you are doing a research project? This is where PyCaret (https://pycaret.org/) shines with the ability to do experimentation quickly and systematically!
# 
# What is PyCaret? Well its creators describe it as an "…open source, low-code machine learning library in Python that allows you to go from preparing your data to deploying your model within seconds in your choice of notebook environment." Anything that lets you spend more time on content and maximizing impact has got to be good, right?
# 
# In the following example we will develop, evaluate and inspect a model for predicting heart disease using PyCaret. As this is my first experience with PyCaret I will also provide some summary thoughts and first impressions at the end. As always in any example application involving health, the associations, inferences and commentary in no way constitute medical advice.

# In[1]:


# Data management
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Cohort tables
from tableone import TableOne

# PyCaret for classification
from pycaret.classification import *


# ### The data and EDA

# In[2]:


# original file (processed.cleveland.data) from https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/
# dataset has some ? values which are unknowns. Column names from https://archive.ics.uci.edu/ml/datasets/heart+Disease
# load the data - we have 13 features and 1 target, and 303 patients in our dataset
hd_df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv",
                    delimiter= ',',
                    # header=None,
                    # names=file_cols,
                    na_values=['?'])

hd_df.shape


# In[3]:


# quick look
hd_df.head()


# For our exploratory data analysis, the features available are:
# 
# 1. age: patient age in years
# 2. sex: patient sex, 0 = Female, 1 = Male
# 3. cp: chest pain type, 1 = typical angina, 2 = atypical angina, 3 = non anginal pain, 4 = asymptomatic
# 4. trestbps: resting blood pressure in mmHg
# 5. chol: serum cholesterol in mg/dl
# 6. fbs: fasting blood sugar > 120 mg/dl, 0 = No, 1 = Yes
# 7. restecg: resting electrocardiographic results, 0 = normal, 1 = ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), 2 = probable or definite left ventricular hypertrophy by Estes' criteria
# 8. thalach: maximum heart rate achieved in beats per minute
# 9. exang: exercise induced angina, 0 = No, 1 = Yes
# 10. oldpeak: ST depression induced by exercise relative to rest, the higher the value the greater the abnormality in the patient electrocardiogram
# 11. slope: the slope of the peak exercise ST segment, 1 = upsloping, 2 = flat, 3 = downsloping
# 12. ca: count of the number of major vessels colored by flourosopy
# 13. thal: measured blood flow to the heart as a result of a thallium stress test, 3 = normal, 6 = fixed defect, 7 = reversable defect
# 
# First we create a summary table as appropriate for these features and for continuous features generate additional visualizations to further help our understanding.

# In[4]:


# lets re-label and re-name just to make our understanding a little easier

# specific
d_sex = {1: 'female', 0: 'male'}
d_cp = {1: 'typical_angina', 2: 'atypical_angina', 3: 'non_anginal_pain', 4: 'asymptomatic'}
d_restecg = {0: 'normal', 1: 'ST_T_abnormality', 2: 'LV_hypertrophy'}
d_slope = {1: 'upsloping', 2: 'flat', 3: 'downsloping'}
d_thal = {3: 'normal', 6: 'fixed_defect', 7: 'reversable_defect'}

# generic
d_no_yes = {0: 'no', 1: 'yes'}
d_yes_no = {1: 'no', 0: 'yes'}

# apply mapping
hd_df['sex'].replace(d_sex, inplace=True)
hd_df['cp'].replace(d_cp, inplace=True)
hd_df['restecg'].replace(d_restecg, inplace=True)
hd_df['slope'].replace(d_slope, inplace=True)
# hd_df['thal'].replace(d_thal, inplace=True)
hd_df['fbs'].replace(d_yes_no, inplace=True)
hd_df['exang'].replace(d_yes_no, inplace=True)

# rename features to also be a bit easier for direct interpretation
# bp = blood pressure, ecg = electrocardiograph, hr = heart rate, bf = blood flow
d_cols = {
'cp': 'chest_pain',
'trestbps': 'resting_bp',
'chol': 'serum_cholesterol',
'fbs': 'high_fasting_blood_sugar',
'restecg': 'resting_ecg',
'thalach': 'maximum_hr',
'exang': 'exercise_induced_angina',
'oldpeak': 'ST_depression_exercise_vs_rest',
'slope': 'peak_exercise_ST_segment_slope',
# 'ca': 'num_affected_major_vessels',
# 'thal': 'thallium_stress_test_bf'
}

# rename selected columns
hd_df.rename(columns=d_cols, inplace=True)

hd_df.head()


# In[5]:


# lets create the target, where the values 1, 2, 3, 4 indicate varying degrees of heart disease
# hd_df['num'].value_counts(normalize=True)


# In[6]:


# Lets create a numeric and labelled target. 46% of the cohort have some form of heart disease. Target reasonably balanced.
hd_df['target'] = np.where(hd_df['target'] == 0, 1, 0)


# In[7]:


# Create a standard "cohort" table as a first look, define columns of interest for Table
features = list(hd_df.columns[0:13])
cat_features = ['sex', 'chest_pain', 'high_fasting_blood_sugar', 'resting_ecg', 'exercise_induced_angina',
                'peak_exercise_ST_segment_slope']
nonnormal_features = ['age', 'resting_bp', 'serum_cholesterol', 'maximum_hr', 'ST_depression_exercise_vs_rest']

cohort_table = TableOne(hd_df,
                        columns=features,
                        categorical=cat_features,
                        groupby='target',
                        nonnormal=nonnormal_features,
                        overall=False)

#print(cohort_table.tabulate(tablefmt = "fancy_grid"))
print(cohort_table)


# Patients with heart disease are more likely to be older, male, have asymptomatic chest pain, higher serum cholesterol, lower maximum heart rate, exercise induced angina, higher ST depression during exercise compared to rest, a flat or downsloping peak exercise ST segment slope, affected major vessels, and a blood flow fixed or reversable defect identified by the thallium stress test.

# In[8]:


# Now lets look at the continuous variables and the target in a bit more detail
# ft_plot = sns.pairplot(hd_df[nonnormal_features + ['target']], hue='target')
# ft_plot.savefig("Fig_2_hd_feature_plot.png")


# A few observations that we will need to consider for our pre-preocessing:  
# 1. A handful of missing values for a continuous (count of vessls identified via fluoroscopy) and a categorical feature (thalium stress test blood flow to the heart) - so we need to decide how we want to handling missing values in our pipeline
# 2. The resting electrocardiographic result of ST-T abnormality has only 4 cases in the entire cohort - this is a rare level which we could combine with another. It also consitutes a feature with low variance
# 3. Some potential outliers, with a patient with no heart disease having a serum cholesterol level above 500, and also two patients in the heart disease group with ST depression values > 5 - how do we want to manange outliers in our pipeline

# ### Experiment set-up

# Some of the following steps I will employ may not be in the best interests of developing a model with optimal performance, but the aim here is to highlight how wonderful the set-up for building models can be with PyCaret.

# In[9]:


# lets do some re-grouping just to avoid small numbers or categorical features with minimal variance
cp_regroup = {'typical_angina': 'anginal_pain', 'atypical_angina': 'anginal_pain', 'non_anginal_pain': 'non_anginal_pain', 'asymptomatic': 'asymptomatic'}
restecg_regroup = {'normal': 'normal', 'ST_T_abnormality': 'not_normal', 'LV_hypertrophy': 'not_normal'}
# slope_regroup = {'upsloping': 'upsloping', 'flat': 'flat', 'downsloping': 'flat_downsloping'}
# thal_regroup = {'normal': 'normal', 'fixed_defect': 'defect', 'reversable_defect': 'defect'}

# apply re-grouping
hd_df['chest_pain'].replace(cp_regroup, inplace=True)
hd_df['resting_ecg'].replace(restecg_regroup, inplace=True)
# hd_df['peak_exercise_ST_segment_slope'].replace(slope_regroup, inplace=True)
# hd_df['thallium_stress_test_bf'].replace(thal_regroup, inplace=True)

# check
print(TableOne(hd_df,
               columns=features,
               categorical=cat_features,
               groupby='target',
               nonnormal=nonnormal_features,
               overall=False))


# In[10]:


# deal with the few outliers by trimming to 99th percentile
# hd_df['serum_cholesterol'].clip(upper=np.percentile(hd_df['serum_cholesterol'], 99), inplace=True)
# hd_df['ST_depression_exercise_vs_rest'].clip(upper=np.percentile(hd_df['ST_depression_exercise_vs_rest'], 99), inplace=True)

# check
sns.pairplot(hd_df[nonnormal_features + ['target']], hue='target')


# In[11]:


# couple of key parameters
prop_train = 0.65
num_folds = 10

# hd_df['chest_pain_anginal_pain'] = np.where(hd_df['chest_pain'] == 'anginal_pain', 'yes', 'no')
# hd_df['chest_pain_non_anginal_pain'] = np.where(hd_df['chest_pain'] == 'non_anginal_pain', 'yes', 'no')

import imblearn
# experiment set-up
exp1 = setup(
    hd_df,
    train_size=prop_train,
    target='target',
    session_id=101,
    # ignore_features=['num', 'target_c'],
    categorical_features=cat_features,
    numeric_imputation = 'median',
    categorical_imputation = 'mode',
    # fix_imbalance=True,
    # fix_imbalance_method=imblearn.under_sampling.TomekLinks(),
)

# lets look at our candidate best model
selected_model = create_model('xgboost', fold=num_folds, verbose=False)


# In[17]:


# can check the defaults used
plot_model(selected_model, plot = 'parameter')


# ### Tuning

# From the classification.py function we can see the parameter grid used via RandomizedSearchCV
# 
# ```
# param_grid = {'learning_rate': np.arange(0,1,0.01),
#               'n_estimators':[10, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 
#               'subsample': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1],
#               'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)], 
#               'colsample_bytree': [0.5, 0.7, 0.9, 1],
#               'min_child_weight': [1, 2, 3, 4]
# }
# ```

# In[18]:


# this is an example of using tune_model with defaults
tuned_model = tune_model(selected_model, fold=num_folds)


# The tuning improved things slightly with all metrics improving ever so slightly. If you look under the hood a randomized grid-search with 10 iterations is being used. Of course, many more iterations could be used or even a different approach such as hyperopt. Additionaly, at this stage if we were interested in using the predicted probability we could also do some post-processing calibration. PyCaret has a great calibrate_model() function for this. For now we will put this aside. Typically I prefer non-parametric calibration (istonic to sigmoid), however, isotonic can be prone to overfit unless you have a large sample size, which we do not here.
# 
# Note if we had wanted to perform calibration of the predcited probabilities from the tuned model we could have used:  
# `calibrated_model = calibrate_model(tuned_model, fold=num_folds)`  
# `plot_model(calibrated_model, plot='calibration')`

# In[19]:


# we can also take a peak at the tuned model object - not too much change given the samll amount of randomized search
plot_model(tuned_model, plot = 'parameter')

# ### Evaluating the model
# Briefly evaluate model provides an interactive cell output with many common useful plots and outputs - some of which will depend upon the model you are looking at. However, a great many are common to all models and it is always helpful to keep in mind what it is that is important to gleen from each output.  
# 
# Please note some options require additional processing when selected and so you may have to wait a bit depending upon how complex the task is that generates the plot. Currently, I don't believe results are cached so when you select another and go back it will require the same amount of wait time to produce again.

# In[26]:


# using evaluate model provides a summary of many outputs that can be obtained individually via plot_model()
evaluate_model(tuned_model)

# finalize the model (i.e., re-train using all available data)
hd_model = finalize_model(tuned_model)

# save the final trained model for use in our application and also save the experiment as well
save_model(hd_model, 'heart_disease_prediction_model_Jul2020')
