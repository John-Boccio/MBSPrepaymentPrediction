import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
os.chdir('/Users/chongguo/Documents/Stanford/AI_Certificate/CS229/project/')
import sys
sys.path.insert(0, '/Users/chongguo/Documents/Stanford/AI_Certificate/CS229/project/code')
from StandardLoanLevelDatasetParser import StandardLoanLevelDatasetParser

def model_performance(model, train, test):
    model.fit(train.loc[:, train.columns != 'prepay'], train['prepay'])
    score_train = model.score(train.loc[:, train.columns != 'prepay'], train['prepay'])
    score_test = model.score(test.loc[:, test.columns != 'prepay'], test['prepay'])
    preds = model.predict(test.loc[:, test.columns != 'prepay'])
    confusion_mat = confusion_matrix(test['prepay'], preds)
    report = classification_report(test['prepay'], preds)
    return score_train, score_test, preds, confusion_mat, report

def plot_confusion_matrix(matrix, path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(matrix)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i, j], ha='center', va='center', color='red')
    plt.show()
    fig.savefig(path)

est_data = pd.read_csv('data/est_data.csv', encoding='utf-8')
categorical_input = ['occupancy_status', 'property_state', 'loan_purpose', 'month']
numerical_input = [
    'credit_score', 'first_time_homebuyer_flag', 'MI_%', 'number_of_units', 'orig_CLTV', 
    'orig_DTI', 'orig_UPB', 'orig_LTV', 'orig_interest_rate', 'PPM',
    'number_of_borrowers', 'program_indicator', 'harp_indicator', 'current_UPB', 'loan_age', 'months_to_maturity', 
    'modification', 'current_interest_rate', 'current_deffered_UPB', 'step_modification', 
    'deferred_payment_plan', 'borrower_assistance_status', 'hpa', 'mtgrate', 'unemployment']
est_data[categorical_input] = est_data[categorical_input].astype('category')
df_dummy = pd.get_dummies(est_data[categorical_input])
df_full = pd.concat([est_data[numerical_input], df_dummy, est_data['prepay']], axis=1)

train, test = train_test_split(df_full, test_size=0.3)
train.to_csv('data/train_data.csv', index=False)
test.to_csv('data/test_data.csv', index=False)

model_baseline = LogisticRegression(max_iter=1000, solver='liblinear')
train_sc_base, test_sc_base, pred_base, confusion_mat_base, report_base = model_performance(model_baseline, train, test)
plot_confusion_matrix(confusion_mat_base, 'model/logistic/baseline/confusion_matrix.png')

model_regularized = LogisticRegression(max_iter=1000, penalty='l1', solver='liblinear')
res_regularized = model_performance(model_regularized, train, test)
np.where(np.abs(model2.coef_[0]) < 1e-3)
train_sc_reg, test_sc_reg, pred_reg, confusion_mat_reg, report_reg = model_performance(model_baseline, train, test)
plot_confusion_matrix(confusion_mat_reg, 'model/logistic/baseline/confusion_matrix.png')





