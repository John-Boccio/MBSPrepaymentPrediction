import pandas as pd
import numpy as np
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model

from Datasets.StandardLoanLevelDataset.Parser.StandardLoanLevelDatasetParser import StandardLoanLevelDatasetParser


seed = 10

sll_data_parser = StandardLoanLevelDatasetParser(dump_to_csv="./help.csv", max_rows_per_quarter=10000, rows_to_sample=1000, seed=seed)
sll_data_parser.load()

train, test = sklearn.model_selection.train_test_split(sll_data_parser.get_dataset(), test_size=0.3)
train_prepayment = train['zero_balance_code']
train_prepayment = train_prepayment.drop(columns='zero_balance_code')
test_prepayment = test['zero_balance_code']
test_prepayment = test_prepayment.drop(columns='zero_balance_code')

print(f'Train prepayment {len(train_prepayment.loc[train_prepayment == 1]) / len(train_prepayment) * 100}%')
print(f'Test prepayment {len(test_prepayment.loc[test_prepayment == 1]) / len(test_prepayment) * 100}%')

logreg = sklearn.linear_model.LogisticRegression(max_iter=10000, solver='liblinear', random_state=seed)
logreg.fit(train, train_prepayment)
score_train = logreg.score(train, train_prepayment)
print(score_train)
score_test = logreg.score(test, test_prepayment)
print(score_test)
