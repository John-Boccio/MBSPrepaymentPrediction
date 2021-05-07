import pandas as pd
import numpy as np
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import matplotlib.pyplot as plt

from Datasets.StandardLoanLevelDataset.Parser.StandardLoanLevelDatasetParser import StandardLoanLevelDatasetParser


seed = 10

sll_data_parser = StandardLoanLevelDatasetParser(max_rows_per_quarter=10000, rows_to_sample=1000, seed=seed)
sll_data_parser.load()

train, test = sklearn.model_selection.train_test_split(sll_data_parser.get_dataset(), test_size=0.3)
train_prepayment = train['zero_balance_code']
train = train.drop(columns='zero_balance_code')
test_prepayment = test['zero_balance_code']
test = test.drop(columns='zero_balance_code')

logreg = sklearn.linear_model.LogisticRegression(max_iter=10000, solver='liblinear', random_state=seed)
logreg.fit(train, train_prepayment)
score_train = logreg.score(train, train_prepayment)
print(score_train)
score_test = logreg.score(test, test_prepayment)
print(score_test)

predictions = logreg.predict(test)
sklearn.metrics.plot_confusion_matrix(logreg, test, test_prepayment)
plt.show()
classification_report = sklearn.metrics.classification_report(test_prepayment, predictions)
print(classification_report)

