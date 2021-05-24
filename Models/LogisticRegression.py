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
train_y = train['zero_balance_code']
train_x = train.drop(columns='zero_balance_code')
test_y = test['zero_balance_code']
test_x = test.drop(columns='zero_balance_code')

logreg = sklearn.linear_model.LogisticRegression(max_iter=10000, solver='liblinear', random_state=seed)
logreg.fit(train_x, train_y)
score_train = logreg.score(train_x, train_y)
print(score_train)
score_test = logreg.score(test_x, test_y)
print(score_test)

predictions = logreg.predict(test)
sklearn.metrics.plot_confusion_matrix(logreg, test_x, test_y, cmap='Blues')
plt.show()
classification_report = sklearn.metrics.classification_report(test_y, predictions)
print(classification_report)

