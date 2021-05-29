import sklearn.metrics
import sklearn.linear_model
import matplotlib.pyplot as plt

from Datasets.StandardLoanLevelDataset.Parser.StandardLoanLevelDatasetParser import StandardLoanLevelDatasetParser


sll_data_parser = StandardLoanLevelDatasetParser(max_rows_per_quarter=250000, rows_to_sample=75000)
sll_data_parser.load()

df = sll_data_parser.get_dataset()

train, val = sklearn.model_selection.train_test_split(df, test_size=0.2)
train, test = sklearn.model_selection.train_test_split(train, test_size=0.2)

train_y = train['zero_balance_code']
train_x = train.drop(columns='zero_balance_code')
val_y = val['zero_balance_code']
val_x = val.drop(columns='zero_balance_code')
test_y = test['zero_balance_code']
test_x = test.drop(columns='zero_balance_code')

logreg = sklearn.linear_model.LogisticRegression(max_iter=10000, solver='liblinear')
logreg.fit(train_x, train_y)
score_train = logreg.score(train_x, train_y)
print(score_train)
score_test = logreg.score(val_x, val_y)
print(score_test)

predictions = logreg.predict(test_x)
sklearn.metrics.plot_confusion_matrix(logreg, test_x, test_y, values_format='', cmap='Blues')
plt.show()
classification_report = sklearn.metrics.classification_report(test_y, predictions)
print(classification_report)

