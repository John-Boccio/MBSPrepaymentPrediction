import sklearn.metrics
import sklearn.svm
import sklearn.pipeline
import sklearn.preprocessing
import matplotlib.pyplot as plt
from Datasets.StandardLoanLevelDataset.Parser.StandardLoanLevelDatasetParser import StandardLoanLevelDatasetParser

sll_data_parser = StandardLoanLevelDatasetParser(max_rows_per_quarter=10000, rows_to_sample=1000)
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

kernels = ['poly', 'rbf', 'sigmoid']
for kernel in kernels:
    svm = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.svm.SVC(kernel=kernel))
    svm.fit(train_x, train_y)

    predictions = svm.predict(test_x)
    classification_report = sklearn.metrics.classification_report(test_y, predictions)
    print(f'Classification report for SVM with {kernel} Kernel: \n{classification_report}')

    sklearn.metrics.plot_confusion_matrix(svm, test_x, test_y, display_labels=['No Prepayment', 'Prepayment'], cmap=plt.cm.Blues)
    plt.title(f'SVM with {kernel} kernel')
    plt.show()
    plt.clf()
