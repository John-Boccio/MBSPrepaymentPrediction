import sklearn.metrics
import sklearn.svm
import sklearn.pipeline
import sklearn.preprocessing
import matplotlib.pyplot as plt
from Datasets.StandardLoanLevelDataset.Parser.StandardLoanLevelDatasetParser import StandardLoanLevelDatasetParser


seed = 10

sll_data_parser = StandardLoanLevelDatasetParser(dump_to_csv="help.csv", max_rows_per_quarter=10000, rows_to_sample=1000, seed=seed)
sll_data_parser.load()

train, test = sklearn.model_selection.train_test_split(sll_data_parser.get_dataset(), test_size=0.3)
train_y = train['zero_balance_code']
train_x = train.drop(columns='zero_balance_code')
test_y = test['zero_balance_code']
test_x = test.drop(columns='zero_balance_code')

kernels = ['poly', 'rbf', 'sigmoid']
for kernel in kernels:
    svm = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.svm.SVC(gamma='auto', kernel=kernel))
    svm.fit(train_x, train_y)

    predictions = svm.predict(test_x)
    classification_report = sklearn.metrics.classification_report(test_y, predictions)
    print(f'Classification report for SVM with {kernel} Kernel: \n{classification_report}')

    sklearn.metrics.plot_confusion_matrix(svm, test_x, test_y, display_labels=['No Prepayment', 'Prepayment'], cmap=plt.cm.Blues)
    plt.title(f'SVM with {kernel} kernel')
    plt.show()
    plt.clf()

