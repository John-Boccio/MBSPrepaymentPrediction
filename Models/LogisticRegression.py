import pandas as pd
import numpy as np
import scipy

from Datasets.StandardLoanLevelDataset.Parser.StandardLoanLevelDatasetParser import StandardLoanLevelDatasetParser

sll_data_parser = StandardLoanLevelDatasetParser(max_rows_per_quarter=500)
sll_data_parser.load()
print(sll_data_parser.data)

# TODO perform logistic regression
