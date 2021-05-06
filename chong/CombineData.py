import pandas as pd
import numpy as np
import math
import os
os.chdir('/Users/chongguo/Documents/Stanford/AI_Certificate/CS229/project/')
import sys
sys.path.insert(0, '/Users/chongguo/Documents/Stanford/AI_Certificate/CS229/project/code')
from StandardLoanLevelDatasetParser import StandardLoanLevelDatasetParser

freddie_data_parser = StandardLoanLevelDatasetParser()
freddie_data_parser.load()
# print(freddie_data_parser.data.head())

raw_data = freddie_data_parser.data
raw_data['report_year'] = raw_data['report_month'] // 100
raw_data['report_quarter'] = raw_data['report_month'] % 100 // 4 + 1

us_hpa = pd.read_csv("data/Macro/US_HPA.csv", header=None)
us_hpa.columns = ['state', 'year', 'quarter', 'hpi']
us_hpa = us_hpa[us_hpa.state == 'USA']
us_hpa['hpa'] = ((us_hpa.hpi / us_hpa.hpi.shift(1)) ** 4 - 1) * 100
us_hpa['state'] = 'PR'

state_hpa = pd.read_csv("data/Macro/state_HPA.csv", header=None)
state_hpa.columns = ['state', 'year', 'quarter', 'hpi']
state_hpa['hpa'] = ((state_hpa.hpi / state_hpa.hpi.shift(1)) ** 4 - 1) * 100

hpa = pd.concat([state_hpa, us_hpa])
raw_data = raw_data.join(hpa.set_index(['state', 'year', 'quarter']), on = ['property_state', 'report_year', 'report_quarter'])

mtgrate = pd.read_csv("data/Macro/mortgage_rate.csv", header=8)
mtgrate['yearmon'] = pd.DatetimeIndex(mtgrate.date).year * 100 + pd.DatetimeIndex(mtgrate.date).month
mtgrate = mtgrate.drop(columns='date')
mtgrate.columns = ['mtgrate', 'date']
raw_data = raw_data.join(mtgrate.set_index('date'), on = 'report_month')

unemp = pd.read_csv("data/Macro/unemployment_rate.csv", delimiter='\t')
unemp['date'] = unemp.Year * 100 + unemp.Period.str.slice(1,3).astype(int)
unemp = unemp.drop(columns=['Series ID', 'Year', 'Period'])
unemp.columns = ['unemployment', 'date']
raw_data = raw_data.join(unemp.set_index('date'), on = 'report_month')

raw_data.to_csv('data/raw_data.csv', index=False)


est_data = pd.read_csv('data/raw_data.csv')
est_cols = ['loan_sequence_number', 'report_month', 'credit_score', 'first_payment_date', \
            'first_time_homebuyer_flag', 'MSA', 'MI_%', 'number_of_units', 'occupancy_status', 'orig_CLTV', \
            'orig_DTI', 'orig_UPB', 'orig_LTV', 'orig_interest_rate', 'PPM', 'property_state', 'loan_purpose', \
            'number_of_borrowers', 'program_indicator', 'harp_indicator', 'current_UPB', 'loan_age', \
            'months_to_maturity', 'modification', 'zero_balance_code', 'current_interest_rate', \
            'current_deffered_UPB', 'step_modification', 'deferred_payment_plan', 'borrower_assistance_status', \
            'hpa', 'mtgrate', 'unemployment']
est_data = est_data[est_cols]

# code dummy variables
est_data['MSA'] = est_data['MSA'].apply(lambda x: 99999 if math.isnan(x) else x)
est_data['number_of_units'] = est_data['number_of_units'].apply(lambda x: 99999 if x == '.' else x)
est_data['harp_indicator'] = est_data['harp_indicator'].apply(lambda x: 0 if x != 'Y' else 1)
est_data['modification'] = est_data['modification'].apply(lambda x: 0 if x != 'Y' else 1)
est_data['step_modification'] = est_data['step_modification'].apply(lambda x: 0 if x != 'Y' else 1)
est_data['deferred_payment_plan'] = est_data['deferred_payment_plan'].apply(lambda x: 0 if x != 'Y' else 1)
est_data['borrower_assistance_status'] = est_data['borrower_assistance_status'].apply(
    lambda x: 0 if x not in ['F', 'R', 'T'] else 1)
est_data['first_time_homebuyer_flag'] = est_data['first_time_homebuyer_flag'].apply(lambda x: 0 if x == 'N' else 1)
est_data['program_indicator'] = est_data['program_indicator'].apply(lambda x: 0 if x == 9 else 1)
est_data['PPM'] = est_data['PPM'].apply(lambda x: 0 if x == 'N' else 1)
est_data['month'] = est_data['report_month'] % 100

# generate y variable
est_data['prepay'] = est_data['zero_balance_code'].apply(lambda x: 1 if x == 1 else 0)
est_data = est_data.drop(columns='zero_balance_code')

est_data = est_data.astype({'loan_sequence_number':'S', 'report_month':'int64', 'credit_score':'int64', \
                            'first_payment_date': 'int64', 'first_time_homebuyer_flag':'int64', 'MSA':'int64', \
                            'MI_%':'int64', 'number_of_units':'int64', 'occupancy_status':'S', \
                            'orig_CLTV':'int64', 'orig_DTI':'int64', 'orig_UPB':'int64', 'orig_LTV':'int64', \
                            'orig_interest_rate':'float64','PPM':'int64', 'property_state':'S', 'loan_purpose':'S', \
                            'number_of_borrowers':'int64', 'program_indicator':'int64', 'harp_indicator':'int64', \
                            'current_UPB':'float64', 'loan_age':'int64', 'months_to_maturity':'int64', \
                            'modification':'int64', 'current_interest_rate':'float64', \
                            'current_deffered_UPB':'float64', 'step_modification':'int64', \
                            'deferred_payment_plan':'int64', 'borrower_assistance_status':'int64', 'hpa':'float64', \
                            'mtgrate':'float64', 'unemployment':'float64', 'prepay':'int64', 'month':'int64'})
est_data = est_data[est_data.hpa.isnull() == False]

est_data.to_csv('data/est_data.csv', index=False)