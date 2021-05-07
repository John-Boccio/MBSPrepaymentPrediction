import os
import math
import pandas as pd

from Definitions import ROOT_DIR


class StandardLoanLevelDatasetParser:

    def __init__(self, max_rows_per_quarter=None, rows_to_sample=None, dump_to_csv=None, seed=None):
        self.sll_dataset_dir = os.path.join(ROOT_DIR, "Datasets/StandardLoanLevelDataset/Data")
        self._issuance_cols = [
            'credit_score', 'first_payment_date', 'first_time_homebuyer_flag', 'maturity_date', 'MSA', 'MI_%',
            'number_of_units', 'occupancy_status', 'orig_CLTV', 'orig_DTI', 'orig_UPB', 'orig_LTV',
            'orig_interest_rate', 'channel', 'PPM', 'amortization_type', 'property_state', 'property_type',
            'postal_code', 'loan_sequence_number', 'loan_purpose', 'orig_loan_term', 'number_of_borrowers',
            'seller_name', 'servicer_name', 'super_conforming', 'pre-harp_sequence_number', 'program_indicator',
            'harp_indicator', 'property_valuation_method', 'io_indicator'
        ]
        self._issuance_string = "historical_data"

        self._performance_cols = [
            'loan_sequence_number', 'report_month', 'current_UPB', 'current_loan_dlqc_status', 'loan_age',
            'months_to_maturity', 'repurchase', 'modification', 'zero_balance_code', 'zero_balance_date',
            'current_interest_rate', 'current_deffered_UPB', 'DDLPI', 'MI_recoveries', 'net_sales_proceeds',
            'non_MI_recoveries', 'expenses', 'legal_costs', 'maintenance_costs', 'taxes_and_insurence',
            'miscellaneous_expenses', 'actual_loss', 'modification_cost', 'step_modification', 'deferred_payment_plan',
            'estimated_LTV', 'zero_balance_removal_UPB', 'dlq_accrued_interest', 'dlqc_due_to_disaster',
            'borrower_assistance_status'
        ]
        self._performance_string = self._issuance_string + "_time"

        self.us_hpa_path = os.path.join(ROOT_DIR, "Datasets/MacroData/US_HPA.csv")
        self.us_hpa_cols = ['hpa_state', 'hpa_year', 'hpa_quarter', 'hpi']

        self.state_hpa_path = os.path.join(ROOT_DIR, "Datasets/MacroData/state_HPA.csv")
        self.state_hpa_cols = ['hpa_state', 'hpa_year', 'hpa_quarter', 'hpi']

        self.mtg_rate_path = os.path.join(ROOT_DIR, "Datasets/MacroData/mortgage_rate.csv")
        self.mtg_rate_cols = ['mtg_date', 'mtg_rate']
        self.mtg_rate_header = 8

        self.unemployment_rate_path = os.path.join(ROOT_DIR, "Datasets/MacroData/unemployment_rate.csv")
        self.unemployment_rate_cols = ['unemp_series_id', 'unemp_year', 'unemp_period', 'unemp_rate']
        self.unemployment_rate_header = 0

        self.data_types = {
            # Issuance data types
            'credit_score': 'int64', 'first_payment_date': 'int64', 'first_time_homebuyer_flag': 'S',
            'maturity_date': 'int64', 'MSA': 'float64', 'MI_%': 'int64', 'number_of_units': 'S',
            'occupancy_status': 'S', 'orig_CLTV': 'int64', 'orig_DTI': 'int64', 'orig_UPB': 'int64', 'orig_LTV': 'int64',
            'orig_interest_rate': 'float64', 'channel': 'S', 'PPM': 'S', 'amortization_type': 'S',
            'property_state': 'S', 'property_type': 'S', 'postal_code': 'int64', 'loan_sequence_number': 'S',
            'loan_purpose': 'S', 'orig_loan_term': 'int64', 'number_of_borrowers': 'int64', 'seller_name': 'S',
            'servicer_name': 'S', 'super_conforming': 'S', 'pre-harp_sequence_number': 'S', 'program_indicator': 'S',
            'harp_indicator': 'S', 'property_valuation_method': 'int64', 'io_indicator': 'S',
            # Performance data types
            'report_month': 'int64', 'current_UPB': 'float64', 'current_loan_dlqc_status': 'S', 'loan_age': 'int64',
            'months_to_maturity': 'int64', 'repurchase': 'S', 'modification': 'S', 'zero_balance_code': 'int64',
            'zero_balance_date': 'int64', 'current_interest_rate': 'float64', 'current_deffered_UPB': 'int64',
            'DDLPI': 'int64', 'MI_recoveries': 'float64', 'net_sales_proceeds': 'S', 'non_MI_recoveries': 'float64',
            'expenses': 'float64', 'legal_costs': 'float64', 'maintenance_costs': 'float64',
            'taxes_and_insurence': 'float64', 'miscellaneous_expenses': 'float64', 'actual_loss': 'float64',
            'modification_cost': 'float64', 'step_modification': 'S', 'deferred_payment_plan': 'S',
            'estimated_LTV': 'float64', 'zero_balance_removal_UPB': 'float64', 'dlq_accrued_interest': 'float64',
            'dlqc_due_to_disaster': 'S', 'borrower_assistance_status': 'S',
            # US / State HPA data types
            'hpa_state': 'S', 'hpa_year': 'int64', 'hpa_quarter': 'int64', 'hpi': 'float64',
            # Mortgage Rate data types
            'mtg_date': 'S', 'mtg_rate': 'float64',
            # Unemployment Rate data types
            'unemp_series_id': 'S', 'unemp_year': 'int64', 'unemp_period': 'S', 'unemp_rate': 'float64'
        }

        self.categorical_cols = ['occupancy_status', 'property_state', 'loan_purpose', 'month']
        self.numerical_cols = [
            'credit_score', 'first_time_homebuyer_flag', 'MI_%', 'number_of_units', 'orig_CLTV', 'orig_DTI', 'orig_UPB',
            'orig_LTV', 'orig_interest_rate', 'PPM', 'number_of_borrowers', 'program_indicator', 'harp_indicator',
            'current_UPB', 'loan_age', 'months_to_maturity', 'modification', 'current_interest_rate',
            'current_deffered_UPB', 'step_modification', 'deferred_payment_plan', 'borrower_assistance_status', 'hpa',
            'mtg_rate', 'unemp_rate', 'zero_balance_code'
        ]

        self.max_rows_per_quarter = max_rows_per_quarter
        self.data = pd.DataFrame()
        self.seed = seed
        self.rows_to_sample = rows_to_sample
        self.dump_to_csv = dump_to_csv

    def load(self):
        print(f"Loading Standard Loan-Level Dataset at path {self.sll_dataset_dir}")
        for root, dirs, _ in os.walk(self.sll_dataset_dir):
            for dir_name in dirs:
                if "historical_data_" not in dir_name or "Q" not in dir_name:
                    continue

                print(f"Adding data from {dir_name} to dataset...")
                dir_path = os.path.join(root, dir_name)
                split_name = dir_path.split('_')
                year = split_name[-1][:4]
                quarter = split_name[-1][4:]

                issuance_path = os.path.join(dir_path, self._issuance_string + "_" + year + quarter + ".txt")
                performance_path = os.path.join(dir_path, self._performance_string + "_" + year + quarter + ".txt")

                issuance = pd.read_csv(issuance_path, delimiter='|', names=self._issuance_cols, dtype=self.data_types, nrows=self.max_rows_per_quarter)
                issuance = issuance[issuance.eval("amortization_type=='FRM' & property_type=='SF' & orig_loan_term==360 & io_indicator=='N'")]
                if self.rows_to_sample:
                    issuance = issuance.sample(self.rows_to_sample, random_state=self.seed)

                performance = pd.read_csv(performance_path, delimiter='|', names=self._performance_cols, nrows=self.max_rows_per_quarter)
                performance = performance.loc[performance['loan_sequence_number'].isin(set(issuance['loan_sequence_number']))]

                full_data = performance.join(issuance.set_index('loan_sequence_number'), on='loan_sequence_number')
                full_data['year'] = int(year)
                full_data['quarter'] = int(quarter[1])
                full_data['report_year'] = full_data['report_month'] // 100
                full_data['report_quarter'] = full_data['report_month'] % 100 // 4 + 1
                self.data = self.data.append(full_data)

        us_hpa = pd.read_csv(self.us_hpa_path, delimiter=',', names=self.us_hpa_cols, dtype=self.data_types)
        us_hpa = us_hpa[us_hpa['hpa_state'] == 'USA']
        us_hpa['hpa'] = ((us_hpa.hpi / us_hpa.hpi.shift(1)) ** 4 - 1) * 100
        us_hpa['hpa_state'] = 'PR'

        state_hpa = pd.read_csv(self.state_hpa_path, delimiter=',', names=self.state_hpa_cols, dtype=self.data_types)
        state_hpa['hpa'] = ((state_hpa.hpi / state_hpa.hpi.shift(1)) ** 4 - 1) * 100

        hpa = pd.concat([state_hpa, us_hpa])
        self.data = self.data.join(hpa.set_index(['hpa_state', 'hpa_year', 'hpa_quarter']), on=['property_state', 'report_year', 'report_quarter'])

        mtg_rate = pd.read_csv(self.mtg_rate_path, header=self.mtg_rate_header, names=self.mtg_rate_cols, dtype=self.data_types)
        split_date = mtg_rate['mtg_date'].str.split('-', expand=True)
        mtg_rate['yearmon'] = split_date[0].astype(int) * 100 + split_date[1].astype(int)
        mtg_rate = mtg_rate.drop(columns='mtg_date')
        self.data = self.data.join(mtg_rate.set_index('yearmon'), on='report_month')

        unemp = pd.read_csv(self.unemployment_rate_path, delimiter='\t', header=self.unemployment_rate_header, names=self.unemployment_rate_cols, dtype=self.data_types)
        unemp.columns = self.unemployment_rate_cols
        unemp['yearmon'] = unemp['unemp_year'] * 100 + unemp['unemp_period'].str.slice(1, 3).astype(int)
        unemp = unemp.drop(columns=['unemp_series_id', 'unemp_year', 'unemp_period'])
        self.data = self.data.join(unemp.set_index('yearmon'), on='report_month')

        self._clean()

        if self.dump_to_csv:
            self.data.to_csv(self.dump_to_csv, index=False)

    def _clean(self):
        self.data['MSA'] = self.data['MSA'].apply(lambda x: 999 if math.isnan(x) else x)
        self.data['number_of_units'] = self.data['number_of_units'].apply(lambda x: 99 if x == '.' else int(x))
        self.data['harp_indicator'] = self.data['harp_indicator'].apply(lambda x: 0 if x != 'Y' else 1)
        self.data['modification'] = self.data['modification'].apply(lambda x: 0 if x != 'Y' else 1)
        self.data['step_modification'] = self.data['step_modification'].apply(lambda x: 0 if x != 'Y' else 1)
        self.data['deferred_payment_plan'] = self.data['deferred_payment_plan'].apply(lambda x: 0 if x != 'Y' else 1)
        self.data['borrower_assistance_status'] = self.data['borrower_assistance_status'].apply(lambda x: 0 if x not in ['F', 'R', 'T'] else 1)
        self.data['first_time_homebuyer_flag'] = self.data['first_time_homebuyer_flag'].apply(lambda x: 0 if x == 'N' else 1)
        self.data['program_indicator'] = self.data['program_indicator'].apply(lambda x: 0 if x == 9 else 1)
        self.data['current_deffered_UPB'] = self.data['current_deffered_UPB'].apply(lambda x: 0.0 if x == '.' else x)
        self.data['zero_balance_code'] = self.data['zero_balance_code'].apply(lambda x: 1 if x == 1 else 0)
        self.data['PPM'] = self.data['PPM'].apply(lambda x: 0 if x == 'N' else 1)
        self.data['month'] = self.data['report_month'] % 100

        self.data = self.data[self.data.hpa.isnull() == False]

    def get_dataset(self):
        self.data[self.categorical_cols] = self.data[self.categorical_cols].astype('category')
        df_dummy = pd.get_dummies(self.data[self.categorical_cols])
        return pd.concat([self.data[self.numerical_cols], df_dummy], axis=1)

    def _load_issuance_data(self, file_path):
        issuance = pd.read_csv(file_path, delimiter='|', header=None)
        issuance.columns = self._issuance_cols
        self.data.join(issuance.set_index('loan'))

