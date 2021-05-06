import os
import random
import pandas as pd


class StandardLoanLevelDatasetParser:
    _issuance_string = "historical_data"
    _performance_string = _issuance_string + "_time"
    _issuance_cols = [
        'credit_score', 'first_payment_date', 'first_time_homebuyer_flag', 'maturity_date', 'MSA', 'MI_%',
        'number_of_units', 'occupancy_status', 'orig_CLTV', 'orig_DTI', 'orig_UPB', 'orig_LTV', 'orig_interest_rate',
        'channel', 'PPM', 'amortization_type', 'property_state', 'property_type', 'postal_code', 'loan_sequence_number',
        'loan_purpose', 'orig_loan_term', 'number_of_borrowers', 'seller_name', 'servicer_name', 'super_conforming',
        'pre-harp_sequence_number', 'program_indicator', 'harp_indicator', 'property_valuation_method', 'io_indicator'
    ]
    _performance_cols = [
        'loan_sequence_number', 'report_month', 'current_UPB', 'current_loan_dlqc_status', 'loan_age',
        'months_to_maturity', 'repurchase', 'modification', 'zero_balance_code', 'zero_balance_date',
        'current_interest_rate', 'current_deffered_UPB', 'DDLPI', 'MI_recoveries', 'net_sales_proceeds',
        'non_MI_recoveries', 'expenses', 'legal_costs', 'maintenance_costs', 'taxes_and_insurence',
        'miscellaneous_expenses', 'actual_loss', 'modification_cost', 'step_modification', 'deferred_payment_plan',
        'estimated_LTV', 'zero_balance_removal_UPB', 'dlq_accrued_interest', 'dlqc_due_to_disaster',
        'borrower_assistance_status'
    ]

    def __init__(self, path="data/Freddie", max_rows_per_quarter=999999999999, num_select=500, seed=10):
        self.path = path
        self.max_rows_per_quarter = max_rows_per_quarter
        self.data = pd.DataFrame()
        self.num_select = num_select
        random.seed(seed)

    def load(self):
        print(f"Loading Standard Loan-Level Dataset at path {self.path}")

        for root, dirs, _ in os.walk(self.path):
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
                issuance = pd.read_csv(issuance_path, delimiter='|', header=None, nrows=self.max_rows_per_quarter)
                issuance.columns = self._issuance_cols
                issuance = issuance[issuance.eval("amortization_type=='FRM' & property_type=='SF' & orig_loan_term==360 & io_indicator=='N'")]
                issuance = issuance.sample(self.num_select)
                performance = pd.read_csv(performance_path, delimiter='|', header=None, nrows=self.max_rows_per_quarter)
                performance.columns = self._performance_cols
                performance = performance.loc[performance['loan_sequence_number'].isin(set(issuance['loan_sequence_number']))]
                full_data = performance.join(issuance.set_index('loan_sequence_number'), on='loan_sequence_number')
                full_data['year'] = int(year)
                full_data['quarter'] = int(quarter[1])
                self.data = self.data.append(full_data)
            

