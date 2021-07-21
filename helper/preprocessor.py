#python library imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

#custom imports
from helper.utils import get_config, break_date, extract_date, get_preproc_params

class Preprocessor:
    train= None
    train_ids = None
    test = None
    test_ids = None
    data = None
    config = None

    def __init__(self):
        self.config = get_config()
        self.train = pd.read_csv(f'{self.config["input_path"]}/Train.csv')
        self.train_ids = self.train['ID']
        self.test = pd.read_csv(f'{self.config["input_path"]}/Test.csv')
        self.test_ids = self.test['ID']
        self.data = pd.read_csv(f'{self.config["input_path"]}/data.csv')

    def start_preprocessing(self):
        print('\nStarting preprocessing of data...\n')
        self.preprocess_missing_data()
        self.preprocess_date_columns()
        self.preprocess_categorical_columns()
        self.drop_skip_columns()
        self.separate_data()

    def preprocess_missing_data(self):
        preproc_args = get_preproc_params()
        for col in preproc_args['na_columns']:
            if preproc_args['replacements'][col] == 'MODE':
                val = self.data[col].mode(dropna=True)
                self.data[col] = self.data[col].fillna(val[0])
            elif preproc_args['replacements'][col] == 'MEAN':
                self.data[col] = self.data[col].fillna(self.data[col].mean())
            else:
                self.data[col] = self.data[col].fillna(preproc_args['replacements'][col])
            print(col,'has null values: ',self.data[col].isna().sum())

    def preprocess_date_columns(self):
        compute_date = lambda x: break_date(extract_date(x))
        for col in ['RegistrationDate','UpsellDate', 'FirstPaymentDate', 'LastPaymentDate', 'ExpectedTermDate']:
            new_colname = col.replace('Date','').rstrip()
            date_col = self.data[col]
            self.data[[f'{new_colname}_Day',f'{new_colname}_Month', f'{new_colname}_Year']] = pd.DataFrame(date_col.apply(compute_date).to_list())

    def apply_ordinal_encoding(self, column):
        encoder = OrdinalEncoder()
        return encoder.fit_transform(column)

    def apply_onehot_encoding(self, column):
        pass

    def preprocess_categorical_columns(self):
        preproc_args = get_preproc_params()
        for col in ['MainApplicantGender','rareEntityType', 'Region','Town', 'Occupation']:
            if preproc_args['encoding_type'] == 'ordinal':
                self.data[col] = self.apply_ordinal_encoding(col)
            else:
                self.apply_onehot_encoding(col)

    def drop_skip_columns(self):
        preproc_args = get_preproc_params()
        self.data = self.data.drop(columns=preproc_args['skip_columns'])

    def separate_data(self):
        target_cols = ['m1','m2','m3','m4','m5','m6']
        self.test = self.data[self.data['ID'].isin(self.test_ids)]
        self.train = self.data[self.data['ID'].isin(self.train_ids)]
        self.train_Y = self.train[target_cols]
        self.train = self.train.drop(columns=target_cols+['ID'])
        return self.train, self.train_Y, self.test, self.test_ids