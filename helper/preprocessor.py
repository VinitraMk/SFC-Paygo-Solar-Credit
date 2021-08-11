#python library imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import os

#custom imports
from helper.utils import get_config, break_date, extract_date, get_preproc_params, save_fig

class Preprocessor:
    train= None
    train_ids = None
    test = None
    test_ids = None
    data = None
    preproc_args = None
    config = None

    def __init__(self):
        self.config = get_config()
        self.train = pd.read_csv(f'{self.config["input_path"]}/Train.csv')
        self.train_ids = self.train['ID']
        self.test = pd.read_csv(f'{self.config["input_path"]}/Test.csv')
        self.test_ids = self.test['ID']
        self.data = pd.read_csv(f'{self.config["input_path"]}/data.csv')
        self.preproc_args = get_preproc_params()

    def start_preprocessing(self):
        print('\nStarting preprocessing of data...')
        self.remove_dirty_values()
        self.preprocess_missing_data()
        self.preprocess_date_columns()
        self.drop_skip_columns()
        self.preprocess_categorical_columns()
        self.plot_distribution()
        return self.separate_data()
    
    def remove_dirty_values(self):
        print('\tRemoving dirty values...')
        for col in self.data.columns:
            if col == 'Town':
                self.data[col] = self.data[col].replace('UNKNOWN', np.NaN)

    def preprocess_missing_data(self):
        print('\tProcessing missing values...')
        for col in self.preproc_args['na_columns']:
            if self.preproc_args['replacements'][col] == 'MODE':
                val = self.data[col].mode(dropna=True)
                self.data[col] = self.data[col].fillna(val[0])
            elif self.preproc_args['replacements'][col] == 'MEAN':
                self.data[col] = self.data[col].fillna(self.data[col].mean())
            else:
                self.data[col] = self.data[col].fillna(self.preproc_args['replacements'][col])
            #print(col,'has null values: ',self.data[col].isna().sum())

    def preprocess_date_columns(self):
        print('\tConverting date columns...')
        compute_date = lambda x: break_date(extract_date(x))
        for col in self.preproc_args['date_columns']:
            new_colname = col.replace('Date','').rstrip()
            date_col = self.data[col]
            self.data[[f'{new_colname}_Day',f'{new_colname}_Month', f'{new_colname}_Year']] = pd.DataFrame(date_col.apply(compute_date).to_list())

    def apply_ordinal_encoding(self, column_list):
        encoder = OrdinalEncoder()
        return encoder.fit_transform(self.data[column_list])

    def apply_onehot_encoding(self, column):
        pass

    def preprocess_categorical_columns(self):
        print('\tEncoding categorical variables...')
        if self.preproc_args['encoding_type'] == 'ordinal':
            target_cols = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']
            column_list = [x for x in self.data.columns if x != 'ID' and x not in target_cols]
            self.data[column_list] = self.apply_ordinal_encoding(column_list)
        else:
            for col in ['MainApplicantGender','rateTypeEntity', 'Region','Town', 'Occupation']:
                self.apply_onehot_encoding(col)

    def drop_skip_columns(self):
        print('\tDropping skip columns...')
        self.preproc_args = get_preproc_params()
        self.data = self.data.drop(columns=self.preproc_args['skip_columns'])

    def separate_data(self):
        self.test = self.data[self.data['ID'].isin(self.test_ids)]
        self.train = self.data[self.data['ID'].isin(self.train_ids)]
        self.train = self.train.drop(columns=['ID'])
        self.test = self.test.drop(columns=['ID','m1', 'm2', 'm3', 'm4', 'm5', 'm6'])
        return self.train, self.test, self.test_ids

    def plot_distribution(self):
        print('\tPlotting data distribution...\n')
        for col in self.data.columns:
            if col in self.preproc_args['skip_columns'] or col in self.preproc_args['date_skip'] or col == 'ID':
                pass
            else:
                plt.hist(self.data[col])
                save_fig(f'{col}_plot', plt)
                plt.clf()
