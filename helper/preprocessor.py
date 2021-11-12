#python library imports
from constants.contract_value_enum import Contract
from constants.age_enums import Age
from constants.term_enums import Term
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures
from scipy import stats
import matplotlib.pyplot as plt
import os
import seaborn as sns
import re
#custom imports
from helper.utils import get_config, break_date, extract_date, get_preproc_params, save_fig
from constants.general_enums import GeneralConstants

class Preprocessor:
    train= None
    train_ids = None
    test = None
    test_ids = None
    data = None
    preproc_args = None
    config = None
    poly_features = None
    selected_features = None

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
        if self.preproc_args['use_history_statistics']:
            self.generate_statistics_from_history()
        self.generate_polynomial_features()
        self.drop_skip_columns()
        self.preprocess_categorical_columns()
        self.plot_correlation()
        self.preprocess_date_columns()
        self.summarize_features()
        if not(self.preproc_args['use_history_statistics']):
            self.preprocess_history_columns()
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

    def generate_polynomial_features(self):
        print('\tGenerating polynomial features')
        poly = PolynomialFeatures(self.preproc_args['degree'])
        feature_matrix = poly.fit_transform(self.data[self.preproc_args['numeric_columns']])
        feature_matrix = feature_matrix[:, 1:]
        self.poly_features = poly.get_feature_names(self.preproc_args['numeric_columns'])[1:]
        new_feature_df = pd.DataFrame(feature_matrix, columns=self.poly_features)
        self.data = self.data.join(new_feature_df, lsuffix='DROP').filter(regex="^(?!.*DROP)")


    def plot_correlation(self):
        print('\tPlotting Correlation Matrix...')
        f = plt.figure(figsize=(60, 50))
        corr = self.data.corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        to_drop = [col for col in upper.columns if any(upper[col][:6] < self.preproc_args['correlation_threshold'])]
        poly_features_to_drop = [x for x in to_drop if x in self.poly_features]
        curr_poly_features = self.poly_features
        self.poly_features = [x for x in curr_poly_features if x not in poly_features_to_drop]
        corr.style.background_gradient(cmap='coolwarm').set_precision(2)
        sns.heatmap(corr, annot = True)
        save_fig('correlation_matrix', plt)
        plt.clf()
        self.data = self.data.drop(columns = poly_features_to_drop)

    def preprocess_date_columns(self):
        print('\tConverting date columns...')
        compute_date = lambda x: break_date(extract_date(x))
        col_list = [x for x in self.preproc_args['date_columns'] if x not in self.preproc_args['skip_columns']]
        for col in col_list:
            new_colname = col.replace('Date','').rstrip()
            date_col = self.data[col]
            self.data[[f'{new_colname}_Day',f'{new_colname}_Month', f'{new_colname}_Year']] = pd.DataFrame(date_col.apply(compute_date).to_list())
        self.data = self.data.drop(columns=col_list)

    def apply_ordinal_encoding(self, column_list):
        encoder = OrdinalEncoder()
        return encoder.fit_transform(self.data[column_list])

    def apply_onehot_encoding(self, column):
        pass

    def preprocess_term_binning(self):
        compute_term = lambda x: x/365
        self.data['TermCat'] = self.data['Term'].astype(str)
        self.data['Term'] = self.data['Term'].apply(compute_term)
        self.data.loc[(self.data['Term'] < 1.0),'TermCat'] = Term.LESS_THAN_1_YR.value
        self.data.loc[((self.data['Term'] >= 1.0) & (self.data['Term'] <= 1.5)), 'TermCat'] = Term.ONE_TO_ONEHALF_YR.value
        self.data.loc[(self.data['Term'] > 1.5), 'TermCat'] = Term.MORE_THAN_1_YR.value
        self.data = self.data.drop(columns=['Term'])

    def preprocess_age_binning(self):
        self.data['AgeCat'] = self.data['Age'].astype(str)
        self.data.loc[(self.data['Age'] < 30),'AgeCat'] = Age.LESS_THAN_30.value
        self.data.loc[((self.data['Age'] >= 30) & (self.data['Age'] <= 45)), 'AgeCat'] = Age.BTWN_30_AND_45.value
        self.data.loc[((self.data['Age'] > 45) & (self.data['Age'] <= 60)), 'AgeCat'] = Age.BTWN_45_AND_60.value
        self.data.loc[(self.data['Age'] > 60), 'AgeCat'] = Age.GREATER_THAN_60.value
        self.data = self.data.drop(columns=['Age'])

    def preprocess_contract_column(self):
        self.data['ContractCat'] = self.data['TotalContractValue'].astype(str)
        self.data.loc[(self.data['TotalContractValue'] < 25000),'ContractCat'] = Contract.LESS_THAN_25K.value
        self.data.loc[((self.data['TotalContractValue'] >= 25000) & (self.data['TotalContractValue'] <= 5000)), 'ContractCat'] = Contract.BTWN_25K_AND_50K.value
        self.data.loc[(self.data['TotalContractValue'] > 50000), 'ContractCat'] = Contract.MORE_THAN_50K.value
        self.data = self.data.drop(columns=['TotalContractValue'])

    def summarize_features(self):
        selected_features = self.poly_features + list(self.data.columns)
        self.selected_features = [x for x in selected_features if x not in ['m1','m2','m3','m4','m5','m6']]
        print('\tSelected Features: ',self.selected_features)

    def preprocess_categorical_columns(self):
        print('\tEncoding categorical variables...')
        self.preprocess_term_binning()
        #self.preprocess_age_binning()
        #self.preprocess_contract_column()
        column_list = [x for x in self.preproc_args['categorical_columns'] if x not in self.preproc_args['skip_columns']]
        if self.preproc_args['encoding_type'] == 'ordinal':
            self.data[column_list] = self.apply_ordinal_encoding(column_list)
        else:
            for col in column_list:
                self.apply_onehot_encoding(col)

    def preprocess_history_columns(self):
        print("\tProcessing history fields...")
        transaction_dates = self.data[['TransactionDates','ID']]
        payment_history = self.data[['PaymentsHistory','ID']]
        payment_history_new = []
        history_month = []
        history_year = []
        payment_history_indices = ['ID']
        transaction_month_indices = ['ID']
        transaction_year_indices = ['ID']
        for i in range(1, GeneralConstants.MAX_TERM_LENGTH.value + 1):
            payment_history_indices.append(f'transaction{i}Amount')
            transaction_month_indices.append(f'transaction{i}Month')
            transaction_year_indices.append(f'transaction{i}Year')
        for i, row in payment_history.iterrows():
            if(type(row['PaymentsHistory']) != str):
                print(row)
            history_data = [float(x) for x in row['PaymentsHistory'][1:len(row['PaymentsHistory'])-1].split(',')]
            padding = GeneralConstants.MAX_TERM_LENGTH.value - len(history_data)
            history_data.extend([-1] * padding)
            history_data.insert(0, row['ID'])
            payment_history_new.append(history_data)
        new_df = pd.DataFrame(payment_history_new, columns=payment_history_indices)
        self.data = pd.merge(self.data, new_df, on='ID')
        history_data=[]

        for i, row in transaction_dates.iterrows():
            history_data = [list(break_date(extract_date(re.sub('\s*[\']',"", x), GeneralConstants.DATE_FORMAT_MMYY), GeneralConstants.DATE_FORMAT_MMYY)) for x in row['TransactionDates'][1:len(row['TransactionDates'])-1].split(',')]
            padding = GeneralConstants.MAX_TERM_LENGTH.value - len(history_data)
            history_data.extend([[-1,-1]] * padding)
            history_month_temp = [x[0] for x in history_data]
            history_month_temp.insert(0, row['ID'])
            history_month.append(history_month_temp)
            history_year_temp = [x[1] for x in history_data]
            history_year_temp.insert(0, row['ID'])
            history_year.append(history_year_temp)
        new_df = pd.DataFrame(history_month, columns=transaction_month_indices)
        self.data = pd.merge(self.data, new_df, on='ID')
        new_df = pd.DataFrame(history_year, columns=transaction_year_indices)
        self.data = pd.merge(self.data, new_df, on='ID')
        self.data = self.data.drop(columns=['TransactionDates', 'PaymentsHistory'])

    def generate_monthly_ohe(self, history_data, history_month, stat):
        stat_indices = np.where(history_data == stat)
        stat_months = np.take(history_month, stat_indices)
        stat_months_bool = [0 if x not in stat_months else 1 for x in range(1, 13)]
        return stat_months_bool

    def generate_statistics_from_history(self):
        print('\tGenerating Statistics from Payment History...')
        payment_history = self.data[['PaymentsHistory','ID','TransactionDates']]
        payment_history_cols = ['ID', 'PaymentsCount','PaymentsMean','PaymentsMin','PaymentsMax','PaymentsMedian', 'PaymentsMode',
            'PaymentsStd','PaymentsPercentile25','PaymentsPercentile50','PaymentsPercentile75']
        '''
        for i in range(1, 13):
            payment_history_cols.append(f'Mode_Payment_{i}')
            payment_history_cols.append(f'Max_Payment_{i}')
            payment_history_cols.append(f'Min_Payment_{i}')
        '''
        payment_history_stats = []
        for i, row in payment_history.iterrows():
            history_data = np.array([float(x) for x in row['PaymentsHistory'][1:len(row['PaymentsHistory'])-1].split(',')])
            '''
            transaction_data = [list(break_date(extract_date(re.sub('\s*[\']',"", x), GeneralConstants.DATE_FORMAT_MMYY), GeneralConstants.DATE_FORMAT_MMYY)) for x in row['TransactionDates'][1:len(row['TransactionDates'])-1].split(',')]
            history_month = np.array([int(x[0]) for x in transaction_data])
            '''
            count = len(history_data)
            mean = history_data.mean()
            minh = history_data.min()
            maxh = history_data.max()
            median = np.median(history_data)
            mode = stats.mode(history_data)[0][0]
            std = history_data.std()
            percentile_25 = np.percentile(history_data, 25)
            percentile_50 = np.percentile(history_data, 50)
            percentile_75 = np.percentile(history_data, 75)
            '''
            monthly_ohes = self.generate_monthly_ohe(history_data, history_month, mode)
            monthly_ohes = monthly_ohes + self.generate_monthly_ohe(history_data, history_month, maxh)
            monthly_ohes = monthly_ohes + self.generate_monthly_ohe(history_data, history_month, minh)
            '''
            payment_history_stats.append([row['ID'], count, mean, minh, maxh, median, mode, std, percentile_25, percentile_50, percentile_75])
        new_df = pd.DataFrame(payment_history_stats, columns = payment_history_cols)
        self.data = pd.merge(self.data, new_df, on='ID')
        self.data = self.data.drop(columns = ['TransactionDates', 'PaymentsHistory'])


        
    def drop_skip_columns(self):
        print('\tDropping skip columns...')
        self.preproc_args = get_preproc_params()
        self.data = self.data.drop(columns=self.preproc_args['skip_columns'])
        #self.data = self.data.drop(columns=self.preproc_args['date_cleanup_columns'])

    def separate_data(self):
        print('Features list: ', self.data.shape[1])
        self.test = self.data[self.data['ID'].isin(self.test_ids)]
        self.train = self.data[self.data['ID'].isin(self.train_ids)]
        self.train = self.train.drop(columns=['ID'])
        self.test = self.test.drop(columns=['ID','m1', 'm2', 'm3', 'm4', 'm5', 'm6'])
        return self.train, self.test, self.test_ids, self.selected_features

    def plot_distribution(self):
        print('\tPlotting data distribution...\n')
        for col in self.data.columns:
            if col in self.preproc_args['skip_columns'] or col in self.preproc_args['date_skip'] or col == 'ID' or col.startswith('transaction') or col in self.poly_features:
                pass
            else:
                plt.hist(self.data[col])
                save_fig(f'{col}_plot', plt)
                plt.clf()
