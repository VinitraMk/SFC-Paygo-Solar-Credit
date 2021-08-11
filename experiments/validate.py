from helper.utils import get_validation_params, get_config
import numpy as np
import pandas as pd

class Validate:
    data = None
    train_X = None
    train_y = None
    val_X = None
    val_y = None
    test_X = None
    test_ids = None
    validation_params = None
    config_params = None

    def __init__(self, data, test_X, test_ids):
        self.data = data
        self.test_X = test_X
        self.test_ids = test_ids
        self.validation_params = get_validation_params()
        self.config_params = get_config()

    def prepare_validation_dataset(self):
        target_cols = ['m1','m2','m3','m4','m5','m6']
        k = self.validation_params['k']
        val_len = int(self.data.shape[0] / k)
        val_indices = np.random.randint(self.data.shape[0], size=val_len)
        val_data = self.data.iloc[val_indices]
        train_data = self.data.drop(val_indices, axis=0)
        self.train_y = train_data[target_cols]
        self.train_X = train_data.drop(columns=target_cols, axis=1)
        self.val_y = val_data[target_cols]
        self.val_X = val_data.drop(columns=target_cols, axis=1)
        self.train_X.to_csv(f'{self.config_params["processed_io_path"]}\\input\\train_X.csv', index = False)
        self.train_y.to_csv(f'{self.config_params["processed_io_path"]}\\input\\train_y.csv', index = False)
        #print(self.train_y.head())
        self.val_X.to_csv(f'{self.config_params["processed_io_path"]}\\input\\valid_X.csv', index = False)
        self.val_y.to_csv(f'{self.config_params["processed_io_path"]}\\input\\valid_y.csv', index = False)
        #return self.train_X, self.train_y, self.val_X, self.val_y

    def prepare_full_dataset(self):
        target_cols = ['m1','m2','m3','m4','m5','m6']
        self.train_X = self.data.drop(columns=target_cols)
        self.train_y = self.data[target_cols]
        self.train_X.to_csv(f'{self.config_params["processed_io_path"]}\\input\\train_X.csv', index = False)
        self.train_y.to_csv(f'{self.config_params["processed_io_path"]}\\input\\train_y.csv', index = False)
        self.test_X.to_csv(f'{self.config_params["processed_io_path"]}\\input\\test_X.csv', index = False)
        self.test_ids.to_csv(f'{self.config_params["processed_io_path"]}\\input\\test_ids.csv', index = False)
        #return self.train_X, self.train_y
