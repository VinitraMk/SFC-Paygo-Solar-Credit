import json
import os
import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
import datetime

if __name__ == "__main__":
    mounted_input_path = sys.argv[1]
    mounted_output_path = sys.argv[2]
    model_name = sys.argv[3]
    save_preds = sys.argv[4] == 'True'
    epoch_index = int(sys.argv[5])
    k = int(sys.argv[6])
    model_filename = sys.argv[7]
    model_params = json.loads(sys.argv[8])
    input_data_path = f'{mounted_input_path}/input'
    model_path = f'{mounted_input_path}/models'
    model_output_path = f'{mounted_output_path}/models'
    csv_output_path = f'{mounted_output_path}/results'
    log_output_path = f'{mounted_output_path}/experiment_logs'
    

    if os.path.isdir(model_output_path) == False:
        os.mkdir(model_output_path)
    if os.path.isdir(csv_output_path) == False:
        os.mkdir(csv_output_path)
    if os.path.isdir(log_output_path) == False:
        os.mkdir(log_output_path)

    model = joblib.load(f'{model_path}/{model_name}_model.sav')
    X = pd.read_csv(f'{input_data_path}/train_X.csv')
    y = pd.read_csv(f'{input_data_path}/train_y.csv')
    valid_X = pd.read_csv(f'{input_data_path}/valid_X.csv')
    valid_y = pd.read_csv(f'{input_data_path}/valid_y.csv')
    test_X = None
    test_ids = None
    y_preds = None
    avg_score = 0.0


    if save_preds:
        test_X = pd.read_csv(f'{input_data_path}/test_X.csv')
        test_ids = pd.read_csv(f'{input_data_path}/test_ids.csv')

    model = model.fit(X, y)
    joblib.dump(model, f'{model_output_path}/{model_name}_model.sav')
    if save_preds:
        ypreds = model.predict(test_X)
    else:
        ypreds = model.predict(valid_X)
    
    log_path = f'{mounted_output_path}/{model_name}_log.txt'
    log_file_contents = dict()
    if not(save_preds):
        avg_score = mean_squared_error(valid_y.values, ypreds, squared = False)
    if os.path.isfile(log_path):
        with open(log_path) as log_file:
            log_file_contents = json.load(log_file)
        if not(save_preds):
            avg_score = avg_score + log_file_contents['model_output']['current_run_rmse']
        else:
            avg_score = log_file_contents['model_output']['current_run_rmse']
        if epoch_index == k-1:
            avg_score = avg_score / k
    log_file_contents['model_output'] = { 'current_run_rmse': avg_score }
    if save_preds == True:
        log_path = f'{log_output_path}/{model_filename}.txt'
        log_file_contents['model_params'] = model_params
        #log_file_contents['model_output'] = { 'final_validation_rmse': avg_score }
        preds_df = pd.DataFrame(ypreds, columns=['m1','m2','m3','m4','m5','m6'])
        preds_df = test_ids.join(preds_df)
        preds_df['temp_ID'] = preds_df['ID']
        subm_df = pd.melt(preds_df, id_vars=['temp_ID'], value_vars=['m1','m2','m3','m4','m5','m6'])
        subm_df['ID'] = subm_df['temp_ID'].str.cat(subm_df['variable'], sep = ' x ')
        subm_df['Target'] = subm_df['value']
        subm_df = subm_df.drop(columns=['temp_ID','variable','value'])
        subm_df.to_csv(f'{csv_output_path}/{model_filename}.csv', index = False)
    #joblib.dump(log_file_contents, log_path)
    with open(log_path, 'w') as out_file:
        json.dump(log_file_contents, out_file)

