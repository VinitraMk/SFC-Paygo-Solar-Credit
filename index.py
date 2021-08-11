#python imports
import json
import os
from azureml.core import Workspace, Experiment as AzExperiment, Environment, ScriptRunConfig
from azureml.core.dataset import Dataset
from azureml.data import OutputFileDatasetConfig
from azure.storage.blob import BlobServiceClient
from datetime import date
import pandas as pd

#custom imports
from helper.utils import get_config, get_filename, get_model_params, get_validation_params, save_model 
from helper.preprocessor import Preprocessor
from models.linear_regression import LinearRegressor
from experiments.validate import Validate
from experiments.experiment import Experiment
from experiments.validate import Validate

def make_model(args):
    if args['model'] == 'linear':
        linear_regression = LinearRegressor()
        return linear_regression.get_model()
    else:
        print('Invalid model name :-( \n')
        exit()

def make_azure_res():
    print('\nConfiguring Azure Resources...')
    # Configuring workspace
    print('\tConfiguring Workspace...')
    today = date.today()
    todaystring = today.strftime("%d-%m-%Y")
    ws = Workspace.from_config()

    print('\tConfiguring Environment...\n')
    user_env = Environment.get(workspace=ws, name="vinazureml-env")
    experiment = AzExperiment(workspace=ws, name=f'{todaystring}-experiments')
    
    return experiment, ws, user_env
    

def train_model_in_azure(azexp, azws, azuserenv, model_name, epoch_index, validation_k, model_args_string, save_preds = False, filename = ''):
    def_blob_store = azws.get_default_datastore()
    def_blob_store.upload(src_dir='./processed_io', target_path='input/', overwrite=True)
    input_data = Dataset.File.from_files(path=(def_blob_store,'/input'))
    input_data = input_data.as_named_input('input').as_mount()
    output = OutputFileDatasetConfig(destination=(def_blob_store, '/output'))

    config = ScriptRunConfig(
        source_directory='./models',
        script='train.py',
        arguments=[input_data, output, model_name, save_preds, epoch_index, validation_k, filename, model_args_string],
        compute_target='mikasa',
        environment=azuserenv)
    run = azexp.submit(config)
    run.wait_for_completion(show_output=True)
    aml_url = run.get_portal_url()
    print(aml_url)

def preprocess_data():
    preprocessor = Preprocessor()
    return preprocessor.start_preprocessing()

def download_blob(local_filename, blob_client_instance):
    with open(local_filename, "wb") as my_blob:
        blob_data = blob_client_instance.download_blob()
        blob_data.readinto(my_blob)

def download_output(filename, model_name):
    config = get_config()
    STORAGEACCOUNTURL= 'https://mlintro1651836008.blob.core.windows.net/'
    LOCALCSVPATH = f'{config["output_path"]}\\{filename}.csv'
    LOCALLOGPATH = f'{config["experimental_output_path"]}\\{filename}.txt'
    MAINCONTAINER = 'azureml-blobstore-31aeaa24-564c-4aa8-bdf4-fc4b5707bd1b/output'
    CSVCONTAINER = f'{MAINCONTAINER}/results'
    LOGCONTAINER = f'{MAINCONTAINER}/experiment_logs'
    CSVBLOB = f'{filename}.csv'
    LOGBLOB = f'{filename}.txt'
    EXPLOGBLOB = f'{model_name}_log.txt'
    blob_service_client = BlobServiceClient(account_url= STORAGEACCOUNTURL, credential = os.environ['AZURE_STORAGE_CONNECTIONSTRING'])
    blob_client_csv = blob_service_client.get_blob_client(CSVCONTAINER, CSVBLOB, snapshot = None)
    blob_client_log = blob_service_client.get_blob_client(LOGCONTAINER, LOGBLOB, snapshot = None)
    blob_client_explog = blob_service_client.get_blob_client(MAINCONTAINER, EXPLOGBLOB, snapshot = None)
    download_blob(LOCALCSVPATH, blob_client_csv)
    download_blob(LOCALLOGPATH, blob_client_log)
    blob_client_explog.delete_blob()

def start_validation(data, args, validation_args, test_ids, test_X):
    azexp, azws, azuserenv = make_azure_res()
    print('Starting experiment...')
    validate = Validate(data, test_X, test_ids)
    model_args_string = json.dumps(args)
    for i in range(validation_args['k']):
        print('\n*************** Run', i,'****************')
        validate.prepare_validation_dataset()
        train_model_in_azure(azexp, azws, azuserenv, args['model'], i , validation_args['k'], model_args_string)
    filename = get_filename(args['model'])
    print('\n\n*************** Final Run ****************')
    validate.prepare_full_dataset()
    train_model_in_azure(azexp, azws, azuserenv, args['model'], -1 , validation_args['k'], model_args_string, True, filename)
    download_output(filename, args['model'])

def read_args():
    args = get_model_params()
    config = get_config()
    validation_args = get_validation_params()
    data, test_X, test_ids = preprocess_data()
    model = make_model(args)
    model_path = f'{config["processed_io_path"]}/models'
    save_model(model, model_path, args['model'])
    start_validation(data, args, validation_args, test_ids, test_X)
    #main(args, validation_args)

def set_root_dir():
    if not(os.getenv('ROOT_DIR')):
        os.environ['ROOT_DIR'] = os.getcwd()

if __name__ == "__main__":
    set_root_dir()
    read_args()