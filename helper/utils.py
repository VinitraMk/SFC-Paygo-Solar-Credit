from datetime import datetime
from os import path, getenv
import yaml
import pandas as pd

def get_filename(filename):
    now = datetime.now().strftime('%d%m%Y-%H%M%S')
    return f'{filename}_{now}'

def get_config_path():
    return path.join(getenv('ROOT_DIR'),'args.yaml')

def get_all_args():
    config_path = get_config_path()
    all_args = {}
    with open(config_path) as file:
        all_args = yaml.full_load(file)
    return all_args

def get_config():
    all_args = get_all_args()
    return all_args['config']

def get_model_params(ensemble = False, model_name = ''):
    if not(ensemble):
        all_args = get_all_args()
        return all_args['model_args']
    else:
        all_args = get_all_args()
        model_args = all_args['model_args']['ensembler_args'][model_name]
        return model_args

def get_preproc_params():
    all_args = get_all_args()
    return all_args['preproc']

def get_validation_params():
    all_args = get_all_args()
    return all_args['validation']

def break_date(date):
    if date != '':
        return (date.day, date.month, date.year)
    return ('','','')

def extract_date(datestring):
    date = ''
    try:
        date = datetime.strptime(datestring,'%d-%m-%Y')
    except ValueError:
        try:
            date = datetime.strptime(datestring, '%Y-%m-%d')
        except ValueError:
            try:
                date = datetime.strptime(datestring,'%m-%d-%Y')
            except ValueError:
                return ''
    return date

def is_null(value):
    return pd.isnull(value) or pd.isna(value)

def save_fig(file_name, plt):
    config = get_all_args()['config']
    plt.savefig(f'{config["visualizations"]}/{file_name}.png')