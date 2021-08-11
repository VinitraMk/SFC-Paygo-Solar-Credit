from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.dataset import Dataset
from azureml.data import OutputFileDatasetConfig

from datetime import date
import os

# Building dataset
paths=[
    "https://drive.google.com/file/d/18jbOhpzWzsQXMiA8zmasMZtnNCp3x1wQ/view?usp=sharing", #Train.csv
    "https://drive.google.com/file/d/1TGGyIkb_naiLm04vuoOGLnQ81-_HKM3C/view?usp=sharing", #Test.csv
    "https://drive.google.com/file/d/1SEHYPSyIhowSlnzZrgO2TdHmrNG1o4XM/view?usp=sharing", #data.csv
]


# Configuring workspace
print('Configuring Workspace...\n')
today = date.today()
todaystring = today.strftime("%d-%m-%Y")
ws = Workspace.from_config()
def_blob_store = ws.get_default_datastore()
output = OutputFileDatasetConfig(destination=(def_blob_store, '/output'))

print('Configuring Dataset...\n')
def_blob_store.upload(src_dir='./input', target_path='input/', overwrite=True)
input_data = Dataset.File.from_files(path=(def_blob_store,'/input'))
input_data = input_data.as_named_input('input').as_mount()
# Configuring Environment
#azure_env = Environment.get(workspace=ws, name="AzureML-lightgbm-3.2-ubuntu18.04-py37-cpu")
#user_env = Environment.from_pip_requirements(name='vinazureml-env', file_path="./requirements.txt", pip_version="18.1")
print('Configuring Environment...\n')
user_env = Environment.get(workspace=ws, name="vinazureml-env")
experiment = Experiment(workspace=ws, name=f'{todaystring}-experiments')
config = ScriptRunConfig(
    source_directory='./',
    script='index.py',
    arguments=[input_data, output],
    compute_target='mikasa',
    environment=user_env)
run = experiment.submit(config)
run.wait_for_completion(show_output=True)
aml_url = run.get_portal_url()
print(aml_url)
