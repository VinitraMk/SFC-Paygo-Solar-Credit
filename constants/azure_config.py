import enum

class AzureConfig(str, enum.Enum):
    STORAGEACCOUNTURL= 'https://mlintro1651836008.blob.core.windows.net/'
    MAINCONTAINER = 'azureml-blobstore-31aeaa24-564c-4aa8-bdf4-fc4b5707bd1b/output'
    CSVCONTAINER = f'{MAINCONTAINER}/results'
    LOGCONTAINER = f'{MAINCONTAINER}/experiment_logs'
    ENVIRONMENT_NAME = 'vinazureml-env'