import os
from helper.utils import get_model_params

def run_model(args):
    model = None
    if args['model'] == 'logistic':
        print(args['model'])
    else:
        print('Invalid model name :-( \n')
        exit()

def main(args):
    run_model(args)

def read_args():
    args = get_model_params()
    main(args)

def set_root_dir():
    if not(os.getenv('ROOT_DIR')):
        os.environ['ROOT_DIR'] = os.getcwd()

if __name__ == "__main__":
    set_root_dir()
    read_args()