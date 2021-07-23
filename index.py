#python imports
from experiments.validate import Validate
from sklearn.utils.validation import as_float_array
from experiments.experiment import Experiment
from experiments.validate import Validate
import os

#custom imports
from helper.utils import get_model_params, get_validation_params
from helper.preprocessor import Preprocessor

def run_model(args):
    model = None
    if args['model'] == 'logistic':
        print(args['model'])
        return model
    else:
        print('Invalid model name :-( \n')
        exit()

def main(args, validation_args):
    model_name = args['model']
    preprocessor = Preprocessor()
    data, test_X, test_ids = preprocessor.start_preprocessing()
    avg_score = 0
    validate = Validate(data)
    for i in range(validation_args['k']):
        train_X, train_Y, valid_X, valid_Y = validate.prepare_validation_dataset()
        model = run_model(args, train_X, train_Y)
        experiment = Experiment(f'{os.getcwd()}/args.yaml', model)
        score = experiment.validate(valid_X, valid_Y)
        avg_score = avg_score + score
    print(f"\nFinal mean score of the {model_name}:", avg_score)
    train_X, train_Y = validate.prepare_full_dataset()
    model = run_model(args, train_X, train_Y)
    experiment = Experiment(f'{os.getcwd()}/args.yaml', model)
    experiment.predict_and_save_csv(test_X, test_ids, avg_score)

def read_args():
    args = get_model_params()
    validation_args = get_validation_params()
    main(args, validation_args)

def set_root_dir():
    if not(os.getenv('ROOT_DIR')):
        os.environ['ROOT_DIR'] = os.getcwd()

if __name__ == "__main__":
    set_root_dir()
    read_args()