from sklearn.linear_model import Ridge
from helper.utils import get_model_params

class RidgeRegressor:
    model = None

    def __init__(self, model_args):
        self.model = Ridge(
            alpha = model_args['alpha'],
            normalize = model_args['normalize'],
            max_iter = model_args['max_iter'],
            solver = model_args['solver'],
            tol = model_args['tolerance'],
            random_state = 42
        )

    def get_model(self):
        return self.model