from sklearn.linear_model import Lasso

class LassoRegressor:
    model = None

    def __init__(self, model_args):
        self.model = Lasso(
            alpha = model_args['alpha'],
            normalize = model_args['normalize'],
            max_iter = model_args['max_iter'],
            tol = model_args['tolerance'],
            positive = model_args['is_coef_positive'],
            selection = model_args['coef_selection'],
            random_state = 42
        )

    def get_model(self):
        return self.model