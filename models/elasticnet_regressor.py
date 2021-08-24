from sklearn.linear_model import ElasticNet

class ElasticNetRegressor:
    model = None

    def __init__(self, model_args):
        self.model = ElasticNet(
            alpha = model_args['alpha'],
            normalize = model_args['normalize'],
            max_iter = model_args['max_iter'],
            tol = model_args['tolerance'],
            positive = model_args['is_coef_positive'],
            selection = model_args['coef_selection'],
            random_state = 42,
            l1_ratio = model_args['l1_ratio']
        )

    def get_model(self):
        return self.model