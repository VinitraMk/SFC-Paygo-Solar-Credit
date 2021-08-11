from sklearn.linear_model import LinearRegression
from helper.utils import get_model_params

class LinearRegressor:
    model = None

    def __init__(self):
        model_args = get_model_params()
        self.model = LinearRegression(n_jobs=-1, normalize = model_args['normalize'])

    def get_model(self):
        return self.model
