from sklearn.linear_model import LinearRegression

class LinearRegressor:
    model = None

    def __init__(self, model_args):
        self.model = LinearRegression(n_jobs=-1, normalize = model_args['normalize'])

    def get_model(self):
        return self.model
