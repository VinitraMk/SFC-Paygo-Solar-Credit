from sklearn.neighbors import KNeighborsRegressor

class KNNRegressor:
    model = None

    def __init__(self, model_args):
        self.model = KNeighborsRegressor(
            n_neighbors=model_args['n'],
            weights=model_args['weights'],
            algorithm=model_args['algorithm'],
            leaf_size=model_args['leaf_size'],
            p=model_args['p'],
            metric=model_args['metric'],
            n_jobs=-1
        )

    def get_model(self):
        return self.model
