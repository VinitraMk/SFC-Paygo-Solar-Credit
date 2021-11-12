from sklearn.ensemble import RandomForestRegressor

class RandomForest():
    model = None

    def __init__(self, model_args):
        self.model = RandomForestRegressor(
            n_estimators = model_args['n_estimators'],
            criterion = 'mse',
            max_depth = model_args['max_depth'],
            n_jobs = -1,
            ccp_alpha = model_args['ccp_alpha'],
            random_state = 42,
            min_samples_split = model_args['min_samples_split'],
            min_samples_leaf = model_args['min_samples_leaf'],
            bootstrap = model_args['bootstrap_samples'],
            min_impurity_split = model_args['min_impurity_split']
        )

    def get_model(self):
        return self.model