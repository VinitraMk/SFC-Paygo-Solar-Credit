from sklearn.tree import DecisionTreeRegressor

class DecisionTree():
    model = None

    def __init__(self, model_args):
        self.model = DecisionTreeRegressor(
            criterion = 'mse',
            splitter = 'best',
            max_depth = model_args['max_depth'],
            min_samples_split = model_args['min_samples_split'],
            min_samples_leaf = model_args['min_samples_leaf'],
            max_features = model_args['max_features'],
            random_state = 42,
            max_leaf_nodes = model_args['max_leaf_nodes'],
            min_impurity_split = model_args['min_impurity_split'],
            ccp_alpha = model_args['ccp_alpha']
        )

    def get_model(self):
        return self.model