import enum

class Model(str, enum.Enum):
    LINEAR = 'linear'
    KNN = 'knn'
    LINEAR_RIDGE = 'linear_ridge'
    LINEAR_LASSO = 'linear_lasso'
    LINEAR_ELASTICNET = 'linear_elasticnet'
    DECISION_TREE = 'decision_tree'
    RANDOM_FOREST = 'random_forest'