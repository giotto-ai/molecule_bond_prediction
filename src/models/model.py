import fire
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import KFold
import numpy as np

non_tda_columns = ['atom_index_0', 'atom_index_1', 'type', 'type_0', 'type_1', 'atom_0',
                   'x_0', 'y_0', 'z_0', 'atom_1', 'x_1', 'y_1', 'z_1', 'dist', 'dist_x',
                   'dist_y', 'dist_z', 'dist_to_type_mean', 'dist_to_type_0_mean',
                   'dist_to_type_1_mean', 'molecule_dist_mean_x', 'molecule_dist_std_x',
                   'molecule_dist_skew_x', 'molecule_dist_kurt_x', 'molecule_dist_mean_y',
                   'molecule_dist_std_y', 'molecule_dist_skew_y', 'molecule_dist_kurt_y',
                   'meanx', 'meany', 'meanz', 'dist_0tomean', 'dist_1tomean', 'meanxH',
                   'meanyH', 'meanzH', 'dist_0tomeanH', 'dist_1tomeanH', 'meanxC',
                   'meanyC', 'meanzC', 'dist_0tomeanC', 'dist_1tomeanC', 'meanxN',
                   'meanyN', 'meanzN', 'dist_0tomeanN', 'dist_1tomeanN', 'meanxO',
                   'meanyO', 'meanzO', 'dist_0tomeanO', 'dist_1tomeanO', 'meanxF',
                   'meanyF', 'meanzF', 'dist_0tomeanF', 'dist_1tomeanF', 'atom_count',
                   'atom_0l', 'x_0l', 'y_0l', 'z_0l', 'atom_0r', 'x_0r', 'y_0r', 'z_0r',
                   'dist_0l', 'dist_0r', 'atom_1l', 'x_1l', 'y_1l', 'z_1l', 'atom_1r',
                   'x_1r', 'y_1r', 'z_1r', 'dist_1l', 'dist_1r']


params = {'num_leaves': 128,
          'min_child_samples': 79,
          'objective': 'reg:squarederror',
          'max_depth': 13,
          'learning_rate': 0.1,
          "subsample_freq": 1,
          "subsample": 0.9,
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": 1,
          'reg_alpha': 0.1,
          'reg_lambda': 0.3,
          'colsample_bytree': 1.0}


def cv_model(X, y, features, n_fold=5, random_state=45245, params=params):
    """
    INPUT:
        X: pandas DataFrame with features
        y: pandas DataFrame (or Series object) with coupling constants as target values
        features: list of features to use
        n_fold: number of folds (int)
        random_state: for the KFold split
        params: parameter dictionary for XGBRegressor

    OUTPUT:
        results_mean: list of the scores for each type averaged over all folds
        results_details: list of all the scores as a list of lists
    """

    X = X[features]

    folds = KFold(n_splits=n_fold, shuffle=True, random_state=random_state)
    model = XGBRegressor(**params)
    results_mean = []
    results_details = []

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        scores = group_mean_log_mae(y_pred, y_valid, X_valid['type'])
        results_mean.append(scores[0])
        results_details.append(list(scores[1]))

    print('After {}-fold CV: Mean: '.format(n_fold), np.mean(results_mean), 'Std.:', np.std(results_mean))
    return results_mean, results_details


def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition:
        https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    INPUT:
        y_true: true value of coupling constant
        y_pred: predicted value of coupling constant
        types: bond types to consider
        floor: default = 1e-9
    OUTPUT:
        score: as described in the first link above
    """

    maes = (y_true-y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean(), np.log(maes)




if __name__=="__main__":
    fire.Fire()
