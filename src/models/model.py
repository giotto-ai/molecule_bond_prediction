import fire
from xgboost import XGBRegressor, plot_feature_importance

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


def cv_model(X, y_train, y_test, features, n_fold=5, random_state=43, params=params):
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=random_state)

    model = XGBRegressor(**params)

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        X_train, X_valid = X_non_tda.iloc[train_index], X_non_tda.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        results.append(group_mean_log_mae(y_pred, y_valid, X_valid['type']))

    print('After {}-fold CV: Mean: '.format(n_fold), np.mean(results),
      'Std.:', np.std(results))


if __name__=="__main__":
    fire.Fire()
