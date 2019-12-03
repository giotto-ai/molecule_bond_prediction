#!/usr/bin/env python
# coding: utf-8

# There are several kernels that use brute force feature engineering that achieve better LB scores than this one but some of the features are not easy to understand from a physical point-of-view. In this kernel I only use distances and atom types to derive features that are easy to visualize n one's head (I hope). Then I use LightGBM to predict the scalar coupling constant. Also, to estimate the LB score I use training and validation sets which do not contain the same molecules.


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../use_cases/machine_learning/molecules/data/"))
directory = '../use_cases/machine_learning/molecules/data/'

from scipy.stats import skew, kurtosis
from numpy.random import permutation
from sklearn import metrics
import lightgbm
from sklearn.preprocessing import LabelEncoder

################################################################################
# map atom_index_{0|1} to x_0, ..., z_1
def map_atom_info(df, atom_idx):
    df = pd.merge(df, structures, how='left',
                  left_on=['molecule_name', f'atom_index_{atom_idx}'],
                  right_on=['molecule_name', 'atom_index'])
    #
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df


# statistics of dist by molecule
def mol_dist_stats(df):
    dist_mean = df.groupby('molecule_name')['dist'].apply(np.mean).reset_index()
    dist_mean.rename({'dist': 'molecule_dist_mean'}, axis=1, inplace=True)
    df = pd.merge(df, dist_mean, how='left', on='molecule_name')
    dist_std = df.groupby('molecule_name')['dist'].apply(np.std).reset_index()
    dist_std.rename({'dist': 'molecule_dist_std'}, axis=1, inplace=True)
    df = pd.merge(df, dist_std, how='left', on='molecule_name')
    dist_skew = df.groupby('molecule_name')['dist'].apply(skew).reset_index()
    dist_skew.rename({'dist': 'molecule_dist_skew'}, axis=1, inplace=True)
    df = pd.merge(df, dist_skew, how='left', on='molecule_name')
    dist_kurt = df.groupby('molecule_name')['dist'].apply(kurtosis).reset_index()
    dist_kurt.rename({'dist': 'molecule_dist_kurt'}, axis=1, inplace=True)
    df = pd.merge(df, dist_kurt, how='left', on='molecule_name')
    return df


# https://www.kaggle.com/artgor/artgor-utils
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))
    return df


# distance to nearest neighbours (by atom_index)
# if there is no atom to the "left" (respectively "right") of the atom of interest, then the distance is zero but this could be coded as NA
def lrdist(df):
    # left and right indices - 0
    df['atom_index_0l'] = df['atom_index_0'].apply(lambda i: max(i - 1, 0))
    tmp = df[['atom_index_0', 'atom_count']]
    df['atom_index_0r'] = tmp.apply(lambda row: min(row['atom_index_0'] + 1, row['atom_count']), axis=1)
    # (x,y,z) of left and right indices
    df = map_atom_info(df, '0l')
    df = map_atom_info(df, '0r')
    # (x,y,z) for atom_0 and atom_1 as numpy arrays
    df_p_0l = df[['x_0l', 'y_0l', 'z_0l']].values
    df_p_0r = df[['x_0r', 'y_0r', 'z_0r']].values
    # distance between atom_0 and atom_1
    df_p_0 = df[['x_0', 'y_0', 'z_0']].values
    df['dist_0l'] = np.linalg.norm(df_p_0l - df_p_0, axis=1)
    df['dist_0r'] = np.linalg.norm(df_p_0r - df_p_0, axis=1)
    df.drop(['atom_index_0l', 'atom_index_0r'], axis=1, inplace=True)
    # left and right indices - 1
    df['atom_index_1l'] = df['atom_index_1'].apply(lambda i: max(i - 1, 0))
    tmp = df[['atom_index_1', 'atom_count']]
    df['atom_index_1r'] = tmp.apply(lambda row: min(row['atom_index_1'] + 1, row['atom_count']), axis=1)
    # (x,y,z) of left and right indices
    df = map_atom_info(df, '1l')
    df = map_atom_info(df, '1r')
    # (x,y,z) for atom_1 and atom_1 as numpy arrays
    df_p_1l = df[['x_1l', 'y_1l', 'z_1l']].values
    df_p_1r = df[['x_1r', 'y_1r', 'z_1r']].values
    # distance between atom_1 and atom_1
    df_p_1 = df[['x_1', 'y_1', 'z_1']].values
    df['dist_1l'] = np.linalg.norm(df_p_1l - df_p_1, axis=1)
    df['dist_1r'] = np.linalg.norm(df_p_1r - df_p_1, axis=1)
    df.drop(['atom_index_1l', 'atom_index_1r'], axis=1, inplace=True)
    return df


# evaluation metric for validation
# https://www.kaggle.com/abhishek/competition-metric
def metric(df, preds):
    df["prediction"] = preds
    maes = []
    for t in df.type.unique():
        y_true = df[df.type==t].scalar_coupling_constant.values
        y_pred = df[df.type==t].prediction.values
        mae = np.log(metrics.mean_absolute_error(y_true, y_pred))
        maes.append(mae)
    return np.mean(maes)




if '__name__' == __main__():
    # Datasets
    train = pd.read_csv(directory + 'train.csv')
    test = pd.read_csv(directory + 'test.csv')

    # split type
    train['type_0'] = train['type'].apply(lambda x: x[0])
    train['type_1'] = train['type'].apply(lambda x: x[1:])
    test['type_0'] = test['type'].apply(lambda x: x[0])
    test['type_1'] = test['type'].apply(lambda x: x[1:])

    # import coordinates data
    structures = pd.read_csv(directory + 'structures.csv')

    # get xyz data for each atom
    train = map_atom_info(train, 0)
    train = map_atom_info(train, 1)
    test = map_atom_info(test, 0)
    test = map_atom_info(test, 1)

    # (x,y,z) for atom_0 and atom_1 as numpy arrays
    train_p_0 = train[['x_0', 'y_0', 'z_0']].values
    train_p_1 = train[['x_1', 'y_1', 'z_1']].values
    test_p_0 = test[['x_0', 'y_0', 'z_0']].values
    test_p_1 = test[['x_1', 'y_1', 'z_1']].values
    print('Data import finished.')

    # distance between atom_0 and atom_1
    train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
    test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)

    # distances between atom_0 and atom_1 along each axis
    train['dist_x'] = (train['x_0'] - train['x_1']) ** 2
    test['dist_x'] = (test['x_0'] - test['x_1']) ** 2
    train['dist_y'] = (train['y_0'] - train['y_1']) ** 2
    test['dist_y'] = (test['y_0'] - test['y_1']) ** 2
    train['dist_z'] = (train['z_0'] - train['z_1']) ** 2
    test['dist_z'] = (test['z_0'] - test['z_1']) ** 2

    # distance/mean(distance) by type
    train['dist_to_type_mean'] = train['dist'] / train.groupby('type')['dist'].transform('mean')
    test['dist_to_type_mean'] = test['dist'] / test.groupby('type')['dist'].transform('mean')

    # distance/mean(distance) by type_0
    train['dist_to_type_0_mean'] = train['dist'] / train.groupby('type_0')['dist'].transform('mean')
    test['dist_to_type_0_mean'] = test['dist'] / test.groupby('type_0')['dist'].transform('mean')

    # distance/mean(distance) by type_1
    train['dist_to_type_1_mean'] = train['dist'] / train.groupby('type_1')['dist'].transform('mean')
    test['dist_to_type_1_mean'] = test['dist'] / test.groupby('type_1')['dist'].transform('mean')

    print('Basic feature creation finished.')

    # add distance statistics by molecule
    train = mol_dist_stats(train)
    test = mol_dist_stats(test)

    # distance to centre of molecule
    meanx = structures.groupby('molecule_name')['x'].apply(np.mean).reset_index()
    meanx.rename({'x': 'meanx'}, axis=1, inplace=True)
    train = pd.merge(train, meanx, how='left', on='molecule_name')
    test = pd.merge(test, meanx, how='left', on='molecule_name')

    meany = structures.groupby('molecule_name')['y'].apply(np.mean).reset_index()
    meany.rename({'y': 'meany'}, axis=1, inplace=True)
    train = pd.merge(train, meany, how='left', on='molecule_name')
    test = pd.merge(test, meany, how='left', on='molecule_name')

    meanz = structures.groupby('molecule_name')['z'].apply(np.mean).reset_index()
    meanz.rename({'z': 'meanz'}, axis=1, inplace=True)
    train = pd.merge(train, meanz, how='left', on='molecule_name')
    test = pd.merge(test, meanz, how='left', on='molecule_name')

    train_p_m = train[['meanx', 'meany', 'meanz']].values
    test_p_m = test[['meanx', 'meany', 'meanz']].values

    train['dist_0tomean'] = np.linalg.norm(train_p_0 - train_p_m, axis=1)
    train['dist_1tomean'] = np.linalg.norm(train_p_1 - train_p_m, axis=1)
    test['dist_0tomean'] = np.linalg.norm(test_p_0 - test_p_m, axis=1)
    test['dist_1tomean'] = np.linalg.norm(test_p_1 - test_p_m, axis=1)
    print('Distance feature created.')

    # distance to centre of each atom type in molecule
    # this could perhaps be weighted by properties of the respective atoms, such as no. electrons
    atoms = ['H', 'C', 'N', 'O', 'F']
    for atom in atoms:
        meanx = structures[structures['atom']==atom].groupby('molecule_name')['x'].apply(np.mean).reset_index()
        meanx.rename({'x': 'meanx' + atom}, axis=1, inplace=True)
        train = pd.merge(train, meanx, how='left', on='molecule_name')
        test = pd.merge(test, meanx, how='left', on='molecule_name')

        meany = structures[structures['atom']==atom].groupby('molecule_name')['y'].apply(np.mean).reset_index()
        meany.rename({'y': 'meany' + atom}, axis=1, inplace=True)
        train = pd.merge(train, meany, how='left', on='molecule_name')
        test = pd.merge(test, meany, how='left', on='molecule_name')

        meanz = structures[structures['atom']==atom].groupby('molecule_name')['z'].apply(np.mean).reset_index()
        meanz.rename({'z': 'meanz' + atom}, axis=1, inplace=True)
        train = pd.merge(train, meanz, how='left', on='molecule_name')
        test = pd.merge(test, meanz, how='left', on='molecule_name')

        train_p_m = train[['meanx' + atom, 'meany' + atom, 'meanz' + atom]].values
        test_p_m = test[['meanx' + atom, 'meany' + atom, 'meanz' + atom]].values

        train['dist_0tomean' + atom] = np.linalg.norm(train_p_0 - train_p_m, axis=1)
        train['dist_1tomean' + atom] = np.linalg.norm(train_p_1 - train_p_m, axis=1)
        test['dist_0tomean' + atom] = np.linalg.norm(test_p_0 - test_p_m, axis=1)
        test['dist_1tomean' + atom] = np.linalg.norm(test_p_1 - test_p_m, axis=1)
    print('Distance to the center features created.')

    # no. atoms in each molecule (not a distance feature, but needed below)
    atom_cnt = structures['molecule_name'].value_counts().reset_index(level=0)
    atom_cnt.rename({'index': 'molecule_name', 'molecule_name': 'atom_count'}, axis=1, inplace=True)
    train = pd.merge(train, atom_cnt, how='left', on='molecule_name')
    test = pd.merge(test, atom_cnt, how='left', on='molecule_name')
    del atom_cnt

    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    train = lrdist(train)
    test = lrdist(test)
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    print('Memory usage reduced.')


    # features for prediction (note we have picked up the atom types of the neighbours)
    pred_vars = [v for v in train.columns if v not in ['id', 'molecule_name', 'atom_0', 'atom_1',
                                                       'scalar_coupling_constant']]


    # encode categorical features as integers for LightGBM
    cat_feats = ['type', 'type_0', 'type_1', 'atom_0l', 'atom_0r', 'atom_1l', 'atom_1r']
    for f in cat_feats:
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))
    print('Categorical features encoded.')

    # train-val split by molecule_name (since test molecules are disjoint from train molecules)
    molecule_names = pd.DataFrame(permutation(train['molecule_name'].unique()),columns=['molecule_name'])
    nm = molecule_names.shape[0]
    ntrn = int(0.9*nm)
    nval = int(0.1*nm)

    tmp_train = pd.merge(train, molecule_names[0:ntrn], how='right', on='molecule_name')
    tmp_val = pd.merge(train, molecule_names[ntrn:nm], how='right', on='molecule_name')

    X_train = tmp_train[pred_vars]
    X_val = tmp_val[pred_vars]
    y_train = tmp_train['scalar_coupling_constant']
    y_val = tmp_val['scalar_coupling_constant']
    del tmp_train, tmp_val
    print('Training and test set created.')

    # heuristic parameters for LightGBM
    params = { 'objective': 'regression_l1',
               'learning_rate': 0.1,
               'num_leaves': 1023,
               'num_threads': -1,
               'bagging_fraction': 0.5,
               'bagging_freq': 1,
               'feature_fraction': 0.9,
               'lambda_l1': 10.0,
               'max_bin': 255,
               'min_child_samples': 15,
               }

    # data for LightGBM
    train_data = lightgbm.Dataset(X_train, label=y_train, categorical_feature=cat_feats)
    val_data = lightgbm.Dataset(X_val, label=y_val, categorical_feature=cat_feats)

    # training & validation of LightGBM
    model = lightgbm.train(params,
                           train_data,
                           valid_sets=[train_data, val_data], verbose_eval=500,
                           num_boost_round=4000,
                           early_stopping_rounds=100)
    print('LightGBM model trained.')

    # validation performance
    preds = model.predict(X_val)
    print('Score is:', metric(pd.concat([X_val, y_val], axis=1), preds))

    # save features for future use
    train.to_csv('train_dist.csv', index=False)
    test.to_csv('test_dist.csv', index=False)
    print('Data saved to disk.')
