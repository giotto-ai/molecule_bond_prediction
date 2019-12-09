# Data handling
import pandas as pd
import numpy as np
import networkx as nx

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
import giotto as go
import giotto.time_series as ts
import giotto.graphs as gr
import giotto.diagrams as diag
import giotto.homology as hl

# Plotting functions
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance

# Others
import os
from itertools import product
import time
import random

################################################################################

def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition:
        https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true-y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()


def map_atom_info(df, atom_idx, structures):
    """
    INPUT:
        df: DataFrame of train or test data
        atom_idx: int, either 0 or 1
        structures: structures file
    OUTPUT:
        df: New DataFrame of train or test data with structrue information (x,y,z)
    """
    df = pd.merge(df, structures, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])

    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df


def num_relevant_holes(X_scaled, homology_dim, theta=0.5):
    """
    INPUT:
        X_scaled: scaled persistence diagrams, numpy array
        homology_dim: dimension of the homology to consider, integer
        theta: value between 0 and 1 to be used to calculate the threshold, float

    OUTPUT:
        n_rel_holes: list of the number of relevant holes in each time window
    """

    n_rel_holes = []

    for i in range(X_scaled.shape[0]):
        persistence_table = pd.DataFrame(X_scaled[i], columns=['birth', 'death', 'homology'])
        persistence_table['lifetime'] = persistence_table['death'] - persistence_table['birth']
        threshold = persistence_table[persistence_table['homology'] == homology_dim]['lifetime'].max() * theta
        n_rel_holes.append(persistence_table[(persistence_table['lifetime'] > threshold)
                                             & (persistence_table['homology'] == homology_dim)].shape[0])
    return n_rel_holes


def average_lifetime(X_scaled, homology_dim):
    """
    INPUT:
        X_scaled: scaled persistence diagrams, numpy array
        homology_dim: dimension of the homology to consider, integer

    OUTPUT:
        avg_lifetime_list: list of average lifetime for each time window
    """

    avg_lifetime_list = []

    for i in range(X_scaled.shape[0]):
        persistence_table = pd.DataFrame(X_scaled[i], columns=['birth', 'death', 'homology'])
        persistence_table['lifetime'] = persistence_table['death'] - persistence_table['birth']
        avg_lifetime_list.append(persistence_table[persistence_table['homology']
                                                   == homology_dim]['lifetime'].mean())

    return avg_lifetime_list


def calculate_amplitude_feature(X_scaled, metric='wasserstein', order=2):
    """
    INPUT:
        X_scaled: scaled persistence diagrams, numpy array
        metric: Either 'wasserstein' (default), 'landscape', 'betti', 'bottleneck' or 'heat'
        order: integer

    OUTPUT:
        amplitude: vector with the values for the amplitude feature
    """

    amplitude = diag.Amplitude(metric=metric, order=order)
    return amplitude.fit_transform(X_scaled)


def graph_from_molecule(molecule, source='atom_index_0', target='atom_index_1'):
    """
    INPUT:
        molecule: DataFrame of molecule as a subset of rows of train/test data (incl x,y,z coords etc.)

    OUTPUT:
        graph: networkx object, where edges are given by bonds in molecule
    """
    graph = nx.from_pandas_edgelist(molecule, source=source, target=target)
    return graph


def graph_from_molecule_weighted(molecule):
    """
    INPUT:
        molecule: DataFrame of molecule as a subset of rows of train/test data (incl x,y,z coords etc.)

    OUTPUT:
        graph: networkx object, where edges are given by bonds in molecule and with weights
    """
    edges = molecule[['atom_index_0', 'atom_index_1']].values
    types = molecule[['type_0', 'type_1']].values
    def weight(t):
        if list(t)==[0,1]:
            return 2
        elif list(t)==[0,2]:
            return 1
        else:
            return 5

    G = nx.Graph()
    for e, t in zip(edges, types):
        G.add_edge(e[0], e[1], weight=weight(np.sort(t)))
    return G


def calculate_dist(graph, node_tuple):
    """
    INPUT:
        graph: networkx object
        node_tuple: source and target node to consider
    OUTPUT:
        dist: calculate shortest path between two nodes in a (unweighted) graph
    """
    if not (node_tuple[1] in nx.algorithms.descendants(graph, node_tuple[0])):
        return 1000
    else:
        return nx.shortest_path_length(graph, node_tuple[0], node_tuple[1])


def computing_distance_matrix(graph):
    """
    INPUT:
        graph: networkx graph object
    OUTPUT:
        dist_matrix: distance matrix (np.array)
    """
    nodes = np.unique(list(graph.nodes))
    l = (list(i) for i in product(nodes, nodes) if tuple(reversed(i)) >= tuple(i))
    dist_list = list(map(lambda t: calculate_dist(graph, t), l))

    l_new = np.array(list(list(i) for i in product(list(range(len(nodes))), list(range(len(nodes)))) if tuple(reversed(i)) >= tuple(i)))

    row, column = zip(*l_new)
    dist_mat = np.zeros((len(nodes), len(nodes)))
    dist_mat[row, column] = dist_list
    dist_mat[column, row] = dist_list
    dist_mat[range(len(dist_mat)), range(len(dist_mat))] = 0.
    return dist_mat


def computing_persistence_diagram(G, t=np.inf, homologyDimensions = (0, 1, 2)):
    """
    INPUT:
        G - a graph
        t - persistence threshold
        homologyDimensions - homology dimensions to consider
    OUTPUT:
        pd - persistence diagram calculated by Giotto
    """
    start = time.time()
    dist_mat = computing_distance_matrix(G)
    #dist_mat = np.array(nx.floyd_warshall_numpy(G))
    end = time.time()
    #print('Computing distance matrix time:', end - start)

    start = time.time()
    persistenceDiagram = hl.VietorisRipsPersistence(metric='precomputed', max_edge_length=t,
                                                    homology_dimensions=homologyDimensions,
                                                    n_jobs=-1)
    Diagrams = persistenceDiagram.fit_transform(dist_mat.reshape(1, dist_mat.shape[0], dist_mat.shape[1]))
    end = time.time()
    #print('Computing TDA:', end - start)
    return Diagrams


def get_pd_from_molecule(molecule_name, structures):
    """
    INPUT:
        molecule_name: name of the molecule as given in the structres file
        structures: structures file containing information (x, y, z coordinates) for all molecules

    OUTPUT:
        X_scaled: scaled persistence diagrams
    """
    m = structures[structures['molecule_name'] == molecule_name][['x', 'y', 'z']].to_numpy()
    m = m.reshape((1, m.shape[0], m.shape[1]))
    homology_dimensions = [0, 1, 2]
    persistenceDiagram = hl.VietorisRipsPersistence(metric='euclidean',
                                                homology_dimensions=homology_dimensions, n_jobs=1)
    persistenceDiagram.fit(m)
    X_diagrams = persistenceDiagram.transform(m)

    diagram_scaler = diag.Scaler()
    diagram_scaler.fit(X_diagrams)
    X_scaled = diagram_scaler.transform(X_diagrams)

    return X_scaled




if __name__=='__main__':
    """
    Note: most of the non-TDA features are from this notebook on Kaggle:
        https://www.kaggle.com/artgor/molecular-properties-eda-and-models
    """
    #Define paths
    file_folder = 'data/champs-scalar-coupling' if 'champs-scalar-coupling' in os.listdir('data/') else 'data'
    os.listdir(file_folder)

    #Import files
    train = pd.read_csv(f'{file_folder}/train.csv')
    test = pd.read_csv(f'{file_folder}/test.csv')
    sub = pd.read_csv(f'{file_folder}/sample_submission.csv')
    structures = pd.read_csv(f'{file_folder}/structures.csv')

    #Preprocessing step: get info about atom 1 and atom 2 in a bond
    train = map_atom_info(train, 0, structures)
    train = map_atom_info(train, 1, structures)
    test = map_atom_info(test, 0, structures)
    test = map_atom_info(test, 1, structures)

    #Create files for feature creation below
    train_p_0 = train[['x_0', 'y_0', 'z_0']].values
    train_p_1 = train[['x_1', 'y_1', 'z_1']].values
    test_p_0 = test[['x_0', 'y_0', 'z_0']].values
    test_p_1 = test[['x_1', 'y_1', 'z_1']].values

    #Create distance features
    train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
    test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
    train['dist_x'] = (train['x_0'] - train['x_1']) ** 2
    test['dist_x'] = (test['x_0'] - test['x_1']) ** 2
    train['dist_y'] = (train['y_0'] - train['y_1']) ** 2
    test['dist_y'] = (test['y_0'] - test['y_1']) ** 2
    train['dist_z'] = (train['z_0'] - train['z_1']) ** 2
    test['dist_z'] = (test['z_0'] - test['z_1']) ** 2

    #Create type features, i.e. what's the type of atom 0, atom 1
    train['type_0'] = train['type'].apply(lambda x: x[0])
    test['type_0'] = test['type'].apply(lambda x: x[0])
    train['type_1'] = train['type'].apply(lambda x: x[1:])
    test['type_1'] = test['type'].apply(lambda x: x[1:])

    #Create more features related to the distance
    train['dist_to_type_mean'] = train['dist'] / train.groupby('type')['dist'].transform('mean')
    test['dist_to_type_mean'] = test['dist'] / test.groupby('type')['dist'].transform('mean')

    train['dist_to_type_0_mean'] = train['dist'] / train.groupby('type_0')['dist'].transform('mean')
    test['dist_to_type_0_mean'] = test['dist'] / test.groupby('type_0')['dist'].transform('mean')

    train['dist_to_type_1_mean'] = train['dist'] / train.groupby('type_1')['dist'].transform('mean')
    test['dist_to_type_1_mean'] = test['dist'] / test.groupby('type_1')['dist'].transform('mean')

    train[f'molecule_type_dist_mean'] = train.groupby(['molecule_name', 'type'])['dist'].transform('mean')
    test[f'molecule_type_dist_mean'] = test.groupby(['molecule_name', 'type'])['dist'].transform('mean')

    #Encode categorical labels
    for f in ['atom_0', 'atom_1', 'type_0', 'type_1', 'type']:
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

    max_number_samples = 10000
    random.seed(43)
    idx_sample = random.sample(range(len(train)), max_number_samples)

    X = train.drop(['id', 'molecule_name', 'scalar_coupling_constant'], axis=1).loc[idx_sample]
    y = train['scalar_coupling_constant'].loc[idx_sample]

    X.to_pickle('X.pickle')
    y.to_pickle('y.pickle')

    persistence_diagrams = []
    for i, m in zip(range(len(list(train.loc[idx_sample]['molecule_name'].unique()))), list(train.loc[idx_sample]['molecule_name'].unique())):
        persistence_diagrams.append(get_pd_from_molecule(m, structures))

    rel_holes_1 = [num_relevant_holes(x, homology_dim=1) for x in persistence_diagrams]
    rel_holes_1 = np.array(rel_holes_1).flatten()

    life_time_0 = [average_lifetime(x, homology_dim=0) for x in persistence_diagrams]
    life_time_0 = np.array(life_time_0).flatten()

    life_time_1 = [average_lifetime(x, homology_dim=1) for x in persistence_diagrams]
    life_time_1 = np.array(life_time_1).flatten()

    amp_0 = [calculate_amplitude_feature(x) for x in persistence_diagrams]
    amp_0 = np.array(amp_0).flatten()


    # Molecule as a graph dim=0
    rel_holes_0_molecule = []
    avg_lifetime_0_molecule = []
    amp_0_molecule = []

    for i, m in zip(range(len(list(train.loc[idx_sample]['molecule_name'].unique()))), list(train.loc[idx_sample]['molecule_name'].unique())):
        persistenceDiagram = computing_persistence_diagram(graph_from_molecule(train[train['molecule_name']==m]))
        rel_holes_0_molecule.append(num_relevant_holes(persistenceDiagram, homology_dim=0))
        avg_lifetime_0_molecule.append(average_lifetime(persistenceDiagram, homology_dim=0))
        amp_0_molecule.append(calculate_amplitude_feature(persistenceDiagram))


    # Molecule as a graph dim=1
    rel_holes_1_molecule = []
    avg_lifetime_1_molecule = []
    amp_1_molecule = []

    for i, m in zip(range(len(list(train.loc[idx_sample]['molecule_name'].unique()))), list(train.loc[idx_sample]['molecule_name'].unique())):
        persistenceDiagram = computing_persistence_diagram(graph_from_molecule(train[train['molecule_name']==m]))
        rel_holes_1_molecule.append(num_relevant_holes(persistenceDiagram, homology_dim=1))
        avg_lifetime_1_molecule.append(average_lifetime(persistenceDiagram, homology_dim=1))
        amp_1_molecule.append(calculate_amplitude_feature(persistenceDiagram))


    num_holes_1_dict = dict(zip(list(train.loc[idx_sample]['molecule_name'].unique()), np.array(rel_holes_1).flatten()))
    tda_feature_rel_1 = []
    for i in train['molecule_name'].loc[idx_sample]:
        tda_feature_rel_1.append(num_holes_1_dict[i])

    time_1_dict = dict(zip(list(train.loc[idx_sample]['molecule_name'].unique()), np.array(life_time_1).flatten()))
    tda_feature_time_1 = []
    for i in train['molecule_name'].loc[idx_sample]:
        tda_feature_time_1.append(time_1_dict[i])

    rel_holes_1_dict = dict(zip(list(train.loc[idx_sample]['molecule_name'].unique()), np.array(rel_holes_1_molecule).flatten()))
    graph_holes_1 = []
    for i in train['molecule_name'].loc[idx_sample]:
        graph_holes_1.append(rel_holes_1_dict[i])

    avg_lifetime_1_dict = dict(zip(list(train.loc[idx_sample]['molecule_name'].unique()), np.array(avg_lifetime_1_molecule).flatten()))
    graph_lifetime_1 = []
    for i in train['molecule_name'].loc[idx_sample]:
        graph_lifetime_1.append(avg_lifetime_1_dict[i])

    amp_1_dict = dict(zip(list(train.loc[idx_sample]['molecule_name'].unique()), np.array(amp_1_molecule).flatten()))
    graph_amp_1 = []
    for i in train['molecule_name'].loc[idx_sample]:
        graph_amp_1.append(amp_1_dict[i])

    time_0_dict = dict(zip(list(train.loc[idx_sample]['molecule_name'].unique()), np.array(life_time_0).flatten()))
    tda_feature_time_0 = []
    for i in train['molecule_name'].loc[idx_sample]:
        tda_feature_time_0.append(time_0_dict[i])

    amp_0_dict = dict(zip(list(train.loc[idx_sample]['molecule_name'].unique()), np.array(amp_0).flatten()))
    tda_feature_amp_0 = []
    for i in train['molecule_name'].loc[idx_sample]:
        tda_feature_amp_0.append(amp_0_dict[i])

    rel_holes_0_dict = dict(zip(list(train.loc[idx_sample]['molecule_name'].unique()), np.array(rel_holes_0_molecule).flatten()))
    graph_holes_0 = []
    for i in train['molecule_name'].loc[idx_sample]:
        graph_holes_0.append(rel_holes_0_dict[i])

    X_1 = train.drop(['id', 'molecule_name', 'scalar_coupling_constant', 'atom_index_0', 'atom_index_1'], axis=1).loc[idx_sample]
    y_1 = train['scalar_coupling_constant'].loc[idx_sample]

    X_1['num_tda_1'] = tda_feature_rel_1
    X_1['time_tda_0'] = tda_feature_time_0
    X_1['time_tda_1'] = tda_feature_time_1
    X_1['amp_tda'] = tda_feature_amp_0
    X_1['graph_holes_0'] = graph_holes_0
    X_1['graph_holes_1'] = graph_holes_1
    X_1['graph_lifetime_1'] = graph_lifetime_1
    X_1['graph_amplitude_1'] = graph_amp_1

    X_1.to_pickle('X_1_2811.pickle')
    y_1.to_pickle('y_1_2811.pickle')
