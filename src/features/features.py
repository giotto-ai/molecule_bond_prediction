import numpy as np
import pandas as pd

import fire

from scipy.stats import skew, kurtosis
from numpy.random import permutation
from sklearn import metrics
import lightgbm
from sklearn.preprocessing import LabelEncoder
import giotto.diagrams as diag
from giotto.homology import VietorisRipsPersistence

import time
from itertools import product
import networkx as nx
import pickle


################################################################################

def map_atom_info(df, atom_idx, structures):
    """
    Source: https://www.kaggle.com/robertburbidge/distance-features
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
    print(dist_mat)
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
    persistenceDiagram = VietorisRipsPersistence(metric='precomputed', max_edge_length=t,
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
    persistenceDiagram = VietorisRipsPersistence(metric='euclidean',
                                                homology_dimensions=homology_dimensions, n_jobs=1)
    persistenceDiagram.fit(m)
    X_diagrams = persistenceDiagram.transform(m)

    diagram_scaler = diag.Scaler()
    diagram_scaler.fit(X_diagrams)
    X_scaled = diagram_scaler.transform(X_diagrams)

    return X_scaled

# Binned features
def binned_features(X, homology_dim):
    """Compute binned features from the persistence diagram.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features, 3)
        Input data. Array of persistence diagrams, each a collection of
        triples [b, d, q] representing persistent topological features
        through their birth (b), death (d) and homology dimension (q).

    homology_dim : int
        Homology dimension to consider, must be contained in the persistence diagram

    Returns
    -------
    (count_birth, count_death, count_persistence) : tuple, shape (3),
        count_birth: ndarray, shape (n_samples, n_bins)
        count_death: ndarray, shape (n_samples, n_bins)
        count_persistence: ndarray, shape (n_samples, n_bins)

    """
    count_birth = []
    count_death = []
    count_persistence = []

    for i in range(X.shape[0]):
        max_length = np.max(X[i,:,1]) + 1e-6

        bins = np.linspace(0, max_length, 20)
        bins = [[b[0], b[1]] for b in zip(bins[:-1], bins[1:])]

        count_birth_diag = []
        count_death_diag = []
        count_persistence_diag = []

        for b in bins:
            mask_1 = X[i, :, 2]==homology_dim
            mask_2 = np.array(b[0]<=X[i, :, 0])
            mask_3 = np.array(X[i, :, 0]<b[1])
            mask_4 = np.array(b[0]<=X[i, :, 1])
            mask_5 = np.array(X[i, :, 1]<b[1])

            count_birth_diag.append(len(X[i][np.logical_and(np.logical_and(mask_1, mask_2), mask_3)]))
            count_death_diag.append(len(X[i][np.logical_and(np.logical_and(mask_1, mask_4), mask_5)]))
            count_persistence_diag.append(len(X[i][np.logical_and(np.logical_or(mask_3, mask_5), mask_1)]))

        count_birth.append(count_birth_diag)
        count_death.append(count_death_diag)
        count_persistence.append(count_persistence_diag)

    return count_birth, count_death, count_persistence


def area_under_Betti_curve(X_betti_curves, homology_dim):
    """Compute the area under the Betti curve for a given Betti curve

    Parameters
    ----------
    X_betti_curves : ndarray, shape (n_samples, n_homology_dimensions, n_values)
            Betti curves: one curve (represented as a one-dimensional array
            of integer values) per sample and per homology dimension seen
            in :meth:`fit`. Index i along axis 1 corresponds to the i-th
            homology dimension in :attr:`homology_dimensions_`.

    homology_dim : int
        Homology dimension to consider, must be contained in the persistence diagram

    Returns
    -------
    area : list, shape (n_samples)
        List of areas under the Betti curve for a given homology dimension.

    """
    area = []
    for n in range(X_betti_curves.shape[0]):
        area.append(np.trapz(X_betti_curves[n, homology_dim], dx=1))
    return area



def lrdist(df):
    # distance to nearest neighbours (by atom_index)
    # if there is no atom to the "left" (respectively "right") of the atom of interest,
    # then the distance is zero but this could be coded as NA
    # left and right indices - 0
    # Source: https://www.kaggle.com/robertburbidge/distance-features

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


def map_atom_info(df, atom_idx):
    # Helper function used for calculating non-TDA features
    # Source: https://www.kaggle.com/robertburbidge/distance-features
    file_folder = '../data/raw'
    structures = pd.read_csv(f'{file_folder}/structures.csv')
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


def mol_dist_stats(df):
    # statistics of dist by molecule
    # Source: https://www.kaggle.com/robertburbidge/distance-features
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


def reduce_mem_usage(df, verbose=True):
    # Helper function used for calculating non-TDA features
    # Source: https://www.kaggle.com/robertburbidge/distance-features and
    # https://www.kaggle.com/artgor/artgor-utils

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


# The execution of this cell takes a while
def get_bonds(molecule_name, structures):
    """
    Generates a set of bonds from atomic cartesian coordinates
    Source: https://www.kaggle.com/robertburbidge/distance-features
    """
    atomic_radii = dict(C=0.77, F=0.71, H=0.38, N=0.75, O=0.73)
    cpk_colors = dict(C='black', F='green', H='white', N='blue', O='red')

    molecule = structures[structures.molecule_name == molecule_name]
    coordinates = molecule[['x', 'y', 'z']].values
    x_coordinates = coordinates[:, 0]
    y_coordinates = coordinates[:, 1]
    z_coordinates = coordinates[:, 2]
    elements = molecule.atom.tolist()
    radii = [atomic_radii[element] for element in elements]
    ids = np.arange(coordinates.shape[0])
    bonds = dict()
    coordinates_compare, radii_compare, ids_compare = coordinates, radii, ids

    for _ in range(len(ids)):
        coordinates_compare = np.roll(coordinates_compare, -1, axis=0)
        radii_compare = np.roll(radii_compare, -1, axis=0)
        ids_compare = np.roll(ids_compare, -1, axis=0)
        distances = np.linalg.norm(coordinates - coordinates_compare, axis=1)
        bond_distances = (radii + radii_compare) * 1.3
        mask = np.logical_and(distances > 0.1, distances <  bond_distances)
        distances = distances.round(2)
        new_bonds = {frozenset([i, j]): dist for i, j, dist in zip(ids[mask], ids_compare[mask], distances[mask])}
        bonds.update(new_bonds)
    return [list(x) for x in list(bonds)]


def create_and_save_features(persistence_diagrams, molecule_selection, save_file=False, filename='all_features'):
    """
    Function to create TDA features given a list of persistence diagrams.
    INPUT:
        persistence_diagrams - list of persistence diagrams
        molecule_selection - list of molecule names
        save - boolean: if True: output will be saved to a pickle file
        filename - str: name of the pickle file to use if the file is saved

    OUTPUT:
        all_features - a dictionary of dictionaries where the key is the features
                       and the key of the subdictionary is the molecule name

    """
    num_rel_holes_0 = []
    num_rel_holes_1 = []
    num_rel_holes_2 = []
    num_holes_0 = []
    num_holes_1 = []
    num_holes_2 = []
    avg_lifetime_0 = []
    avg_lifetime_1 = []
    avg_lifetime_2 = []
    amplitude = []

    for m in range(len(molecule_selection)):
        num_rel_holes_0.append(num_relevant_holes(persistence_diagrams[m], homology_dim=0))
        num_rel_holes_1.append(num_relevant_holes(persistence_diagrams[m], homology_dim=1))
        num_rel_holes_2.append(num_relevant_holes(persistence_diagrams[m], homology_dim=2))
        num_holes_0.append(num_relevant_holes(persistence_diagrams[m], homology_dim=0, theta=0))
        num_holes_1.append(num_relevant_holes(persistence_diagrams[m], homology_dim=1, theta=0))
        num_holes_2.append(num_relevant_holes(persistence_diagrams[m], homology_dim=2, theta=0))
        avg_lifetime_0.append(average_lifetime(persistence_diagrams[m], homology_dim=0))
        avg_lifetime_1.append(average_lifetime(persistence_diagrams[m], homology_dim=1))
        avg_lifetime_2.append(average_lifetime(persistence_diagrams[m], homology_dim=2))
        amplitude.append(calculate_amplitude_feature(persistence_diagrams[m]))

    num_rel_holes_0 = np.array(num_rel_holes_0).flatten()
    num_rel_holes_1 = np.array(num_rel_holes_1).flatten()
    num_rel_holes_2 = np.array(num_rel_holes_2).flatten()
    num_holes_0 = np.array(num_holes_0).flatten()
    num_holes_1 = np.array(num_holes_1).flatten()
    num_holes_2 = np.array(num_holes_2).flatten()
    avg_lifetime_0 = np.array(avg_lifetime_0).flatten()
    avg_lifetime_1 = np.array(avg_lifetime_1).flatten()
    avg_lifetime_2 = np.array(avg_lifetime_2).flatten()
    amplitude = np.array(amplitude).flatten()

    # Make dictionaries
    num_rel_holes_0_dict = dict(zip(molecule_selection, num_rel_holes_0))
    num_rel_holes_1_dict = dict(zip(molecule_selection, num_rel_holes_1))
    num_rel_holes_2_dict = dict(zip(molecule_selection, num_rel_holes_2))
    num_holes_0_dict = dict(zip(molecule_selection, num_holes_0))
    num_holes_1_dict = dict(zip(molecule_selection, num_holes_1))
    num_holes_2_dict = dict(zip(molecule_selection, num_holes_2))
    avg_lifetime_0_dict = dict(zip(molecule_selection, avg_lifetime_0))
    avg_lifetime_1_dict = dict(zip(molecule_selection, avg_lifetime_1))
    avg_lifetime_2_dict = dict(zip(molecule_selection, avg_lifetime_2))
    amplitude_dict = dict(zip(molecule_selection, amplitude))

    all_features = {'num_rel_holes_0': num_rel_holes_0_dict,
                    'num_rel_holes_1': num_rel_holes_1_dict,
                    'num_rel_holes_2': num_rel_holes_2_dict,
                    'num_holes_0': num_holes_0_dict,
                    'num_holes_1': num_holes_1_dict,
                    'num_holes_2': num_holes_2_dict,
                    'avg_lifetime_0': avg_lifetime_0_dict,
                    'avg_lifetime_1': avg_lifetime_1_dict,
                    'avg_lifetime_2': avg_lifetime_2_dict,
                    'amplitude': amplitude_dict}

    if save_file==True:
        with open('tda_features', 'wb') as f:
            pickle.dump(all_features, f)

    return all_features


def create_non_TDA_features(directory):
    """
    Create features for the baseline model. Code reused from here:
    https://www.kaggle.com/robertburbidge/distance-features

    INPUT:
        directory - string: where to find train.csv and test.csv

    OUTPUT:
        train_dist.csv - CSV file with non-TDA features
        test_dist.csv - CSV file with non-TDA features
    """
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
               'min_child_samples': 15
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




if __name__ == '__main__()':
    fire.Fire()
