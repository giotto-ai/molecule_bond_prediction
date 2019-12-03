import numpy as np

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