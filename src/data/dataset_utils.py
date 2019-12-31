import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import fire


def get_selected_structures(molecule_selection):
    """
    INPUT:
        molecule_selection: list of molecule names
    OUTPUT:
        selected_structures: part of structures DataFrame with only the selected
                             molecules
    """

    file_folder = "data/raw"
    structures = pd.read_csv(f"{file_folder}/structures.csv.zip", compression="zip")

    selected_structures = (
        structures[structures["molecule_name"].isin(molecule_selection)]
        .reset_index()
        .drop("index", axis="columns")
    )

    x_mean = (
        selected_structures.groupby(["molecule_name", "atom"])["x"]
        .mean()
        .reset_index()
        .rename({"x": "x_mean"}, axis="columns")
    )
    y_mean = (
        selected_structures.groupby(["molecule_name", "atom"])["y"]
        .mean()
        .reset_index()
        .rename({"y": "y_mean"}, axis="columns")
    )
    z_mean = (
        selected_structures.groupby(["molecule_name", "atom"])["z"]
        .mean()
        .reset_index()
        .rename({"z": "z_mean"}, axis="columns")
    )

    selected_structures = selected_structures.merge(
        x_mean, on=["molecule_name", "atom"]
    )
    selected_structures = selected_structures.merge(
        y_mean, on=["molecule_name", "atom"]
    )
    selected_structures = selected_structures.merge(
        z_mean, on=["molecule_name", "atom"]
    )
    selected_structures["x_new"] = (
        selected_structures["x"] - selected_structures["x_mean"]
    )
    selected_structures["y_new"] = (
        selected_structures["y"] - selected_structures["y_mean"]
    )
    selected_structures["z_new"] = (
        selected_structures["z"] - selected_structures["z_mean"]
    )

    return selected_structures


def get_number_of_atoms(g):
    # top 100 molecules with the most number of atoms
    return len(g)


def create_non_TDA_pickle(directory="../data/raw", n_largest=100, save=False):
    """
    INPUT:
        directory: relative path to
        n_largest: how many molecules to include starting from the largest, default: 100
        save: if True: a pickle file of the ouput will be generated
    OUTPUT:
        X: non_TDA features, pandas DataFrame object
        y: target values, pandas Series object
        molecules: array of molecule names
    """
    # Data import
    file_folder = directory
    train = pd.read_csv(f"{file_folder}/train.csv")
    structures = pd.read_csv(f"{file_folder}/structures.csv")

    molecule_selection = (
        train.groupby(by="molecule_name")
        .apply(get_number_of_atoms)
        .nlargest(n_largest)
        .index
    )

    selection = train[train["molecule_name"].isin(molecule_selection)].reset_index()
    molecules = selection["molecule_name"].values

    # Import classical features
    train_dist = pd.read_csv("../data/processed/train_dist.csv")

    X = train_dist[train_dist["molecule_name"].isin(molecule_selection)].reset_index()
    y = X["scalar_coupling_constant"]
    molecules = X["molecule_name"].values
    X.drop(
        ["id", "molecule_name", "scalar_coupling_constant"],
        axis="columns",
        inplace=True,
    )

    lenc = LabelEncoder()
    lenc.fit(X["atom_1"])
    X["atom_0"] = lenc.transform(X["atom_0"])
    X["atom_1"] = lenc.transform(X["atom_1"])

    X.drop("index", axis="columns", inplace=True)
    if save == True:
        with open("../data/processed/non_TDA.pickle", "wb") as f:
            pickle.dump([X, y, molecule_selection, structures, molecules], f)

    return X, y, molecule_selection, structures, molecules


if __name__ == "__main__":
    fire.Fire()
