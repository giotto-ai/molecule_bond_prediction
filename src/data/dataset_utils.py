import pandas as pd
import fire


def get_selected_structures(molecule_selection):
    file_folder = '../data/raw'
    structures = pd.read_csv(f'{file_folder}/structures.csv')

    selected_structures = structures[structures['molecule_name']
                                 .isin(molecule_selection)].reset_index().drop('index', axis='columns')

    x_mean = selected_structures.groupby(['molecule_name', 'atom'])['x'].mean().reset_index().rename({'x': 'x_mean'}, axis='columns')
    y_mean = selected_structures.groupby(['molecule_name', 'atom'])['y'].mean().reset_index().rename({'y': 'y_mean'}, axis='columns')
    z_mean = selected_structures.groupby(['molecule_name', 'atom'])['z'].mean().reset_index().rename({'z': 'z_mean'}, axis='columns')

    selected_structures = selected_structures.merge(x_mean, on=['molecule_name', 'atom'])
    selected_structures = selected_structures.merge(y_mean, on=['molecule_name', 'atom'])
    selected_structures = selected_structures.merge(z_mean, on=['molecule_name', 'atom'])
    selected_structures['x_new'] = selected_structures['x'] - selected_structures['x_mean']
    selected_structures['y_new'] = selected_structures['y'] - selected_structures['y_mean']
    selected_structures['z_new'] = selected_structures['z'] - selected_structures['z_mean']

    return selected_structures


if __name__=='__main__':
    fire.Fire()
