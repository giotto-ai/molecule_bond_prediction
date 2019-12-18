# Inspired by the following notebook: https://www.kaggle.com/mykolazotko/3d-visualization-of-molecules-with-plotly
# Imports
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sympy.geometry import Point3D


def plot_molecule(molecule_name, structures_df):
    """
    INPUT:
        molecule_name: name of the molecule from the structures DataFrame
        structures_df: structures DataFrame
    OUTPUT:
        fig: 3D plotly figure to visualize the chosen molecule
    """
    
    radius = dict(C=0.77, F=0.71, H=0.38, N=0.75, O=0.73)
    element_colors = dict(C='black', F='green', H='white', N='blue', O='red')
    molecule_df = structures_df[structures_df['molecule_name'] == molecule_name]
    x = molecule_df['x'].values
    y = molecule_df['y'].values
    z = molecule_df['z'].values
    elements = molecule_df['atom'].values
    r = [radius[e] for e in elements]
    coordinates = pd.DataFrame([x,y,z]).T

    def get_bonds():
        """Generates a set of bonds from atomic cartesian coordinates"""
        ids = np.arange(coordinates.shape[0])
        bonds = dict()
        coordinates_compare, radii_compare, ids_compare = coordinates, r, ids

        for _ in range(len(ids)):
            coordinates_compare = np.roll(coordinates_compare, -1, axis=0)
            radii_compare = np.roll(radii_compare, -1, axis=0)
            ids_compare = np.roll(ids_compare, -1, axis=0)
            distances = np.linalg.norm(coordinates - coordinates_compare, axis=1)
            bond_distances = (r + radii_compare) * 1.3
            mask = np.logical_and(distances > 0.1, distances < bond_distances)
            distances = distances.round(2)
            new_bonds = {frozenset([i, j]): dist for i, j, dist in zip(ids[mask], ids_compare[mask], distances[mask])}
            bonds.update(new_bonds)
        return bonds

    def get_bond_trace():
        bond_trace = go.Scatter3d(x=[], y=[], z=[], hoverinfo='none', mode='lines',
                             marker=dict(color='grey', size=7, opacity=1))
        for i,j in bonds.keys():
            bond_trace['x'] += (x[i], x[j], None)
            bond_trace['y'] += (y[i], y[j], None)
            bond_trace['z'] += (z[i], z[j], None)
        return bond_trace

    def get_atom_trace():
        """Creates an atom trace for the plot"""
        colors = [element_colors[element] for element in elements]
        markers = dict(color=colors, line=dict(color='lightgray', width=2), size=10, symbol='circle', opacity=0.8)
        trace = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=markers,
                             text=elements, name='', hoverlabel=dict(bgcolor=colors))
        return trace

    bonds = get_bonds()
    annotations_length = []
    for (i, j), dist in bonds.items():
        p_i, p_j = Point3D(coordinates.values[i]), Point3D(coordinates.values[j])
        p = p_i.midpoint(p_j)
        annotation = dict(text=dist, x=float(p.x), y=float(p.y), z=float(p.z), showarrow=False, yshift=15)
        annotations_length.append(annotation)
    data = [get_atom_trace(), get_bond_trace()]

    axis_params = dict(showgrid=False, showbackground=False, showticklabels=False, zeroline=False,
                       titlefont=dict(color='white'))
    layout = dict(scene=dict(xaxis=axis_params, yaxis=axis_params, zaxis=axis_params),
                  margin=dict(r=0, l=0, b=0, t=0), showlegend=False, annotations=[
                        go.layout.Annotation(
                            text='Molecule Name:<br>{}'.format(molecule_name),
                            align='left',
                            showarrow=False,
                            xref='paper',
                            yref='paper',
                            x=0.95,
                            y=0.95,
                            bordercolor='black',
                            borderwidth=1
                        )
                    ])

    fig = go.Figure(data=data, layout=layout)

    return fig
