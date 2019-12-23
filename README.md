## Description of this repository
The goal of this repository is to demonstrate the core funcitionalities of giotto-learn, an open-source topological data analysis library. 

This demo is based on the Kaggle competition 'Predicting Molecular Properties' (Link: https://www.kaggle.com/c/champs-scalar-coupling/overview) where the task is to predict the bond strength between atoms in molecules.

The repository contains the following:

* notebooks: main directory with the notebooks for feature creation and model fitting/testing
* code: directory with additional code (e.g. to create the non-TDA features)
* data: directory to store the data from Kaggle (download from here: https://www.kaggle.com/c/champs-scalar-coupling/data)
* README.md

## Results
The scoring function is described on Kaggle and calculated as follows:
<div align="center">
<p><img src="data/figures/score.png?raw=true" width="450" /></p>
</div>

where:
* ![T](https://render.githubusercontent.com/render/math?math=T) is the number of coupling types
* ![n_t](https://render.githubusercontent.com/render/math?math=n_t) is the number of observations of type t
* ![y_i](https://render.githubusercontent.com/render/math?math=y_i) is the actual coupling value for this sample
* ![](https://render.githubusercontent.com/render/math?math=%5Chat%7By_i%7D) is the predicted coupling value for this sample


The figure below summarizes the results and gives a comparison of the results with and without TDA.
<div align="center">
<p><img src="data/figures/results.png?raw=true" width="1200" /></p>
</div>


## External code
The following Kaggle notebooks were used for this project:

* For non-TDA features: https://www.kaggle.com/robertburbidge/distance-features <br>
* For plotting molecules (but adapted): https://www.kaggle.com/mykolazotko/3d-visualization-of-molecules-with-plotly

## How to get started
The easiest way to get started is to create an environment like this:
'''
conda create python=3.7 --name molecule
pip install -r requirements.txt
'''

## Some related publications
To get an introduction to topological data analysis:
* An introduction to Topological Data Analysis: fundamental and practical aspects for data scientists: https://arxiv.org/pdf/1710.04019.pdf

The idea to use topological data analysis for predictions on molecules is not new. Below you can find some interesting papers related to this:

* Persistent-Homology-based Machine Learning and its Applications – A Survey: https://arxiv.org/abs/1811.00252 (esp. section 5)
* Representability of algebraic topology for biomolecules in machine learning based scoring and virtual screening: https://arxiv.org/pdf/1708.08135.pdf

The following papers were used to get some inspiration for the feature creation:
* The Ring of Algebraic Functions on Persistence Bar Codes: https://arxiv.org/pdf/1304.0530.pdf
* A topological approach for protein classification: https://arxiv.org/pdf/1510.00953.pdf
