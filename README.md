# Predicting Molecular Bond Strengths with Topological Machine Learning

## What is it?
This repository showcases the core functionalities of
[giotto-learn](https://github.com/giotto-ai/giotto-learn), a Python library for
topological machine learning. The accompanying blog post can be found [here](https://medium.com/p/getting-started-with-giotto-learn-a-python-library-for-topological-machine-learning-451d88d2c4bc?source=email-22cbcf87e250--writer.postDistributed&sk=c5dffceb84a1108ea916f136f519258e).

This demo is based on the [Predicting Molecular
Properties](https://www.kaggle.com/c/champs-scalar-coupling/overview)
competition on Kaggle, where the task is to predict the bond strength between atoms in molecules.

## Getting started
The easiest way to get started is to create a conda environment as follows:
```
conda create python=3.7 --name molecule -y
conda activate molecule
pip install -r requirements.txt
```

## Results
The scoring function is described on Kaggle and is calculated as follows:
<div align="center">
<p><img src="data/figures/score.png?raw=true" width="450" /></p>
</div>

where:
* ![T](https://render.githubusercontent.com/render/math?math=T) is the number of coupling types
* ![n_t](https://render.githubusercontent.com/render/math?math=n_t) is the number of observations of type t
* ![y_i](https://render.githubusercontent.com/render/math?math=y_i) is the actual coupling value for this sample
* ![](https://render.githubusercontent.com/render/math?math=%5Chat%7By_i%7D) is the predicted coupling value for this sample


The figure below summarizes the results and gives a comparison of the results
with and without topological features.
<div align="center">
<p><img src="data/figures/results.png?raw=true" width="1200" /></p>
</div>


## External code
The following Kaggle notebooks were used for this project:

* For non-topological features: https://www.kaggle.com/robertburbidge/distance-features <br>
* For plotting molecules (but adapted): https://www.kaggle.com/mykolazotko/3d-visualization-of-molecules-with-plotly

## Some related publications
To get an introduction to the application of topological data analysis to
machine learning, see:
* An introduction to Topological Data Analysis: fundamental and practical aspects for data scientists: https://arxiv.org/pdf/1710.04019.pdf

The idea to use topological data analysis for predictions on molecules is not new. Below you can find some interesting papers related to this:

* Persistent-Homology-based Machine Learning and its Applications – A Survey: https://arxiv.org/abs/1811.00252 (esp. section 5)
* Representability of algebraic topology for biomolecules in machine learning based scoring and virtual screening: https://arxiv.org/pdf/1708.08135.pdf

The following papers were used to get some inspiration for the feature creation:
* The Ring of Algebraic Functions on Persistence Bar Codes: https://arxiv.org/pdf/1304.0530.pdf
* A topological approach for protein classification: https://arxiv.org/pdf/1510.00953.pdf
