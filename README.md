# Estimating Heterogeneous Treatment Effects by Combining Weak Instruments and Observational Data

Two-stage framework that combines observational and IV data to reliably estimate conditional average treatment effects (CATEs), addressing both unobserved confounding and low compliance issues.

Replication code for **Estimating Heterogeneous Treatment Effects by Combining Weak Instruments and Observational Data**.


## Requirements

* [scikit-learn](https://scikit-learn.org/stable/) 
* [pytorch](https://pytorch.org/)
* [doubleml](https://github.com/DoubleML/doubleml-for-py)

## Replication Code for Paper

Use the following commands to replicate the figures from the "Estimating Heterogeneous Treatment Effects by Combining Weak Instruments and Observational Data" paper:

* For Figure 1, top row: ```python para_sims.py```
* For Figure 1, bottom row: ```python rep_sims.py```
* For Figure 2 & 3: ```python 401k.py```

## Interactive Experimentation

For interactive experimentation, we provide three Jupyter notebooks in the ```notebooks``` folder. Each notebook demonstrates the results of one simulation/run for the three considered settings: parametric extrapolation simulation, representation learning simulation, and the 401(k) participation treatment effect.