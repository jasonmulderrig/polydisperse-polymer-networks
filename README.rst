#######################################################
Polydisperse polymer networks with irregular topologies
#######################################################

A repository of research codes that micromechanically model polydisperse polymer networks via a discrete polymer network modeling approach founded upon the statistical mechanics of polydisperse/irregular cross-link structures.

*****
Setup
*****

Once the contents of the repository have been cloned or downloaded, the Python virtual environment associated with the project needs to be installed. The installation of this Python virtual environment and some essential packages is handled by the ``virtual-environment-install-master.sh`` Bash script. Before running the script, make sure to change the ``VENV_PATH`` parameter to comply with your personal filetree structure. Alternatively, a Conda environment can be installed with the required packages. All required packages are listed in the ``requirements.txt`` file.

********
Packages
********

This work is built upon the Python scientific computing stack, e.g., `Numpy <https://numpy.org/>`_, `Scipy <https://scipy.org/>`_, `Matplotlib <https://matplotlib.org/>`_, and `scikit-learn <https://scikit-learn.org/stable/>`_. The popular Python graph/network analysis package `NetworkX <https://networkx.org/en/>`_ is also utilized. Parameter configuration management is handled via the `Hydra <https://hydra.cc/>`_ package.

We use the `pylimer-tools <https://genietim.github.io/pylimer-tools/>`_ package to computationally synthesize polydisperse polymer networks and capture the microstructural topology of these networks as graphs for further (mechanics) analysis. We encourage other researchers who are interested in computationally synthesizing and analyzing the microstructural topology of polydisperse polymer networks to adopt and use the pylimer-tools package.

Note that the dependency requirements for the `pylimer-tools <https://genietim.github.io/pylimer-tools/>`_ package must also be installed in your environment (e.g., `Pint <https://pint.readthedocs.io/en/stable/index.html>`_).

In this work, we use NetworkX exclusively for geometric graph isomorphism analysis of polydisperse cross-link structures. We do not use NetworkX any further for the following reasons:

1. NetworkX is written in pure Python, which ultimately means that analyzing large network structures requires long computational runtimes. (On the contrary, `igraph <https://igraph.org/>`_ and `rustworkx <https://www.rustworkx.org/>`_ — two Python graph analysis packages respectively written on top of C++ and Rust backends — are able to analyze large network structures much more efficiently that NetworkX can.)
2. In this work, we develop and utilize a bespoke graph data structure — the sparse adjacency array — to represent and analyze the microstructural topology of (computationally-synthesized) polydisperse polymer networks. As far as we are aware, NetworkX is only capable of representing graphs via the standard adjacency matrix and edge list graph data structures. Because of this, we implement the sparse adjacency array (and associated analysis routines) in Numpy. Since Numpy is written on top of a C++ backend, the sparse adjacency array-based network analysis routines are quite computationally efficient.

We are sincerely grateful to the contributers of each of the aforementioned open-sourced Python packages. Without the state-of-the-art capabilities these open-sourced packages provide, this work would not have been possible.

*********
Structure
*********

The core functions in this repository are modularly distributed in Python files that reside in the following source directories:

* ``src/descriptors``
* ``src/file_io``
* ``src/helpers``
* ``src/spherical_quadrature``

The core functions can then be called upon in Python files for various tasks.

An example ``master template`` Python file that analyzes polydisperse polymer networks with irregular cross-links can be found in the ``clnk_rves`` directory (``clnk_rves/clnk_rves.py``).

Analysis of polydisperse polymer networks with irregular cross-links formed by inextensible Gaussian chains and inextensible Kuhn-Grün chains respectively reside in the following directories:

* ``polydisperse_inext_gaussian_fjc_networks_clnk_rves``
* ``polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves``

Analysis of polydisperse polymer networks with irregular cross-links formed by composite arbitrarily-inextensible freely jointed chains (cuFJCs) reside in the ``polydisperse_cufjc_networks_clnk_rves`` directory. See the following manuscript which details the cuFJC model: *Jason Mulderrig, Brandon Talamini, and Nikolaos Bouklas, A statistical mechanics framework for polymer chain scission, based on the concepts of distorted bond potential and asymptotic matching, `Journal of the Mechanics and Physics of Solids 174, 105244 (2023) `(DOI link) <https://doi.org/10.1016/j.jmps.2023.105244>`_*

Analysis of bimodal PDMS polymer networks reside in the ``bimodal_pdms_networks`` directory, and mechanics data associated with these networks resides in the ``src/data`` source directory.

Parameter configuration settings for each of these analyses are stored in an appropriately named sub-directory within the ``configs`` directory:

* ``configs/clnk_rves``
* ``configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves``
* ``configs/polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves``
* ``configs/polydisperse_inext_cufjc_networks_clnk_rves``
* ``configs/bimodal_pdms_networks``

Each of these sub-directories contain a ``config.yaml`` YAML file defining a wide variety of parameter configuration settings. Moreover, each of these sub-directories contain two more sub-directories, ``topology`` and ``deformation``. Within each of these sub-directories are YAML files that define parameter configuration settings specifically related to network topology and deformation protocol(s), respectively. The Hydra package is employed to load in the settings from the YAML files.

The ``irregular_topologies_i.sh`` Bash script analyzes the mechanics of various polydisperse polymer networks with irregular cross-links formed by inextensible Gaussian chains and inextensible Kuhn-Grün chains under uniaxial and simple shear deformation. This analysis renders some of the data and results used in the ``Polydisperse polymer networks with irregular topologies I`` manuscript. Running this Bash script on my 2022 MacBook Air with an Apple M2 chip and 8 GM RAM took approximately 10 hours to complete (i.e., overnight).

*****
Usage
*****

**Before running any of the code, it is required that the user verify the baseline filepath in the ``filepath_str()`` function of the ``file_io.py`` Python file in the ``file_io`` directory. Note that filepath string conventions are operating system-sensitive.**

********
Citation
********

\Jason Mulderrig, Michael Buche, and Matthew Grasinger, Polydisperse polymer networks with irregular topologies I: Mechanics of cross-link distributions, `arXiv (2025) <https://arxiv.org/abs/2504.17239>`_.

\Jason Mulderrig, Matthew Grasinger, and Phil Buskohl, Polydisperse polymer networks with irregular topologies II: Investigating the synthesis-topology-property relationship of bimodal PDMS networks, In preparation.
