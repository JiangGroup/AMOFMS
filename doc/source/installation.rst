Installation
============

Conda
-----

The easiest way to install **AMOFMS** is through the `conda-forge
<https://anaconda.org/conda-forge>`__ channel of `Conda
<https://docs.conda.io/en/latest/index.html>`__::

    conda config --add channels conda-forge
    conda create -n amofms -c conda-forge python=3.9 amofms
    conda activate amofms

This will install **AMOFMS** along with all of its dependencies into a new virtual environment.

If you do not already have Conda installed on your machine, we recommend
downloading and installing `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__
--- a lightweight version of Conda.

PyPI
----

It's also possible to install **AMOFMS** from the `Python Package
Index <https://pypi.org/>`__. You can do this using `pip`::

    pip install amofms

Alternatively, you can also install the in-development version with::

    pip install https://github.com/JiangGroup/AMOFMS/archive/main.zip

Dependencies
------------

**AMOFMS** uses `MDAnalysis <https://www.mdanalysis.org/>`__ for molecular dynamics analysis,
`NumPy <https://numpy.org/>`__ for numerical computations, and `SciPy <https://www.scipy.org/>`__
for additional scientific computing tasks.

The following python dependencies are required for AMOFMS:

- lipyphilic==0.10.0
- matplotlib==3.4.3
- MDAnalysis==2.7.0
- networkx==3.2
- numpy==1.22.3
- pexpect==4.9.0
- Pillow==10.2.0
- pyswarms==1.3.0
- rdkit==2023.9.4
- Requests==2.31.0
- scikit_learn==1.3.2
- scipy==1.8.1
- seaborn==0.13.2
- scikit-image
- torch==2.1.2
- torch_geometric==2.4.0
- tqdm==4.66.1
- tensorboard
- openbabel

As mentioned above, the simplest way to install these packages,
along with **AMOFMS**, is with `Conda <https://docs.conda.io/en/latest/index.html>`__.
However, it is also possible to install MDAnalysis, NumPy, and SciPy using pip, or from source. See
the `MDAnalysis <https://userguide.mdanalysis.org/stable/installation.html>`_,
`NumPy <https://numpy.org/install/>`_, and
`SciPy <https://scipy.org/install.html>`_ installation instructions for further information.

.. note::

   When installing OpenBabel through PyPI, please note that there may be issues. You can install OpenBabel via conda with the following `conda command`::

       conda install -c conda-forge openbabel


   AMOFMS uses `GROMACS <https://www.gromacs.org>`__ for molecular dynamics simulations. When conducting simulations with AMOFMS, ensure that you have installed `GROMACS <https://www.gromacs.org>`__ version 2019.6 or later. Earlier GROMACS versions have not been tested. We highly recommend using GROMACS version `2021.4 <https://manual.gromacs.org/2021.4/download.html>`__ for optimal performance.

