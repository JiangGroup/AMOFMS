======
AMOFMS
======

.. start-description

**An Automated Mapping and Optimization Framework for Molecular Simulation**

.. start-badges


|mdanalysis|
|gromacs|
|conda|
|pypi|
|docs|
|supported-versions|
|license|



.. |mdanalysis| image:: https://img.shields.io/badge/powered%20by-MDAnalysis-blue.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA
    :alt: Powered by MDAnalysis
    :target: https://www.mdanalysis.org

.. |gromacs| image:: https://img.shields.io/badge/simulation%20with-GROMACS-orange.svg
    :alt: Powered by GROMACS
    :target: https://gromacs.org

.. |conda| image:: https://img.shields.io/badge/conda--forge-v0.11.0-orange
    :alt: Conda-fogre latest release
    :target: https://anaconda.org/conda-forge/amofms

.. |pypi| image:: https://img.shields.io/badge/pypi-v0.11.0-orange
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/amofms

.. |docs| image:: https://readthedocs.org/projects/amofms/badge/?style=flat
    :target: https://readthedocs.org/projects/amofms
    :alt: Documentation Status

.. |supported-versions| image:: https://img.shields.io/badge/python-3.9%7C3.10%7C3.11-blue
    :alt: Supported versions
    :target: https://pypi.org/project/amofms

.. |license| image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :alt: License **AMOFMS** is free software licensed under the MIT License.
    :target: https://github.com/dropletsimuli/amofms/LICENSE
    
.. end-badges


Overview
========

**AMOFMS** (Automated Mapping and Optimization Framework for Molecular Simulation) is a versatile tool designed to streamline the creation and optimization of coarsed grained force field (**CGFF**) for molecular simulations. With its automated mapping function, AMOFMS simplifies the process of generating coarse-grained models by mapping fine-grained structures, reducing manual effort and increasing accuracy. Built on top of powerful libraries like `MDAnalysis <https://www.mdanalysis.org/>`__, `NumPy <https://numpy.org/>`__ and `SciPy <https://www.scipy.org/>`__ and conducting simulations on `GROMACS <https://www.gromacs.org>`__, the tool offers comprehensive optimization capabilities, allowing users to fine-tune a wide range of molecular parameters for precise simulations and improved force fields. Moreover, AMOFMS provides flexibility and customization options, enabling users to define custom objective functions and parameter equivalences to tailor the optimization process to their specific research needs. Through its user-friendly interface and powerful functionality, AMOFMS empowers researchers to accelerate the development of force field models and advance molecular simulation studies.


Installation
============

1. The easiest way to install **AMOFMS** along with its dependencies is through the `conda-forge
<https://anaconda.org/conda-forge>`__ channel of `Conda
<https://docs.conda.io/en/latest/index.html>`__::

    conda config --add channels conda-forge
    conda create -n amofms -c conda-forge python=3.9 amofms
    conda activate amofms

2. It's also possible to install **AMOFMS** from the `Python Package
Index <https://pypi.org/>`__. You can do this using `pip`::

    pip install amofms

Alternatively, you can also install the in-development version with::

    pip install https://github.com/dropletsimuli/amofms/archive/main.zip


Citing
======

If you use AMOFMS in your research, please cite it as follows:

Zhixuan Zhong, Lifeng Xu, Jian Jiang*. A Neural-Network-Based Mapping and Optimization Framework for High-Precision Coarse-Grained Simulation. 2024.

BibTeX entry:

.. code-block:: bibtex

    @software{2024amofms,
      author = {Zhixuan Zhong},
      title = {AMOFMS: Automated Mapping and Optimization Framework for Molecular Simulation},
      year = {2024},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\\url{https://github.com/dropletsimuli/amofms}},
      version = {0.1.0}
    }

Please also cite `MDAnalysis <https://www.mdanalysis.org/pages/citations/>`__, on which **AMOFMS** is built.

.. end-description

Full documentation
==================

Head to `amofms.readthedocs.io <https://amofms.readthedocs.io/en/stable/>`__, where you will find the full
documentation of **lipyphilic**'s API as well as examples of how to use the analysis tools.

