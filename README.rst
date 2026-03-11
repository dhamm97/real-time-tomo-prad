real-time-tomo-prad
============

Repository for real-time tomography-based radiated power estimation
----------------------------------------

This repository provides open access to all the computational routines implemented by our paper
"Real-time tomography-based Bayesian inference from TCV bolometry data", as well as the analyzed data.
This is an integral part of our work, as we strongly believe in the importance of **reusability** and **reproducibility** in science. ♻️

The paper was recently submitted for publication. In the meanwhile, the arXiv version of this paper
can be found `here <https://arxiv.org/abs/2506.20232>`_ .

The code here provided can be used to reproduce all the results presented in the paper.

In particular, we provide instructions for:

* **Installation** ⚙️ : simple instructions to get started
* **Results generation** 📊 : scripts and Jupyter Notebooks allowing to reproduce all analyses and plots

The Jupyter Notebooks allow to easily inspect and visualize how the results included in the paper were generated,
even without needing to actually re-run them. 🔎

The pipelines proposed in this paper rely on the open-source computational imaging framework
`Pyxu <https://github.com/pyxu-org/pyxu>`_ and its plugin `pyxu-diffops <https://github.com/dhamm97/pyxu-diffops>`_ ,
both (co-)developed by us: check them out!


Installation: let's get started! 🚀
------------

Create a dedicated Python environment, e.g., by

.. code-block:: bash

   conda create --name rt-tomo-env python=3.12
   conda activate rt-tomo-env


Then, clone the repo and install the necessary dependencies by:

.. code-block:: bash

   git clone https://github.com/dhamm97/real-time-tomo-prad.git
   pip install .

Computational routines
----------------------

The folder ``src/routines`` contains all the code used in this project.

* The subfolder ``src/routines/tomo_fusion`` contains all that's needed to define the Bayesian reconstruction models used in our work
  (both for offline and real-time applications).

* The file ``src/routines/rt_roi_prad.py`` contains all the routines implemented for real-time estimation of the radiated power.


Data
----

The folder ``src/data`` contains information from the TCV discharges analyzed in this work. In particular, we include all the
data necessary to reproduce the presented results.

Results generation
--------------

The folder ``src/results`` contains the obtained results, the scripts used to generate them, and notebooks to visualize them.

* The notebook ``bolometer_diagnostic.ipynb`` plots the configuration of the TCV bolometer diagnostic  (Figure 1 of the paper).

* The notebook ``fbt_vs_liuqe.ipynb`` compares the FBT vs LIUQE magnetic equilibria for shot #85270 (Figure 2 of the paper).

* The subfolder ``src/results/hyperparameter_study_results`` contains the code and phantoms 👻 used to tune the model hyperparameters.
  The script ``generate_phantoms.py`` can be run to generate the SOLPS-inspired phantoms stored in the subfolder ``phantoms``.
  The script ``hyperparameter_study.py`` can be run to reproduce the hyperparameter study results. Warning ⚠️ : running ``hyperparameter_study.py``` takes a long time.
  If you have access to a workstation featuring multiple cores, run the ``.sh`` version of the script, splitting the computations among (in our case) 10
  different cores, which will greatly reduce the computation time. The results (Figure 7 of the paper) can be inspected in the notebook ``hyperparameter_tuning.ipynb``,
  and are stored in the subfolder ``phantom_analysis_results``.

* The subfolder ``src/results/campaign_analysis`` contains the results from the study conducted on 50 TCV shots from the 2025 TCV campaign.
  The script ``campaign_analysis.py`` can be run to generate the results that can be found in the subfolder ``campaign_study_results``.
  Warning ⚠️ : as discussed in the previous bullet point, you can run the ``.sh`` version of the script to split computations among multiple cores
  (20 in our case). The notebook ``paper_results.ipynb`` allows inspection of the results, and contains the code and analyses used to
  generate Figure 3, Figure 4, and Table 1 fro the paper.

Geometry
--------

The folder ``src/tcv_geometry`` contains some basic information about the geometry of the TCV tokamak and of the bolometer diagnostic
whose data we rely on in this work.


License
-------

Distributed under the terms of the `MIT`_ license,
this is free and open source software

Issues
------

Hopefully the provided instructions will be enough, but if you encounter any problems feel free to contact us!

Cite us
------

If any of this was useful for your own research, you can cite our `paper <https://arxiv.org/abs/2506.20232>`_ !