.. calmly documentation master file, created by
   sphinx-quickstart on Mon Apr 28 08:35:12 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. calmly documentation master file, created by sphinx-quickstart

Welcome to calmly: Cavity ALignment with Machine Learning
=========================================================

**Calmly** is a Python package for **automated optical cavity alignment** using **reinforcement learning**.
It provides a flexible, modular, and scalable framework for simulating cavity systems, training machine learning agents, and deploying them in both simulation and real experiments.

Calmly supports:

- Dataset generation with heuristic policies
- Behavioral cloning (BC) from heuristic datasets
- PPO fine-tuning from BC or scratch
- Full evaluation and benchmarking
- YAML-based project configuration
- CLI and HPC-friendly operation
- Easy extensibility with user-defined cavity simulations and preprocessing

---

Getting Started
===============

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   configuration
   api_reference

---

Main Features
=============

- **Factory-based extensibility**: supply your own cavity simulation and scan preprocessing functions.
- **Flexible policies**: supports heuristic, imitation learning, and reinforcement learning agents.
- **Automated workflow**: manage full training pipelines through a single YAML config file.
- **Cluster-ready**: calmly can run in non-interactive HPC environments via CLI commands.
- **Easy evaluation**: benchmark different agents across consistent environments.
- **Clean-up utilities**: safely remove generated models, logs, and datasets when needed.

---

Citing Calmly
=============

If you use **Calmly** for research or publication, please consider citing it!

---

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


