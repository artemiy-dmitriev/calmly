Quickstart
==========

Setting up your first **calmly** project is easy!

1. **Create a calmly configuration YAML file** (e.g., `calmly_config.yaml`).
2. **Provide your own cavity and preprocessing factories** in a Python module (e.g., `my_factories.py`).
3. **Generate a dataset** using a heuristic policy:

   .. code-block:: bash

      calmly-generate-dataset --config calmly_config.yaml

4. **Train a behavioral cloning (BC) model** on the dataset:

   .. code-block:: bash

      calmly-train-bc --config calmly_config.yaml

5. **Train a PPO model** starting from the BC model (or from scratch):

   .. code-block:: bash

      calmly-train-ppo --config calmly_config.yaml

6. **Evaluate the model**:

   .. code-block:: bash

      calmly-evaluate --config calmly_config.yaml

7. (Optional) **Clean up** intermediate files:

   .. code-block:: bash

      calmly-clean

Example Directory Structure
----------------------------

After running a few steps, your project directory might look like:

.. code-block:: text

   example_project/
   ├── calmly_config.yaml
   ├── my_factories.py
   ├── models/
   │   ├── bc_policy.pt
   │   └── ppo_policy.zip
   ├── logs/
   │   └── ppo_tensorboard/
   ├── data/
   │   └── heuristic_dataset.pkl
   └── evaluation/
       └── evaluation_results.yaml

Notes
-----

- By default, calmly will output models to the `models/` folder and logs to the `logs/` folder.
- You can customize everything (training parameters, file paths, logging options) through the YAML config file.
- Calmly is designed for both **interactive use (e.g., Jupyter notebooks)** and **non-interactive environments (e.g., HPC clusters)**.

Next Steps
----------

See :doc:`configuration` for full details on configuring calmly projects.
