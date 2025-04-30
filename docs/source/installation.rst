Installation
============

To install **calmly**, you will need Python >= 3.9 and `pip`.

It is strongly advised to try `calmly` in a separate conda environment. Install conda of any flavour and run

.. code-block:: bash

   conda create --name calmly python pip
   conda activate calmly

You can then clone the repository and install it manually:

.. code-block:: bash

   git clone https://github.com/artemiy-dmitriev/calmly.git
   cd calmly
   pip install -e .

Or install the package directly (currently not recommended) with pip:

.. code-block:: bash

   pip install git+https://github.com/artemiy-dmitriev/calmly.git

Dependencies
------------

Calmly will automatically install the following major dependencies:

- NumPy
- PyTorch
- Gymnasium
- Stable-Baselines3
- Imitation
- tqdm
- PyYAML
- Finesse (for optical simulation)

Make sure you have a working C++ compiler (needed by some dependencies) and sufficient permissions to install packages.

Optional (Recommended)
-----------------------

If you plan to build the documentation locally:

.. code-block:: bash

   pip install sphinx sphinx-rtd-theme

If you plan to run on Mac M1/M2/M3 with GPU acceleration (MPS backend):

.. code-block:: bash

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
