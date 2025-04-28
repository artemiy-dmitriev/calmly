Installation
============

To install **calmly**, you will need Python >= 3.9 and `pip`.

You can install the package locally using:

.. code-block:: bash

   pip install git+https://github.com/artemiy-dmitriev/calmly.git

Or clone the repository and install it manually:

.. code-block:: bash

   git clone https://github.com/artemiy-dmitriev/calmly.git
   cd calmly
   pip install .

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
