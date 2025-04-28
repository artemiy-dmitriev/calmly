Configuration
=============

All **calmly** workflows are controlled by a single YAML configuration file.

This file specifies the paths, parameters, and options for data generation, training, and evaluation.

Example
-------

Here is a minimal example `calmly_config.yaml`:

.. code-block:: yaml

    factories:
      module: my_factories.py

    dataset:
      enabled: true
      path: data/heuristic_dataset.pkl
      policy: default
      n_episodes: 500
      n_peaks: 1
      maxtem: 3
      mis_angle_min: -2e-4
      mis_angle_max: 2e-4
      seed: 42
      quiet: false

    bc:
      enabled: true
      n_envs: 4
      torch_num_threads: 10
      learning_rate: 1e-4
      n_epochs: 10
      net_arch: [64, 64]
      policy_save_path: models/bc_policy.pt
      device: cpu
      seed: 42
      quiet: false

    ppo:
      enabled: true
      n_envs: 10
      learning_rate: 3e-4
      total_timesteps: 400000
      tensorboard_log: logs/ppo_tensorboard
      ent_coef: 0.02
      save_path: models/ppo_policy
      seed: 42
      continue_training: false
      skip_bc: false
      quiet: false

    evaluation:
      enabled: true
      agent_path: models/ppo_policy.zip
      n_episodes: 100
      save_path: evaluation/evaluation_results.yaml
      device: cpu
      quiet: false
      seed: 42
      n_peaks: 1
      maxtem: 3
      mis_angle_min: -2e-4
      mis_angle_max: 2e-4

Sections
--------

**factories**
  - ``module``: Python file containing user-defined `cav_sim_factory()` and `scan_processor_factory()`.

**dataset**
  - ``enabled``: Whether to generate a dataset.
  - ``path``: Path to save the generated dataset (pickle file).
  - ``policy``: Heuristic policy to use ("default" or path to custom .pkl or module.function).
  - ``n_episodes``: Number of episodes to simulate.
  - ``n_peaks``, ``maxtem``, ``mis_angle_min``, ``mis_angle_max``: Optical cavity parameters.
  - ``seed``: Random seed for reproducibility.
  - ``quiet``: Suppress progress bar and verbose output.

**bc** (Behavioral Cloning)
  - ``enabled``: Whether to train a BC model.
  - ``n_envs``: Number of parallel environments.
  - ``torch_num_threads``: Number of CPU threads for PyTorch (set carefully on clusters).
  - ``learning_rate``: Learning rate for BC optimizer.
  - ``n_epochs``: Number of epochs for training.
  - ``net_arch``: Neural network architecture, list of hidden layer sizes.
  - ``policy_save_path``: Where to save the trained BC policy (.pt file).
  - ``device``: Device to use ("cpu", "cuda", or "mps").
  - ``seed``: Random seed.
  - ``quiet``: Suppress verbose output.

**ppo** (Reinforcement Learning with PPO)
  - ``enabled``: Whether to train a PPO agent.
  - ``n_envs``: Number of parallel environments (SubprocVecEnv).
  - ``learning_rate``: Learning rate.
  - ``total_timesteps``: Total training steps.
  - ``tensorboard_log``: Directory for Tensorboard logs.
  - ``ent_coef``: Entropy regularization coefficient.
  - ``save_path``: File path to save the PPO model (.zip file).
  - ``continue_training``: Continue from previous PPO model checkpoint.
  - ``skip_bc``: Start PPO without using BC initialization.
  - ``quiet``: Suppress verbose output.

**evaluation**
  - ``enabled``: Whether to evaluate an agent.
  - ``agent_path``: Path to the agent to evaluate (".zip", ".pt", or "iterative_policy").
  - ``n_episodes``: Number of episodes to evaluate over.
  - ``save_path``: YAML file to save evaluation results.
  - ``device``: Evaluation device ("cpu", "cuda", or "mps").
  - ``quiet``: Suppress evaluation output.
  - ``seed``: Random seed for reproducibility.
  - ``n_peaks``, ``maxtem``, ``mis_angle_min``, ``mis_angle_max``: Optical cavity settings for evaluation.

Notes
-----

- Calmly expects the factories Python file (`my_factories.py`) to define two functions: `cav_sim_factory()` and `scan_processor_factory()`.
- Calmly automatically creates missing directories for models, logs, and evaluation outputs.
- The YAML file is flexible: unset fields will use sane defaults.

Further Information
--------------------

- See :doc:`quickstart` for a practical example.
- See :doc:`api_reference` for detailed information on available modules and functions.
