Configuration
=============

The recommended way to use `calmly` is to keep each project in its own separate directory. Typically there are three main configuration files per project, which will be described in this section: the main Calmly configuration file (YAML), another YAML with settings for the environment, and the factory Python module or pickle which defines factory functions for the cavity simulation model and the preprocessing of cavity scans.

The example project included in the repository (``example_project/``) contains these files and a Jupyter notebook showing the typical workflow. Interacting with Calmly through Jupyter notebooks is convenient if it physically runs on user's hardware. In addition, several bash scripts are provided to run Calmly on HPC clusters.

Calmly configuration file
-------------------------

This file specifies the paths, parameters, and options for data generation, training, and evaluation. By default, it is located in the project's root directory and called `calmly_config.yaml`.

Example calmly_config.yaml
~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is a minimal example `calmly_config.yaml`:

.. code-block:: yaml

    # User factory definition module and environment settings
    general:
      factory_module: "my_factories.py"  # User-defined Python or pickle file containing the factory functions
      env_settings: "env_settings.yaml" # Environment settings (can also be a nested dictionary)
      norm_observations: true # Use VecNormalize to normalize the observations
      norm_rewards: true # Use VecNormalize to normalize the rewards
    
    # Heuristic dataset generation
    dataset:
      enabled: true
      quiet: false
      n_episodes: 200
      path: "data/heuristic_dataset.pkl"
      n_peaks: 5
      maxtem: 5
      mis_angle_min: -2.0e-4 
      mis_angle_max: 2.0e-4
      policy: "default" # or a pickle path
      seed: 42
    
    # Behavioral Cloning
    bc:
      enabled: true
      quiet: false
      policy_save_path: "models/bc_policy.pt"
      torch_num_threads: 10
      n_envs: 4
      device: "mps"
      net_arch: [64, 64]
      learning_rate: 1e-4
      n_epochs: 10
      seed: 142
    
    # PPO Fine-Tuning
    ppo:
      enabled: true
      quiet: false
      continue_training: false
      skip_bc: false
      device: "cpu"
      n_envs: 10
      torch_num_threads: 1
      learning_rate: 3e-4
      total_timesteps: 400000
      ent_coef: 0.02
      exploration_settings:
        enabled: true
      tensorboard_log_dir: "logs/ppo"
      tensorboard_log_name: "PPO_MultiEnv"
      save_path: "models/ppo_policy"
      load_path: "models/ppo_policy"
      save_freq: 50000
      n_peaks: 5
      maxtem: 5
      mis_angle_min: -4.0e-4 
      mis_angle_max: 4.0e-4
      seed: 242
      
    # Evaluation
    evaluation:
      enabled: true
      quiet: false
      n_episodes: 20
      agent_path: "models/bc_policy.pt"
      n_peaks: 5
      maxtem: 5
      mis_angle_min: -4.0e-4 
      mis_angle_max: 4.0e-4
      seed: 342
      save_path: "evaluation/evaluation_results.yaml"

Sections
~~~~~~~~

**general**
  - ``factory_module``: Python file (.py) or a pickle file (.pkl) containing user-defined `cav_sim_factory()` and `scan_processor_factory()`.
  - ``env_settings``: YAML file containg settings for the environment, e.g. the reward logic. Can also be a nested dictionary with settings instead of a file path. See below for the description of this file.
  - ``norm_observations``: If True (default), calmly will use `VecNormalize() <https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#stable_baselines3.common.vec_env.VecNormalize>`_ to normalise the observation space.
  - ``norm_rewards``: If True (default), calmly will use `VecNormalize() <https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#stable_baselines3.common.vec_env.VecNormalize>`_ to normalise the rewards.

.. warning::
    The values of ``norm_observations`` and ``norm_rewards`` must be preserved over the whole workflow (including dataset generation, BC cloning, PPO learning, and evaluation, as well as in the final deployment) to maintain compatibility of obtained policies.

**dataset**
  - ``enabled``: Whether to generate a dataset.
  - ``quiet``: Suppress progress bar and verbose output.
  - ``n_episodes``: Number of episodes to simulate. Each episode starts with a misaligned cavity and ends when the alignment is optimised (or is hopelessly lost).
  - ``path``: Path to save the generated dataset (pickle file). Recommended location is in `data` subdirectory.
  - ``n_peaks``: Number of resonant peaks within one cavity free spectral range to include in the observation space of the environment. This setting will also be used by BC.
  - ``maxtem``: Maximum total order of TEM modes to include in the cavity simulation. For example, '3' will include 10 modes: 00, 10, 01, 20, 11, 02, 30, 21, 12, and 03. See `Finesse documentation <https://finesse.ifosim.org/docs/latest/physics/higher_order_modes/shifted_beam_convergence.html>`_ for details. This setting will also be used by BC.
  - ``mis_angle_min``, ``mis_angle_max``: Minimal and maximal initial angular misalignment for each of the "misalignable" cavity mirrors. These settings will also be used by BC.
  - ``policy``: Heuristic policy to use ("default" or path to custom .pkl or module.function). If "default" is specified, :py:func:`calmly.policies.iterative_policy` will be used.
  - ``seed``: Random seed for reproducibility.

**bc** (Behavioral Cloning)
  - ``enabled``: Whether to train a BC model.
  - ``quiet``: Suppress progress bar and verbose output.
  - ``policy_save_path``: Where to save the trained BC policy (.pt file).
  - ``torch_num_threads``: Number of CPU threads for PyTorch (set carefully on clusters).
  - ``n_envs``: Number of parallel environments.
  - ``device``: Device to use ("cpu", "cuda", "mps"...)
  - ``net_arch``: Neural network architecture, list of hidden layer sizes.
  - ``learning_rate``: Learning rate for BC optimizer.
  - ``n_epochs``: Number of epochs for training.
  - ``seed``: Random seed.

**ppo** (Reinforcement Learning with PPO)
  - ``enabled``: Whether to train a PPO agent.
  - ``quiet``: Suppress progress bar and verbose output.
  - ``continue_training``: Whether to continue from previous PPO model checkpoint (must be stored at ``save_path``).
  - ``skip_bc``: Start PPO without using BC initialization. If ``continue_training`` is True, this setting is ignored and BC will be skipped in any case.
  - ``device``: Device to use ("cpu", "cuda", "mps"...)
  - ``n_envs``: Number of parallel environments (SubprocVecEnv).
  - ``torch_num_threads``: Number of CPU threads for PyTorch (set carefully on clusters).
  - ``learning_rate``: Learning rate.
  - ``total_timesteps``: Total training steps.
  - ``ent_coef``: Entropy regularization coefficient.
  - ``exploration_settings``: Settings that control the action noise in PPO training to encourage exploration.

    - ``enabled``:  Whether or not to use action noise. Default is `True`.
    - ``decay_steps``: Number of timesteps to use action noise for. Default is `total_timesteps/2`.
    - ``initial_probability``: Initial chance of choosing a random action instead of the predicted one. Default is `0.25`.
    - ``final_probability``: Final chance (after decay_steps) of choosing a random action. Default is `0.01`.

  - ``tensorboard_log_dir``: Directory for Tensorboard logs. Recommended location is in ``logs/`` subdirectory, e.g. ``logs/ppo``.
  - ``tensorboard_log_name``: Name of the TB log.
  - ``save_path``: File path to save the PPO model (.zip file). Recommended location is in ``models/`` subdirectory.
  - ``load_path``: File path to load the PPO model from (without .zip extension). If not specified, will default to ``save_path``. This setting is only used if `continue_training` is set to `True`.
  - ``save_freq``: How often the checkpoints will be saved during training (in the units of steps). Default is 50000.
  - ``n_peaks``: Number of resonant peaks within one cavity free spectral range to include in the observation space of the environment.
  - ``maxtem``: Maximum total order of TEM modes to include in the cavity simulation. For example, '3' will include 10 modes: 00, 10, 01, 20, 11, 02, 30, 21, 12, and 03. See `Finesse documentation <https://finesse.ifosim.org/docs/latest/physics/higher_order_modes/shifted_beam_convergence.html>`_ for details.
  - ``mis_angle_min``, ``mis_angle_max``: Minimal and maximal initial angular misalignment for each of the "misalignable" cavity mirrors.
  - ``seed``: Random seed.

**evaluation**
  - ``enabled``: Whether to evaluate an agent.
  - ``quiet``: Suppress evaluation output.
  - ``n_episodes``: Number of episodes to evaluate over.
  - ``agent_path``: Path to the agent to evaluate (".zip", ".pt", or "iterative_policy").
  - ``seed``: Random seed for reproducibility.
  - ``n_peaks``, ``maxtem``, ``mis_angle_min``, ``mis_angle_max``: See above.
  - ``save_path``: YAML file to save evaluation results. Recommended location is in ``evaluation/`` subdirectory.
  - ``message``: Optional message that will be added to the evaluation results.

Notes
~~~~~

- Calmly expects the factories Python file (`my_factories.py`) to define two functions: `cav_sim_factory()` and `scan_processor_factory()`.
- Calmly automatically creates missing directories for models, logs, and evaluation outputs.
- The YAML file is flexible: unset fields will use sane defaults.

Environment settings file
-------------------------

This file specifies options for the generated environments (:py:class:`calmly.env.CavityAlignmentEnv`) which are sublclassed from `gymnasium.Env`. Its location is specified in the Calmly configuration file as `factories:env_settings` (see above). Alternatively, its contents can be provided directly as subsections in `factories:env_settings` of the calmly configuration file.

Example env_settings.yaml
~~~~~~~~~~~~~~~~~~~~~~~~~

Here is the version of `env_settings.yaml` containing default values:

.. code-block:: yaml

    general:
      allowed_best_result_annealing: 0.95
      final_countdown_steps: 10
      max_no_improv_steps: 20
      max_no_peaks_steps: 20
      max_steps: 1000
    motor_counts_per_step:
      SM1:
        pitch:
          neg: 3
          pos: 3
        yaw:
          neg: 3
          pos: 3
      SM2:
        pitch:
          neg: 3
          pos: 3
        yaw:
          neg: 3
          pos: 3
    reward_components:
      final_result:
        enabled: true
        final_countdown_cutoff_penalty: 1.0
        no_improv_cutoff_penalty: 1.0
        peaks_lost_cutoff_penalty: 1.0
        scaling: 308.0
        steplimit_truncation_penalty: 0.2
        success_reward: 1.0
      general:
        overall_reward_scaling: 1.0
        reward_log_compression: false
      main_peak_dominance:
        coef: 5.0
        enabled: true
      main_peak_improvement:
        allow_negative: false
        coef: 850.0
        enabled: true
      motor_switching:
        bad_motor_repeat_coef: 1.0
        bad_motor_switch_coef: 1.0
        bad_switched_motor_although_could_switch_direction: 0.2
        enabled: true
        good_direction_switch_coef: 1.0
        good_motor_repeat_coef: 1.0
        good_motor_switch_coef: 1.0
        scaling: 1.0
      motor_usage_dist:
        coef: 10.0
        enabled: true
      no_peaks:
        enabled: true
        penalty: 1.0
      other_peaks_dominance:
        coef: 2.5
        enabled: true
      other_peaks_improvement:
        allow_negative: false
        coef: 425.0
        enabled: false
      single_peak:
        enabled: true
        reward: 1.0
      time:
        enabled: true
        target_steps: 280

Sections
~~~~~~~~

Notes
~~~~~

- The YAML file is flexible. Any fields that are not specified will default to the values shown above.

Factory definition module
-------------------------

Further Information
--------------------

- See :doc:`quickstart` for a practical example.
- See :doc:`api_reference` for detailed information on available modules and functions.
