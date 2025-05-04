import importlib.util
import pickle
import os
from tqdm import trange
import numpy as np
import argparse

from calmly.utils import load_factories, load_env_settings
from calmly.policies import iterative_policy
from calmly.env import get_env_factory
from calmly.io import load_config

def load_heuristic_policy(policy_path):
    if policy_path == "default":
        return iterative_policy
    elif policy_path.endswith(".pkl"):
        with open(policy_path, "rb") as f:
            return pickle.load(f)
    else:
        # Treat as a dotted path
        mod_path, func_name = policy_path.rsplit(".", 1)
        mod = importlib.import_module(mod_path)
        return getattr(mod, func_name)

def generate_dataset(
    config_path: str = "calmly_config.yaml",
    config : dict = None,
    quiet: bool = False,
    force: bool = False
):
    if config is None:
        if not quiet:
            print(f"Loading calmly config from {config_path}")
        config = load_config(config_path)
    else:
        if not quiet:
            print(f"Using calmly config dict directly passed to train_ppo_model as a parameter")

    if (not config.get("dataset", {}).get("enabled", False)) and (not force):
        print("Dataset generation is disabled in config. Enable it or use --force.")
        return

    if quiet==False:
        quiet = config['dataset'].get('quiet', False)

    factory_module = config["factories"]["module"]
    if not quiet:
        print(f"Loading factories from {factory_module}")    
    cav_sim_factory, scan_proc_factory = load_factories(factory_module)

    env_settings_location = config.get('factories', {}).get('env_settings', False)
    if env_settings_location:
        if not quiet:
            if isinstance(env_settings_location, dict):
                print(f"Loading environment settings directly from the calmly config")
            else:
                print(f"Loading environment settings from {env_settings_location}")
        env_settings = load_env_settings(env_settings_location)
    else:
        env_settings = None
        
    dataset_path = config['dataset']['path']
    n_episodes = config['dataset']['n_episodes']
    seed = config['dataset'].get('seed', None)
        
    env_factory = get_env_factory(
        cav_sim_factory,
        scan_proc_factory,
        training=False,
        monitor=False,
        algo="explicit_policy",
        Nmisalign = 1,
        Npeaks = config["dataset"]["n_peaks"],
        maxtem = config["dataset"]["maxtem"],
        mis_angle_min=config["dataset"]["mis_angle_min"],
        mis_angle_max=config["dataset"]["mis_angle_max"],
        seed=seed,
        env_settings=env_settings
    )
    policy_fn = load_heuristic_policy(config["dataset"]["policy"])

    dataset = []
    episode_iterator = trange(n_episodes) if not quiet else range(n_episodes)
    env = env_factory()
    for _ in episode_iterator:
        obs, info = env.reset()

        while True:
            action = policy_fn(obs)
            dataset.append((obs, action))
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    env.close()

    if not quiet:
        print(f"Saving dataset with {len(dataset)} samples to {dataset_path}...")
    os.makedirs(os.path.dirname(config["dataset"]["path"]), exist_ok=True)
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)
    if not quiet:
        print("Dataset saved.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="calmly_config.yaml", help="Path to YAML config file")
    parser.add_argument("-q", "--quiet", action='store_true', help="Do not verbose the output")
    parser.add_argument("-f", "--force", action='store_true', help="Override the enable/disable setting in the config file")
    args = parser.parse_args()

    generate_dataset(
        config_path=args.config,
        quiet=args.quiet,
        force=args.force
    )

if __name__ == "__main__":
    main()

