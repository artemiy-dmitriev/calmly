import importlib.util
import pickle
import os
from tqdm import trange
import numpy as np
import yaml
import argparse

from ..utils import load_factories
from ..policies import iterative_policy
from ..env import get_env_factory


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

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

def generate_dataset(config_path: str):
    config = load_config(config_path)

    if not config.get("dataset", {}).get("enabled", False):
        print("Dataset generation is disabled in config.")
        return

    dataset_path = config['dataset']['path']
    n_episodes = config['dataset']['n_episodes']
    quiet = config['dataset'].get('quiet', False)
    seed = config['dataset'].get('seed', None)
    
    cav_sim_factory, scan_proc_factory = load_factories(config["factories"]["module"])
        
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
        seed=seed
    )
    policy_fn = load_heuristic_policy(config["dataset"]["policy"])

    dataset = []
    episode_iterator = trange(n_episodes) if not quiet else range(n_episodes)
    for _ in episode_iterator:
        env = env_factory()
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

# Script entry point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="calmly_config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    generate_dataset(config_path=args.config)

