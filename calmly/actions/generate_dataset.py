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

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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
            print(f"Using calmly config dict directly passed as a parameter")

    if (not config.get("dataset", {}).get("enabled", False)) and (not force):
        print("Dataset generation is disabled in config. Enable it or use --force.")
        return

    if quiet==False:
        quiet = config['dataset'].get('quiet', False)

    factory_module = config["general"]["factory_module"]
    if not quiet:
        print(f"Loading factories from {factory_module}")    
    cav_sim_factory, scan_proc_factory = load_factories(factory_module)

    env_settings_location = config.get('general', {}).get('env_settings', False)
    if env_settings_location:
        if not quiet:
            if isinstance(env_settings_location, dict):
                print(f"Loading environment settings directly from the calmly config")
            else:
                print(f"Loading environment settings from {env_settings_location}")
        env_settings = load_env_settings(env_settings_location)
    else:
        env_settings = None

    norm_obs = config['general'].get('norm_observations', True)
    if not quiet:
        norm_obs_status = "On" if norm_obs else "Off"
        print("Normalisation of observations:", norm_obs_status)
    norm_rew = config['general'].get('norm_rewards', True)
    if not quiet:
        norm_rew_status = "On" if norm_rew else "Off"
        print("Normalisation of rewards:", norm_obs_status)
    use_VecNormalize = norm_obs or norm_rew
        
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
    
    env = DummyVecEnv([env_factory])
    
    if use_VecNormalize:
        env = VecNormalize(env, norm_obs=norm_obs, norm_reward=norm_rew)
        
    for _ in episode_iterator:
        obs = env.reset()

        while True:
            obs_unbatched = {k: v[0] for k, v in obs.items()}
            action = policy_fn(obs_unbatched)
            dataset.append((obs_unbatched, action))
            obs, reward, done, info = env.step([action])
            if done:
                break

    if not quiet:
        print(f"Saving dataset with {len(dataset)} samples to {dataset_path}...", end=' ')
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)
    if not quiet:
        print("Done")

    if use_VecNormalize:
        vn_path = os.path.splitext(dataset_path)[0]+"_vecnormalize.pkl"
        if not quiet:
            print(f"Saving normalisation settings to {vn_path}...", end=' ')
        env.save(vn_path)
        if not quiet:
            print("Done")
    env.close()

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

