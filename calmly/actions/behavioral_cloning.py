# Behavioral cloning training from dataset
import importlib.util
import pickle
import os
import yaml
import argparse
import torch
import numpy as np

from imitation.algorithms.bc import BC
from imitation.data.types import Transitions, DictObs

from calmly.env import get_env_factory
from calmly.utils import load_factories
from stable_baselines3.ppo import MultiInputPolicy as PPOMultiInputPolicy

import warnings
warnings.filterwarnings("ignore", message="trying to unwrap object of type")

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_dataset(dataset_path: str) -> Transitions:
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    obs_list, acts_list = zip(*data)  # Each obs is a dict

    obs_dict_np = {
        key: np.array([obs[key] for obs in obs_list], dtype=np.float32)
        for key in obs_list[0].keys()
    }
    obs = DictObs(obs_dict_np)
    actions = np.array(acts_list)

    dummy_length = len(actions)
    empty_infos = [{} for _ in range(dummy_length)]
    empty_dones = np.zeros(dummy_length, dtype=bool)
    empty_next_obs = obs  # Not used for BC

    return Transitions(
        obs=obs,
        acts=actions,
        infos=empty_infos,
        dones=empty_dones,
        next_obs=empty_next_obs,
    )

def train_bc_model(
    config_path: str = "calmly_config.yaml",
    config : dict = None,
    quiet: bool = False,
    force: bool = False
):
    if config is None:
        config = load_config(config_path)

    if (not config.get("bc", {}).get("enabled", False)) and (not force):
        print("Behavioral cloning is disabled in config. Enable it or use --force.")
        return

    if quiet==False:
        quiet = config['bc'].get('quiet', False)
        
    dataset_path = config['dataset']['path']
    torch_num_threads = config['bc'].get("torch_num_threads", 10)
    seed = config['bc'].get("seed", 42)
    device_name = config['bc'].get("device", None)
    n_envs = config['bc'].get("n_envs", 4)
    learning_rate = float(config['bc'].get("learning_rate", 1e-4))
    net_arch = config['bc'].get("net_arch", [64, 64])
    n_epochs = config['bc'].get("n_epochs", 10)
    output_path = config['bc'].get("policy_save_path", "models/bc_policy.pt")
    factory_module = config["factories"]["module"]
    Npeaks = config["dataset"]["n_peaks"]
    maxtem = config["dataset"]["maxtem"]
    mis_angle_min=config["dataset"]["mis_angle_min"]
    mis_angle_max=config["dataset"]["mis_angle_max"]
    
    transitions = load_dataset(dataset_path)
    rng = np.random.default_rng(seed)

    if device_name is not None:
        device = torch.device(device_name)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    torch.set_num_threads(torch_num_threads)

    cav_sim_factory, scan_proc_factory = load_factories(factory_module)
        
    env_factory = get_env_factory(
        cav_sim_factory,
        scan_proc_factory,
        training=True,
        monitor=True,
        algo="BC",
        Nmisalign=1,
        Npeaks = Npeaks,
        maxtem = maxtem,
        mis_angle_min=mis_angle_min,
        mis_angle_max=mis_angle_max
    )
    
    # need to add rank to avoid initialising subprocesses with the same seed
    def make_env(rank, base_seed=seed):
        def _init():
            env = env_factory()
            env.reset(seed=base_seed + rank)
            return env
        return _init

    from stable_baselines3.common.vec_env import DummyVecEnv
    envs = DummyVecEnv([make_env(i) for i in range(n_envs)])

    custom_policy = PPOMultiInputPolicy(
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        lr_schedule=lambda _: learning_rate,
        net_arch=net_arch
    )

    bc_trainer = BC(
        policy=custom_policy,
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        demonstrations=transitions,
        rng=rng,
        device=device
    )

    progress_bar = False if quiet else True
    bc_trainer.train(n_epochs=n_epochs, progress_bar=progress_bar)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(bc_trainer.policy.state_dict(), output_path)
    if not quiet:
        print(f"BC policy saved to {output_path}")

# Script entry point
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="calmly_config.yaml", help="Path to YAML config file")
    parser.add_argument("-q", "--quiet", action='store_true', help="Do not verbose the output")
    parser.add_argument("-f", "--force", action='store_true', help="Override the enable/disable setting in the config file")
    args = parser.parse_args()

    train_bc_model(
        config_path=args.config,
        quiet=args.quiet,
        force=args.force
    )
    
if __name__ == "__main__":
    main()
    
