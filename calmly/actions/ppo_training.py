# PPO fine-tuning from a behavioral cloning policy
import os
import sys
import importlib.util
import yaml
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from calmly.env import get_env_factory
from calmly.utils import load_factories

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_ppo_model(
    config_path: str,
    quiet: bool = False,
    force: bool = False,
    continue_training: bool = False,
    skip_bc: bool = False
):
    config = load_config(config_path)

    if (not config.get("ppo", {}).get("enabled", False)) and (not force):
        print("PPO training is disabled in config. Change the config file or use --force / force=True to override.")
        return

    if quiet == False:
        quiet = config['ppo'].get('quiet', False)

    if continue_training == False:
        continue_training = config['ppo'].get('continue_training', False)

    if skip_bc == False:
        skip_bc = config['ppo'].get('skip_bc', False)

    # Add user factory module directory to sys.path (required for SubprocVecEnv)
    factory_module = config["factories"]["module"]

    seed = config['ppo'].get("seed", 42)
    device_name = config['ppo'].get("device", None)
    n_envs = config['ppo']['n_envs']
    torch_num_threads = config['ppo'].get("torch_num_threads", 1)
    learning_rate = float(config['ppo'].get("learning_rate", 3e-4))
    total_timesteps = config['ppo'].get("total_timesteps", 400_000)
    tensorboard_log = config['ppo'].get("tensorboard_log", "logs/ppo")
    ent_coef = config['ppo'].get("ent_coef", 0.02)
    policy_path = config['bc'].get("policy_save_path", "models/bc_policy.pt")
    save_path = config['ppo'].get("save_path", "models/ppo_policy")
    Npeaks = config["ppo"]["n_peaks"]
    maxtem = config["ppo"]["maxtem"]
    mis_angle_min = config["ppo"]["mis_angle_min"]
    mis_angle_max = config["ppo"]["mis_angle_max"]

    torch.set_num_threads(torch_num_threads)

    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cpu")

    cav_sim_factory, scan_proc_factory = load_factories(factory_module)

    env_factory = get_env_factory(
        cav_sim_factory,
        scan_proc_factory,
        training=True,
        monitor=True,
        algo="PPO",
        Nmisalign=1,
        Npeaks=Npeaks,
        maxtem=maxtem,
        mis_angle_min=mis_angle_min,
        mis_angle_max=mis_angle_max,
        seed=seed
    )

    envs = SubprocVecEnv([env_factory for _ in range(n_envs)])

    if continue_training:
        if skip_bc and (not quiet):
            print("Warning: skip_bc flag is ignored if continue_training is set to True.")
        if os.path.exists(save_path + ".zip"):
            model = PPO.load(save_path, env=envs, device=device)
            if not quiet:
                print(f"Continuing training from saved PPO model at {save_path}.zip")
        else:
            print(f"Cannot continue training because the PPO model was not found at {save_path}.zip")
            return
    else:
        model = PPO(
            "MultiInputPolicy",
            envs,
            verbose=0 if quiet else 1,
            tensorboard_log=tensorboard_log,
            ent_coef=ent_coef,
            learning_rate=learning_rate,
            device=device,
        )
        if (not skip_bc) and policy_path and os.path.exists(policy_path):
            model.policy.load_state_dict(torch.load(policy_path, map_location=device))
            if not quiet:
                print(f"Initialized PPO model with BC policy from {policy_path}")

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=not quiet,
        tb_log_name="PPO_MultiEnv"
    )

    model.save(save_path)
    if not quiet:
        print(f"PPO model saved to {save_path}")

# Script entry point
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="calmly_config.yaml", help="Path to YAML config file")
    parser.add_argument("-q", "--quiet", action='store_true', help="Do not verbose the output")
    parser.add_argument("-f", "--force", action='store_true', help="Override the enable/disable setting in the config file")
    parser.add_argument("-c", "--continue", dest="continue_training", action='store_true', help="Continue from a previously saved PPO model")
    parser.add_argument("-n", "--no-imitation", dest="skip_bc", action='store_true', help="Do not initialize PPO with behavioral cloning policy")
    args = parser.parse_args()

    train_ppo_model(
        config_path=args.config,
        quiet=args.quiet,
        force=args.force,
        continue_training=args.continue_training,
        skip_bc=args.skip_bc
    )

if __name__ == "__main__":
    main()
