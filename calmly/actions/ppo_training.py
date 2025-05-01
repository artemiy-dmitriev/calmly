import os
import sys
import importlib.util
import argparse
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from calmly.env import get_env_factory
from calmly.utils import load_factories
from calmly.io import load_config

from stable_baselines3.common.callbacks import BaseCallback

class AvgRewardLoggerCallback(BaseCallback):
    """
    Logs average reward per step to TensorBoard during training.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.avg_rewards_per_step = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        
        for info in infos:
            if "avg_reward_per_step" in info:
                self.avg_rewards_per_step.append(info["avg_reward_per_step"])

        return True

    def _on_rollout_end(self) -> None:
        if len(self.avg_rewards_per_step) > 0:
            avg_reward = np.mean(self.avg_rewards_per_step)
            self.logger.record("rollout/avg_reward_per_step", avg_reward)
            self.avg_rewards_per_step.clear()

def train_ppo_model(
    config_path: str = "calmly_config.yaml",
    config : dict = None,
    quiet: bool = False,
    force: bool = False,
    continue_training: bool = False,
    skip_bc: bool = False
):
    if config is None:
        config = load_config(config_path)

    if (not config.get("ppo", {}).get("enabled", False)) and (not force):
        print("PPO training is disabled in config. Enable it or use --force.")
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
    exploration_settings = config["ppo"].get(
        "exploration_settings",
        {
            "enabled" : False
        }
    )
    exploration_settings['decay_steps'] = int(exploration_settings.get(
        'decay_steps',
        total_timesteps/2
    ))
    exploration_settings['initial_probability'] = float(exploration_settings.get(
        'initial_probability',
        0.25
    ))
    exploration_settings['final_probability'] = float(exploration_settings.get(
        'final_probability',
        0.01
    ))
    
    if not quiet:
        if exploration_settings["enabled"]:
            print("Will introduce action noise with the following settings:")
            for k, v in exploration_settings.items():
                print(k, ':', v)
        else:
            print("Action noise is switched off in the environment (change with 'exploration_settings' in the config file)")

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
        exploration_settings=exploration_settings,
        use_avg_reward_wrapper=True
    )

    # need to add rank to avoid initialising subprocesses with the same seed
    def make_env(rank, base_seed=seed):
        def _init():
            env = env_factory()
            env.reset(seed=base_seed + rank)
            return env
        return _init
        
    envs = SubprocVecEnv([make_env(i) for i in range(n_envs)])

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
        tb_log_name="PPO_MultiEnv",
        callback=AvgRewardLoggerCallback()
    )

    model.save(save_path)
    if not quiet:
        print(f"PPO model saved to {save_path}")
    envs.close()

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
