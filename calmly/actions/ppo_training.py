import os
import sys
import importlib.util
import argparse
import torch
import numpy as np
import signal
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize


from calmly.env import get_env_factory, load_env_settings
from calmly.utils import load_factories
from calmly.io import load_config

from stable_baselines3.common.callbacks import CallbackList, BaseCallback

class RolloutSuccessRateCallback(BaseCallback):
    """
    Logs average episode success rate to TensorBoard during training.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.successes = []

    def _on_step(self) -> bool:
        # Loop through all infos (one per env)
        for info in self.locals.get("infos", []):
            # Only log on episode end
            if "episode" in info and "is_success" in info:
                self.successes.append(info["is_success"])
        return True

    def _on_rollout_end(self) -> None:
        if self.successes:
            success_rate = sum(self.successes) / len(self.successes)
            self.logger.record("rollout/success_rate", success_rate)
            self.successes.clear()
        
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

class SaveOnStepCallback(BaseCallback):
    """
    Save the model periodically during training (every `save_freq` steps).
    """

    def __init__(self, save_path: str, save_freq: int = 10_000, use_VecNormalize: bool = False, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.use_VecNormalize = use_VecNormalize

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            step_path = f"{self.save_path}_step{self.num_timesteps}.zip"
            self.model.save(step_path)
            if self.verbose > 0:
                print(f"Saved checkpoint: {step_path}")
            if self.use_VecNormalize:
                vn_output_path = os.path.splitext(step_path)[0]+"_vecnormalize.pkl"
                if not quiet:
                    print(f"Saving normalisation settings to {vn_output_path}...", end=' ')
                self.training_env.save(vn_output_path)
                if not quiet:
                    print("Done")
            
        return True


def train_ppo_model(
    config_path: str = "calmly_config.yaml",
    config : dict = None,
    quiet: bool = False,
    force: bool = False,
    continue_training: bool = False,
    skip_bc: bool = False,
    n_envs: int = None
):
    if config is None:
        if not quiet:
            print(f"Loading calmly config from {config_path}")
        config = load_config(config_path)
    else:
        if not quiet:
            print(f"Using calmly config dict directly passed to train_ppo_model as a parameter")

    if (not config.get("ppo", {}).get("enabled", False)) and (not force):
        print("PPO training is disabled in config. Enable it or use --force.")
        return

    if quiet == False:
        quiet = config['ppo'].get('quiet', False)

    if continue_training == False:
        continue_training = config['ppo'].get('continue_training', False)

    if skip_bc == False:
        skip_bc = config['ppo'].get('skip_bc', False)

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
        print("Normalisation of rewards:", norm_rew_status)
    use_VecNormalize = norm_obs or norm_rew

    if n_envs is None:
        n_envs = config['ppo']['n_envs']
    else:
        if not quiet:
            print("Overriding the config setting for n_envs with a directly specified parameter")
    if not quiet:
        print(f"Using {n_envs} environments")
    seed = config['ppo'].get("seed", 42)
    device_name = config['ppo'].get("device", None)
    
    torch_num_threads = config['ppo'].get("torch_num_threads", 1)
    learning_rate = float(config['ppo'].get("learning_rate", 3e-4))
    total_timesteps = config['ppo'].get("total_timesteps", 400_000)
    tensorboard_log_dir = config['ppo'].get("tensorboard_log_dir", "logs/ppo")
    tensorboard_log_name = config['ppo'].get("tensorboard_log_name", "PPO_MultiEnv")
    ent_coef = config['ppo'].get("ent_coef", 0.02)
    policy_path = config['bc'].get("policy_save_path", "models/bc_policy.pt")
    save_path = config['ppo'].get("save_path", "models/ppo_policy")
    load_path = config['ppo'].get("load_path", save_path)
    save_freq = int(config['ppo'].get("save_freq", 50_000)/n_envs)
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
        use_avg_reward_wrapper=True,
        env_settings=env_settings
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
        reset_num_timesteps = False
        if skip_bc and (not quiet):
            print("Warning: skip_bc flag is ignored if continue_training is set to True.")
        if os.path.exists(load_path + ".zip"):
            if use_VecNormalize:
                vn_path = load_path+"_vecnormalize.pkl"
                if not quiet:
                    print(f"Loading normalisation settings from {vn_path}...", end=' ')
                envs = VecNormalize.load(vn_path, envs)
                if not quiet:
                    print("Done")
            model = PPO.load(load_path, env=envs, device=device)
            if not quiet:
                print(f"Continuing training from saved PPO model at {load_path}.zip")
        else:
            print(f"Cannot continue training because the PPO model was not found at {load_path}.zip")
            return
    else:
        reset_num_timesteps = True

        if (not skip_bc) and use_VecNormalize:
            vn_path = os.path.splitext(policy_path)[0]+"_vecnormalize.pkl"
            if not quiet:
                print(f"Loading normalisation settings from {vn_path}...", end=' ')
            envs = VecNormalize.load(vn_path, envs)
            if not quiet:
                print("Done")

        if skip_bc and use_VecNormalize:
            envs = VecNormalize(envs, norm_obs=norm_obs, norm_reward=norm_rew)
        
        model = PPO(
            "MultiInputPolicy",
            envs,
            verbose=0 if quiet else 1,
            tensorboard_log=tensorboard_log_dir,
            ent_coef=ent_coef,
            learning_rate=learning_rate,
            device=device,
        )
        if (not skip_bc) and policy_path and os.path.exists(policy_path):
            model.policy.load_state_dict(torch.load(policy_path, map_location=device))
            if not quiet:
                print(f"Initialized PPO model with BC policy from {policy_path}")

    def cleanup(signum, frame):
        if not quiet:
            print("\n[INFO] Caught interrupt. Saving model and closing environments.")
        model.save(save_path)
        if use_VecNormalize:
            vn_output_path = os.path.splitext(save_path)[0]+"_vecnormalize.pkl"
            if not quiet:
                print(f"Saving normalisation settings to {vn_output_path}...", end=' ')
            envs.save(vn_output_path)
            if not quiet:
                print("Done")
        envs.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    callback = CallbackList([
        AvgRewardLoggerCallback(),
        SaveOnStepCallback(
            save_path=save_path,
            save_freq=save_freq
        ),
        RolloutSuccessRateCallback()
    ])

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=not quiet,
        tb_log_name=tensorboard_log_name,
        callback=callback,
        reset_num_timesteps=reset_num_timesteps
    )
    model.save(save_path)
    if not quiet:
        print(f"PPO model saved to {save_path}")
    if use_VecNormalize:
        vn_output_path = os.path.splitext(save_path)[0]+"_vecnormalize.pkl"
        if not quiet:
            print(f"Saving normalisation settings to {vn_output_path}...", end=' ')
        envs.save(vn_output_path)
        if not quiet:
            print("Done")
        
    envs.close()

# Script entry point
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="calmly_config.yaml", help="Path to YAML config file")
    parser.add_argument("-q", "--quiet", action='store_true', help="Do not verbose the output")
    parser.add_argument("-f", "--force", action='store_true', help="Override the enable/disable setting in the config file")
    parser.add_argument("-c", "--continue", dest="continue_training", action='store_true', help="Continue from a previously saved PPO model")
    parser.add_argument("-n", "--no-imitation", dest="skip_bc", action='store_true', help="Do not initialize PPO with behavioral cloning policy")
    parser.add_argument("-N", "--number_of_environments", type=int, default=-1, dest="n_envs", help="Number of environments to use (overrides the config file.")
    
    args = parser.parse_args()

    if args.n_envs == -1:
        n_envs = None
    else:
        n_envs = args.n_envs

    train_ppo_model(
        config_path=args.config,
        quiet=args.quiet,
        force=args.force,
        continue_training=args.continue_training,
        skip_bc=args.skip_bc,
        n_envs=n_envs
    )

if __name__ == "__main__":
    main()
