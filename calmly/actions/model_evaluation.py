import os
import yaml
import torch
import argparse
import numpy as np    
from datetime import datetime
from tqdm import trange

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from calmly.env import get_env_factory, load_env_settings
from calmly.utils import load_factories
from calmly.policies import iterative_policy
from calmly.io import load_config
from stable_baselines3.ppo import MultiInputPolicy as PPOMultiInputPolicy

def evaluate_model(
    config_path: str = "calmly_config.yaml",
    config : dict = None,
    quiet: bool = False,
    force: bool = False,
    agent_path: str = "",
    message: str = ""
):
    if config is None:
        config = load_config(config_path)

    if (not config.get("evaluation", {}).get("enabled", False)) and (not force):
        print("Evaluation is disabled in config. Enable it or use --force.")
        return
    
    if quiet==False:
        quiet = config['dataset'].get('quiet', False)

    else:
        if not quiet:
            print(f"Loading agent {agent_path} (overriding the config file)")

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

    raw_rewards = config['evaluation'].get('raw_rewards', False)
    if (not quiet) and raw_rewards:
        print("Normalisation of rewards suppressed for evaluation by raw_rewards=True setting in the config")
        
    num_episodes = config["evaluation"]["n_episodes"]
    
    if not message:
        message = config["evaluation"].get("message", "")

    Npeaks = config["evaluation"]["n_peaks"]
    maxtem = config["evaluation"]["maxtem"]
    mis_angle_min = config["evaluation"]["mis_angle_min"]
    mis_angle_max = config["evaluation"]["mis_angle_max"]
    seed = config["evaluation"].get("seed", None)

    device_name = config["evaluation"].get("device", None)
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cpu")

    env_factory = get_env_factory(
        cav_sim_factory,
        scan_proc_factory,
        training=False,
        monitor=False,
        algo="evaluation",
        Nmisalign=1,
        Npeaks=Npeaks,
        maxtem=maxtem,
        mis_angle_min=mis_angle_min,
        mis_angle_max=mis_angle_max,
        seed=seed,
        env_settings=env_settings
    )

    env = DummyVecEnv([env_factory])

    if not agent_path:
        agent_path = config["evaluation"]["agent_path"]
    if not quiet:
        print(f"Loading agent {agent_path}")

    if agent_path == "iterative_policy":
        agent = "iterative_policy"
        if not quiet:
            print("Agent type: iterative_policy (heuristic algorithm)")
        if use_VecNormalize:
            # first checking if there is a dataset with the normalisation stats
            dataset_path = config['dataset']['path']
            vn_path = os.path.splitext(dataset_path)[0]+"_vecnormalize.pkl"
            if os.path.isfile(vn_path):
                if not quiet:
                    print(f"Loading normalisation settings from {vn_path}...", end=' ')
                env = VecNormalize.load(vn_path, env)
                env.training = False
                if raw_rewards:
                    env.norm_reward = False
                if not quiet:
                    print("Done")
            else:
                # There is no dataset, will start fresh statistics
                if not quiet:
                    print("Dataset stats not found, using fresh stats")
                env = VecNormalize(env, norm_obs=norm_obs, norm_reward=norm_rew, training=False)
                if raw_rewards:
                    env.norm_reward = False
            if not quiet:
                print(f"Normalisation settings: norm_obs={env.norm_obs}, norm_reward={env.norm_reward}, training={env.training}")
            
    elif agent_path.endswith(".zip"):
        # currently assuming that any stored model is PPO
        if not quiet:
            print("Agent type: PPO model")

        if use_VecNormalize:
            vn_path = os.path.splitext(agent_path)[0]+"_vecnormalize.pkl"
            if not quiet:
                print(f"Loading normalisation settings from {vn_path}...", end=' ')
            env = VecNormalize.load(vn_path, env)
            env.training = False
            if raw_rewards:
                env.norm_reward = False
            if not quiet:
                print("Done")
                print(f"Normalisation settings: norm_obs={env.norm_obs}, norm_reward={env.norm_reward}, training={env.training}")

        agent = PPO.load(agent_path, env=env, device=device)
    elif agent_path.endswith(".pt"):
        # currently assuming that anything that ends with .pt is a BC policy
        if not quiet:
            print("Agent type: BC policy")

        if use_VecNormalize:
            vn_path = os.path.splitext(agent_path)[0]+"_vecnormalize.pkl"
            if not quiet:
                print(f"Loading normalisation settings from {vn_path}...", end=' ')
            env = VecNormalize.load(vn_path, env)
            env.training=False
            if raw_rewards:
                env.norm_reward = False
            if not quiet:
                print("Done")
                print(f"Normalisation settings: norm_obs={env.norm_obs}, norm_reward={env.norm_reward}, training={env.training}")
            
        agent = PPOMultiInputPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: 0.0,  # Not used during evaluation
        )
        agent.load_state_dict(torch.load(agent_path, map_location=device))
        agent.to(device)
        agent.eval()
    else:
        raise ValueError(f"Unknown agent format: {agent_path}")

    def predict_fn(obs):
        if agent == "iterative_policy":
            return iterative_policy(obs)
        elif isinstance(agent, PPO):
            action, _ = agent.predict(obs, deterministic=True)
            return action
        elif isinstance(agent, PPOMultiInputPolicy):
            with torch.no_grad():
                obs_tensor = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in obs.items()}
                dist = agent.get_distribution(obs_tensor)
                actions = dist.mode()
                return actions.cpu().numpy()[0]
        else:
            raise ValueError("Unknown agent type.")

    episode_iterator = trange(num_episodes) if not quiet else range(num_episodes)

    total_rewards = []
    episode_lengths = []
    successes = []

    for _ in episode_iterator:
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        success = False

        while not done:
            obs_unbatched = {k: v[0] for k, v in obs.items()}
            action = predict_fn(obs_unbatched)
            obs, reward, done, info = env.step([int(action)])
            total_reward += reward
            steps += 1

            if done:
                success = info[0]['success']

        total_rewards.append(total_reward)
        episode_lengths.append(steps)
        successes.append(success)

    mean_reward_per_step = np.mean(np.array(total_rewards) / np.array(episode_lengths))
    mean_episode_reward = np.mean(total_rewards)
    mean_episode_length = np.mean(episode_lengths)
    success_rate = np.mean(successes)

    if not quiet:
        print("\n=== Evaluation Results ===")
        print(f"Mean reward per step: {mean_reward_per_step:.3f}")
        print(f"Mean total reward: {mean_episode_reward:.3f}")
        print(f"Mean episode length: {mean_episode_length:.1f}")
        print(f"Success rate: {success_rate * 100:.1f}%")
        if message:
            print("Comment message:", message)

    eval_log_path = config["evaluation"].get("save_path", "evaluation/evaluation_results.yaml")

    result_entry = {
        "datetime": datetime.now().isoformat(),
        "agent_name": agent_path,
        "n_episodes": num_episodes,
        "observations_normalised": norm_obs,
        "rewards_normalised": norm_rew and (not raw_rewards),
        "mean_reward_per_step": float(mean_reward_per_step),
        "mean_total_reward": float(mean_episode_reward),
        "mean_episode_length": float(mean_episode_length),
        "success_rate": float(success_rate),
        "evaluation_config": {
            "n_peaks": Npeaks,
            "maxtem": maxtem,
            "mis_angle_min": mis_angle_min,
            "mis_angle_max": mis_angle_max,
            "seed": seed
        }
    }

    if message:
        result_entry["message"] = message

    if os.path.exists(eval_log_path):
        with open(eval_log_path, "r") as f:
            existing_entries = list(yaml.safe_load_all(f)) or []
    else:
        existing_entries = []

    existing_entries.append(result_entry)

    dir_path = os.path.dirname(eval_log_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(eval_log_path, "w") as f:
        yaml.dump_all(existing_entries, f, explicit_start=True)
    
    if not quiet:
        print(f"Saved evaluation results to {eval_log_path}")

# Script entry point
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="calmly_config.yaml", help="Path to YAML config file")
    parser.add_argument("-q", "--quiet", action='store_true', help="Do not verbose the output")
    parser.add_argument("-f", "--force", action='store_true', help="Override the enable/disable setting in the config file")
    parser.add_argument("-a", "--agent", type=str, default="", help="Agent/model to use (overrides agent_path in the config file)")
    parser.add_argument("-m", "--message", type=str, default="", help="Optional comment message")

    args = parser.parse_args()

    evaluate_model(
        config_path=args.config,
        quiet=args.quiet,
        force=args.force,
        agent_path=args.agent,
        message = args.message
    )

if __name__ == "__main__":
    main()
