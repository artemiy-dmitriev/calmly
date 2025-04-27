import os
import yaml
import torch
import argparse
import numpy as np    
from datetime import datetime
from tqdm import trange

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from calmly.env import get_env_factory
from calmly.utils import load_factories
from calmly.policies import iterative_policy
from stable_baselines3.ppo import MultiInputPolicy as PPOMultiInputPolicy

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_agent(agent_path, device, env):
    """Loads agent depending on file extension."""
    if agent_path == "iterative_policy":
        return "iterative_policy"
    elif agent_path.endswith(".zip"):
        # currently assuming that any stored model is PPO
        model = PPO.load(agent_path, env=env, device=device)
        return model
    elif agent_path.endswith(".pt"):
        # currently assuming that anything that ends with .pt is a BC policy
        policy = PPOMultiInputPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: 0.0,  # Not used during evaluation
        )
        policy.load_state_dict(torch.load(agent_path, map_location=device))
        policy.to(device)
        policy.eval()
        return policy
    else:
        raise ValueError(f"Unknown agent format: {agent_path}")

def evaluate_model(
    config_path: str,
    quiet: bool = False,
    force: bool = False,
    message: str = ""
):
    config = load_config(config_path)

    if (not config.get("evaluation", {}).get("enabled", False)) and (not force):
        print("Evaluation is disabled in config. Enable it or use --force.")
        return

    agent_path = config["evaluation"]["agent_path"]
    num_episodes = config["evaluation"]["n_episodes"]
    factory_module = config["factories"]["module"]
    quiet = config["evaluation"].get("quiet", False)

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

    cav_sim_factory, scan_proc_factory = load_factories(factory_module)

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
        seed=seed
    )

    # eval_env = DummyVecEnv([env_factory])
    # model = PPO.load(model_path, env=eval_env, device=device)
    env = env_factory()

    agent = load_agent(agent_path, device, env)

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
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        success = False

        while not done:
            action = predict_fn(obs)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1
            done = terminated or truncated

            if done:
                success = info['success']

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

    eval_log_path = config["evaluation"].get("save_path", "evaluation_results.yaml")

    result_entry = {
        "datetime": datetime.now().isoformat(),
        "agent_name": agent_path,
        "n_episodes": num_episodes,
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
    parser.add_argument("-m", "--message", type=str, default="", help="Comment message (optional)")

    args = parser.parse_args()

    evaluate_model(
        config_path=args.config,
        quiet=args.quiet,
        force=args.force,
        message = args.message
    )

if __name__ == "__main__":
    main()
