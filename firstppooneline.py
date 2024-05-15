import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy

model = PPO('MlpPolicy', "CartPole-v1", verbose=1).learn(1000)

env = gym.make("CartPole-v1")

from stable_baselines3.common.evaluation import evaluate_policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, warn=False)
print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")