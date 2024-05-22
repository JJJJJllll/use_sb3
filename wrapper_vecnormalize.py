import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack

env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
normalized_vec_env = VecNormalize(env)

obs = normalized_vec_env.reset()
for _ in range(10):
    action = [normalized_vec_env.action_space.sample()]
    obs, reward, _, _ = normalized_vec_env.step(action)
    print(action, obs, reward)