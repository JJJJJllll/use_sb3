import time
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO, A2C

from stable_baselines3.common.evaluation import evaluate_policy

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = gym.make(env_id)
        # use a seed for reproducibility
        # Important: use a different seed for each environment
        # otherwise they would generate the same experiences
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

#from stable_baselines3.common.env_util import make_vec_env

'''set parameters'''
env_id, ALGO = "CartPole-v1", A2C
NUM_EXP, NUM_ENV, TRAIN_STEP  = 3, [1, 2, 4, 8, 16], 5000
# RL algorithms can often be unstable, so we run several experiments (see https://arxiv.org/abs/1709.06560)
eval_env, EVAL_EPS = gym.make(env_id), 20

'''iterate'''
reward_ave, reward_std, train_time, total_procs = [], [], [], 0
for n_procs in NUM_ENV:
    print(f"Running for n_procs = {n_procs}")
    if n_procs == 1:
        train_env = DummyVecEnv([lambda: gym.make(env_id)])
    else:
        # This is equivalent to make_vec_env(env_id, n_envs=n_procs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
        train_env = SubprocVecEnv(
            [make_env(env_id, i + total_procs) for i in range(n_procs)],
            start_method="fork",
        )
        # for i in range(n_procs):
        #     print(i + total_procs)

    total_procs += n_procs

    rewards, times = [], []
    for experiment in range(NUM_EXP):
        # it is recommended to run several experiments due to variability in results
        train_env.reset()
        model = ALGO("MlpPolicy", train_env, verbose=0)
        start = time.time()
        model.learn(total_timesteps=TRAIN_STEP)
        times.append(time.time() - start)
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
        rewards.append(mean_reward)
    # Important: when using subprocesses, don't forget to close them
    # otherwise, you may have memory issues when running a lot of experiments
    train_env.close()
    reward_ave.append(np.mean(rewards))
    reward_std.append(np.std(rewards))
    train_time.append(np.mean(times))
print("train time: ", [round(t, 2) for t in train_time])

'''plot'''
def plot_training_results(training_steps_per_second, reward_averages, reward_std):
    """
    Utility function for plotting the results of training

    :param training_steps_per_second: List[double]
    :param reward_averages: List[double]
    :param reward_std: List[double]
    """
    plt.figure(figsize=(9, 4))
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(1, 2, 1)
    plt.errorbar(
        NUM_ENV,
        reward_averages,
        yerr=reward_std,
        capsize=2,
        c="k",
        marker="o",
    )
    plt.xlabel("Processes")
    plt.ylabel("Average return")
    plt.subplot(1, 2, 2)
    plt.bar(range(len(NUM_ENV)), training_steps_per_second)
    plt.xticks(range(len(NUM_ENV)), NUM_ENV)
    plt.xlabel("Processes")
    plt.ylabel("Training steps per second")
    plt.show()

train_steps_per_second = [TRAIN_STEP / t for t in train_time]
print("training_steps_per_second: ", [round(num, 2) for num in train_steps_per_second])
plot_training_results(train_steps_per_second, reward_ave, reward_std)