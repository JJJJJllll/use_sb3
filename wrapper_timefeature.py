import numpy as np
import gymnasium as gym

from gym.wrappers import TimeLimit


class TimeFeatureWrapper(gym.Wrapper):
    """
    Add remaining time to observation space for fixed length episodes.
    See https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.

    :param env: (gym.Env)
    :param max_steps: (int) Max number of steps of an episode
        if it is not wrapped in a TimeLimit object.
    :param test_mode: (bool) In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """

    def __init__(self, env, max_steps=1000, test_mode=False):
        assert isinstance(env.observation_space, gym.spaces.Box)
        # Add a time feature to the observation
        low, high = env.observation_space.low, env.observation_space.high
        low, high = np.concatenate((low, [0])), np.concatenate((high, [1.0]))
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        super().__init__(env)

        if isinstance(env, TimeLimit):
            self._max_steps = env._max_episode_steps
        else:
            self._max_steps = max_steps
        self._current_step = 0
        self._test_mode = test_mode

    def reset(self, **kwargs):
        self._current_step = 0
        obs, info = self.env.reset(**kwargs)
        return self._get_obs(obs), info

    def step(self, action):
        self._current_step += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._get_obs(obs), reward, terminated, truncated, info

    def _get_obs(self, obs):
        """
        Concatenate the time feature to the current observation.

        :param obs: (np.ndarray)
        :return: (np.ndarray)
        """
        # Remaining time is more general
        time_feature = 1 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        # Optionally: concatenate [time_feature, time_feature ** 2]
        # print(obs, "\n", np.concatenate((obs, [time_feature])))
        return np.concatenate((obs, [time_feature]))

"""play with environment"""
env = TimeFeatureWrapper(gym.make("LunarLander-v2"), max_steps=1000)#"CartPole-v1" "LunarLander-v2"

obs = env.reset()
for _ in range(1):
    action = env.action_space.sample()
    obs, reward, _, _, _ = env.step(action)
    print(action, obs, reward)

"""train"""
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
model = A2C("MlpPolicy", env, verbose=1)
# from stable_baselines3.ppo.policies import MlpPolicy
# model = PPO(MlpPolicy, env, verbose=1)

mean_reward_init, std_reward_init = evaluate_policy(model, env, n_eval_episodes=1000)

model.learn(total_timesteps=100_000)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000)
print(f"init: mean_reward:{mean_reward_init:.2f} +/- {std_reward_init:.2f}\n end: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

"""record"""
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
def record_video(env_id, model, video_length=500, prefix="", video_folder="videos/"):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])
    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder=video_folder,
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=prefix,
    )

    obs = eval_env.reset()
    # Author: JJJJJllll 
    # Date: 240522
    # Vecenv obs -> flatten -> model -> action -> [action] -> Vecenv
    for i in range(video_length):
        # print("before: ", obs)
        obs = np.concatenate((obs[0], [1 - i/video_length]))
        # print("after: ", obs)
        action, _ = model.predict(obs)
        # print(action)
        obs, _, _, _ = eval_env.step([action])

    # Close the video recorder
    eval_env.close()

record_video("LunarLander-v2", model, video_length=1000, prefix="A2C-LunarLander")
