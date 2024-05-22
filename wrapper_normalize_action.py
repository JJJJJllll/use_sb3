import numpy as np
import gymnasium as gym

class NormalizeActionWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Retrieve the action space
        action_space = env.action_space
        assert isinstance(
            action_space, gym.spaces.Box
        ), "This wrapper only works with continuous action space (spaces.Box)"
        # Retrieve the max/min values
        self.low, self.high = action_space.low, action_space.high

        # We modify the action space, so all actions will lie in [-1, 1]
        env.action_space = gym.spaces.Box(
            low=-1, high=1, shape=action_space.shape, dtype=np.float32
        )

        # Call the parent constructor, so we can access self.env later
        super(NormalizeActionWrapper, self).__init__(env)

    def rescale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.low + (0.5 * (scaled_action + 1.0) * (self.high - self.low))

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float,bool, bool, dict) observation, reward, final state? truncated?, additional informations
        """
        # Rescale action from [-1, 1] to original [low, high] interval
        rescaled_action = self.rescale_action(action)
        obs, reward, terminated, truncated, info = self.env.step(rescaled_action)
        return obs, reward, terminated, truncated, info

# test wrapper
original_env = gym.make("Pendulum-v1")

print("no normalize:\n", original_env.action_space.low)
for _ in range(10):
    print(original_env.action_space.sample())

env = NormalizeActionWrapper(gym.make("Pendulum-v1"))

print("with normalize:\n", env.action_space.low)

for _ in range(10):
    print(env.action_space.sample())

# train
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

env = Monitor(gym.make("Pendulum-v1"))
env = DummyVecEnv([lambda: env])

model = A2C("MlpPolicy", env, verbose=1).learn(int(1000))

normalized_env = Monitor(gym.make("Pendulum-v1"))
# Note that we can use multiple wrappers
normalized_env = NormalizeActionWrapper(normalized_env)
normalized_env = DummyVecEnv([lambda: normalized_env])

model_2 = A2C("MlpPolicy", normalized_env, verbose=1).learn(int(1000))