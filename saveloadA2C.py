import os
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, SAC, PPO, TD3
import gymnasium as gym

# Create save dir
cwd = os.getcwd()
save_dir = f"{cwd}/gym/"
os.makedirs(save_dir, exist_ok=True)

model = A2C("MlpPolicy", "Pendulum-v1", verbose=0, gamma=0.9, n_steps=20).learn(8000)
# The model will be saved under A2C_tutorial.zip
model.save(f"{save_dir}/A2C_tutorial")

del model  # delete trained model to demonstrate loading

# load the model, and when loading set verbose to 1
loaded_model = A2C.load(f"{save_dir}/A2C_tutorial", verbose=1)

# show the save hyperparameters
print(f"loaded: gamma={loaded_model.gamma}, n_steps={loaded_model.n_steps}")

# as the environment is not serializable, we need to set a new instance of the environment
loaded_model.set_env(DummyVecEnv([lambda: gym.make("Pendulum-v1")]))
# and continue training
loaded_model.learn(8_000)