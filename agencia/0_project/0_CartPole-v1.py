import gymnasium as gym
from tqdm import tqdm

env_name = "CartPole-v1"
print("STARTED: {}".format(env_name))

env = gym.make('CartPole-v1') #, render_mode="human")

observation, info = env.reset(seed=42)
for _ in tqdm(range(1000)):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()

print("COMPLETE: {}".format(env_name))