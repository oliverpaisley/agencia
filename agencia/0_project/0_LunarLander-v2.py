import gymnasium as gym
from tqdm import tqdm

print("STARTED: LundarLander-v2")

env = gym.make("LunarLander-v2") #, render_mode="human")
observation, info = env.reset(seed=42)
for _ in tqdm(range(1000)):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()

print("COMPLETE: LundarLander-v2")