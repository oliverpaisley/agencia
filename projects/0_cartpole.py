from agencia.envs import CartPoleEnv
from tqdm import tqdm

cart_pole_env = CartPoleEnv()
env = cart_pole_env.make(render_mode="human")

observation, info = env.reset(seed=42)
for _ in tqdm(range(100000)):
    action = (
        env.action_space.sample()
    )  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
