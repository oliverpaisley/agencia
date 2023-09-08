from abc import ABC, abstractmethod
import gymnasium as gym

class Environment(ABC):
    def __init__(self):
        pass

    def make(self):
        pass


class CartPoleEnv(Environment):
    def __init__(self):
        pass

    def make(self, render_mode: str | None = None):
        return gym.make('CartPole-v1', render_mode=render_mode)