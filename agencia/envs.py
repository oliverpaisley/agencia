from abc import ABC, abstractmethod
import gymnasium as gym

class Environment(ABC):
    def __init__(self):
        pass

    def make(self):
        pass


class CartPole(Environment):
    def __init__(self):
        pass

    def make(self):
        return gym.make('CartPole-v1')