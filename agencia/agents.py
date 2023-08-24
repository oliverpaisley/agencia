from typing import List
import numpy as np
from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def take_action(self, actions: List[int]):
        pass

class Random(Agent):
    def __init__(self):
        pass

    def take_action(self, actions: List[int]):
        return np.random.choice(actions)


# class MCControl(Agent):
#     def __init__(self):
#         pass


# class TDControl(Agent):
#     def __init__(self):
#         pass