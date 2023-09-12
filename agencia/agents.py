from typing import List
import numpy as np
from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self):
        self.action_space = None  #

    @abstractmethod
    def policy(state):
        """
        Mapping from states -> actions
        """
        pass

    @abstractmethod
    def take_action(self, state):
        """
        What do we need to take an action?
        Action = Policy(State)
        """
        pass


class Agent_Random(Agent):
    def __init__(self):
        # self.action_space = None
        pass

    def policy(self, state):
        # action = policy[state]
        action = self.action_space.sample()  # random policy
        return action

    def take_action(self, state):
        action = self.policy(state)
        return action


# We need to take an action with our agent
# To do this, we need to know which actions we can take
# We are dependent on the Env to tell the Agent what actions available
#


# class MCControl(Agent):
#     def __init__(self):
#         pass


# class TDControl(Agent):
#     def __init__(self):
#         pass

# Test test test
