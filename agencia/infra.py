from agencia.agents import Agent
from agencia.envs import Environment

class Infra:

    def __init__(self):
        pass

    def create_task(agent: Agent, env: Environment):
        pass



# 1. Create Environment
# 2. Create Agent (no action space known)
# 3. Create Task
    # 3a. Transfer knowledge of env.action_space to agent => (agent.action_space)


# 0_Cartpole.py
# env = Cartpole()
# agt = Agent_Random()
# task = create_task(agt, env)
# task.run()