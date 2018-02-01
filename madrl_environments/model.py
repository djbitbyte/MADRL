import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = dim_observation * n_agent
        act_dim = self.dim_action * n_agent

        self.FC1 = nn.Linear(obs_dim, 512)  # 1024
        self.FC2 = nn.Linear(512+act_dim, 256)  # 512
        self.FC3 = nn.Linear(256, 100)  # 300
        self.FC4 = nn.Linear(100, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        result = F.relu(self.FC1(obs))
        combined = th.cat([result, acts], 1)
        result = F.relu(self.FC2(combined))
        return self.FC4(F.relu(self.FC3(result)))


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 256)  # 500
        self.FC2 = nn.Linear(256, 64)  # 128
        self.FC3 = nn.Linear(64, dim_action)

    # action output between -2 and 2
    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = F.tanh(self.FC3(result))
        return result


class Models:
    def __init__(self, n_agents, dim_observation, dim_action):
        self.actors = [Actor(dim_observation, dim_action) for i in range(n_agents)]
        self.critics = [Critic(n_agents, dim_observation, dim_action) for i in range(n_agents)]


# for test
if __name__ == "__main__":
    n_agents = 2
    dim_obs = 10
    dim_act = 4
    critic = Critic(n_agents, dim_obs, dim_obs)

