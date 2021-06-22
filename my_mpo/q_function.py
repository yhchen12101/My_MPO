import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

"""
discrete version is not implement yet
"""
class Q_Net(nn.Module):
    """
    q function network
    """
    def __init__(self, env, hidden=200):
        super(Q_Net, self).__init__()
        self.env = env
        #################for gym################
        self.space_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        ##############m##############m##############
        """
        #################for suite################
        self.space_dim = env.observation_spec()['observations'].shape[0]
        self.action_dim = env.action_spec().shape[0]     
        ##############m##############m##############
        """
        self.linear1 = nn.Linear(self.space_dim + self.action_dim, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden, 1)
        
    def forward(self, state, action):
        q = torch.cat([state, action], dim=1)
        
        # not in paper
        q = F.relu(self.linear1(q))
        q = F.relu(self.linear2(q))        
        
        #q = self.linear1(q)
        #q = self.linear2(q)
        q = self.linear3(q)
        
        return q