import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

"""
discrete version is not implement yet
"""
class Policy_Net(nn.Module):
    """
    policy network
    """
    
    def __init__(self, env, hidden=100):
        super(Policy_Net, self).__init__()
        self.env = env
        #for gym
        self.space_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        #for suite
        #self.space_dim = env.observation_spec()['observations'].shape[0]
        #self.action_dim = env.action_spec().shape[0]        
        
        self.linear1 = nn.Linear(self.space_dim, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.mean = nn.Linear(hidden, self.action_dim)
        self.cholesky = nn.Linear(hidden, (self.action_dim * (self.action_dim + 1)) //2)
        
                                  
    def forward(self, state):
        device = state.device
        
        # not in paper
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        #x = self.linear1(state)
        #x = self.linear2(x)
        
        #############for gym###########################
        # not in paper
        action_low = torch.from_numpy(self.env.action_space.low)[None, ...].to(device)  # (1, da)
        action_high = torch.from_numpy(self.env.action_space.high)[None, ...].to(device)  # (1, da)
        mean = torch.sigmoid(self.mean(x))  # (B, da)
        mean = action_low + (action_high - action_low) * mean
        ###############################################
        
        """
        ##############for suite########################
        # not in paper
        action_low = torch.from_numpy(self.env.action_spec().minimum)[None, ...].to(device)  # (1, da)
        action_high = torch.from_numpy(self.env.action_spec().maximum)[None, ...].to(device)  # (1, da)
        mean = torch.sigmoid(self.mean(x))  # (B, da)
        mean = action_low + (action_high - action_low) * mean
        ###############################################
        """
        mean = self.mean(x)
        # build a positive diagonal lower triangular matrix
        cholesky_vector = self.cholesky(x)
        cholesky_diag_index = torch.arange(self.action_dim, dtype=torch.long) + 1
        cholesky_diag_index = (cholesky_diag_index * (cholesky_diag_index + 1)) // 2 - 1
        cholesky_vector[:, cholesky_diag_index] = F.softplus(cholesky_vector[:, cholesky_diag_index])
        tril_indices = torch.tril_indices(row=self.action_dim, col=self.action_dim, offset=0)
        cholesky = torch.zeros(size=(state.size(0), self.action_dim, self.action_dim), dtype=torch.float32).to(device)
        cholesky[:, tril_indices[0], tril_indices[1]] = cholesky_vector
        """
        tril_indices = torch.tril_indices(row=self.action_dim, col=self.action_dim, offset=0)
        cholesky = torch.zeros(size=(state.size(0), self.action_dim, self.action_dim), dtype=torch.float32).to(device)
        cholesky[:, tril_indices[0], tril_indices[1]] = cholesky_vector
        for i in range(self.action_dim):
            cholesky[:,i,i] = F.softplus(cholesky[:,i,i])
        """
        return mean, cholesky
        
        
    def action(self, state):
        with torch.no_grad():
            mean, cholesky = self.forward(state[None, ...])
            action = MultivariateNormal(mean, scale_tril=cholesky)
            sample_action = action.sample()
        
        return sample_action[0]