import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Normal
from helper import init_weights

class PolicyNetwork(nn.Module):
    def __init__(self,s_dim,a_dim,h_dim,action_space,min_log,max_log,epsilon, device):
        super(PolicyNetwork,self).__init__()
        self.min_log = min_log
        self.max_log = max_log
        self.epsilon = epsilon

        self.linear1 = nn.Linear(s_dim,h_dim)
        self.linear2 = nn.Linear(h_dim,h_dim)

        self.linear3a = nn.Linear(h_dim,a_dim)
        self.linear3b = nn.Linear(h_dim,a_dim)
     #   self.sigma = nn.Parameter(torch.Tensor(a_dim))
     #   self.sigma.data.fill_(math.log(1.))

        # Apply weight initialisation to all linear layers
        self.apply(init_weights)    

        # rescale actions
        if action_space is None:
            self.action_scale = torch.tensor(1.).to(device)
            self.action_bias = torch.tensor(0.).to(device)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.).to(device)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.).to(device)


    def forward(self,s):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        mean = self.linear3a(x)

        log_std = self.linear3b(x)
      #  log_std = self.sigma

        # constrain log value in finite range to avoid NaN loss values
        log_std = torch.clamp(log_std, min=self.min_log, max=self.max_log)
        
        return mean, log_std

    def sample_action(self,s):
        mean, log_std = self.forward(s)
        std = log_std.exp()

        # calculate action using reparameterization trick and action scaling
        normal = Normal(mean, std)
        xi = normal.rsample()
        yi = torch.tanh(xi)

        a = yi * self.action_scale + self.action_bias
        log_pi = normal.log_prob(xi)

        # enforcing action bound (appendix of paper)
        log_pi -= torch.log(self.action_scale * (1 - yi.pow(2)) + self.epsilon)
        log_pi = log_pi.sum(1,keepdim=True)
        mean = torch.tanh(mean)*self.action_scale + self.action_bias

        return a, log_pi, mean, log_std