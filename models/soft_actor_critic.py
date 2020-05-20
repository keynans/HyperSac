import torch
import torch.optim as opt
import torch.nn.functional as F
import numpy as np

from .q_network import QNetwork
from .hyper_q_network import Hyper_QNetwork
from .policy_network import PolicyNetwork
from helper import ReplayMemory, copy_params, soft_update
from matplotlib import pyplot as plt

class SoftActorCritic(object):
    def __init__(self,observation_space,action_space, args):
        self.s_dim = observation_space.shape[0]
        self.a_dim = action_space.shape[0]
        self.alpha = args.alpha
        self.entropy_tunning = args.entropy_tunning
        self.reg = {'mean': args.mean_reg, 'std':args.std_reg}
        self.is_Hyper = not args.no_hyper

        # create component networks
        if self.is_Hyper:
            network = Hyper_QNetwork
            qlr = args.hyper_lr
        else:
            network = QNetwork
            qlr = args.lr

        self.q_network_1 = network(self.s_dim,self.a_dim,args.hidden_dim).to(args.device)
        self.q_network_2 = network(self.s_dim,self.a_dim,args.hidden_dim).to(args.device)
        self.target_q_network_1 = network(self.s_dim,self.a_dim,args.hidden_dim).to(args.device)
        self.target_q_network_2 = network(self.s_dim,self.a_dim,args.hidden_dim).to(args.device)
        self.policy_network = PolicyNetwork(self.s_dim, self.a_dim, args.hidden_dim, action_space,args.min_log,
                                args.max_log,args.epsilon, args.device).to(args.device)

        # copy weights from q networks to target networks
        copy_params(self.target_q_network_1, self.q_network_1)
        copy_params(self.target_q_network_2, self.q_network_2)
        
        # optimizers
        self.policy_network_opt = opt.Adam(self.policy_network.parameters(),args.lr)
        self.q_network_1_opt = opt.Adam(self.q_network_1.parameters(),qlr)
        self.q_network_2_opt = opt.Adam(self.q_network_2.parameters(),qlr)
        
        self.device = args.device

        self.hist = {}
        for name, param in self.policy_network.named_parameters():
            if name.endswith("weight"):
                self.hist[name + '.mean'] = []
                self.hist[name + '.max'] = []

        # automatic entropy tuning
        if args.entropy_tunning:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(args.device)).item()
            self.log_alpha = torch.tensor([0.], requires_grad=True, device=args.device)
            self.alpha_optim = opt.Adam([self.log_alpha], lr=args.lr)
                
        self.replay_memory = ReplayMemory(int(args.replay_memory_size))

    def get_action(self, s):
        state = torch.FloatTensor(s).to(self.device).unsqueeze(0)
        action, _, _, _ = self.policy_network.sample_action(state)
        return action.detach().cpu().numpy()[0]

    def save_gradient(self, network):
        #check policy gradient
        for name, param in network.named_parameters():
            if name.endswith("weight"):
                if param.grad is not None:
                    self.hist[name + '.mean'].append(torch.mean(param.grad).item())
                    self.hist[name + '.max'].append(torch.max(param.grad).item())
                else:
                    self.hist[name + '.mean'].append(0.)
                    self.hist[name + '.max'].append(0.)

    def gradClamp(self, parameters, clip):
        if clip < 0:
            return
        for p in parameters:
            p.grad.data.clamp_(max=clip)

    def reduce_lr(self, lr=1e-5):
        for param_group in self.policy_network_opt.param_groups:
            param_group['lr'] = lr
        print ("### lr drop %s ###"%lr)


    def update_params(self, batch_size, gamma, tau, grad_clip):

        states, actions, rewards, next_states, ndones = self.replay_memory.sample(batch_size)
        
        # make sure all are torch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        ndones = torch.FloatTensor(np.float32(ndones)).unsqueeze(1).to(self.device)

        # compute targets
        with torch.no_grad():
            next_action, next_log_pi,_, _ = self.policy_network.sample_action(next_states)
            next_target_q1 = self.target_q_network_1(next_states,next_action)
            next_target_q2 = self.target_q_network_2(next_states,next_action)
            next_target_q = torch.min(next_target_q1,next_target_q2) - self.alpha*next_log_pi
            next_q = rewards + gamma*ndones*next_target_q

        # compute losses
        q1 = self.q_network_1(states,actions)
        q2 = self.q_network_2(states,actions)

        q1_loss = F.mse_loss(q1,next_q)
        q2_loss = F.mse_loss(q2,next_q)

        # gradient descent
        self.q_network_1_opt.zero_grad()
        q1_loss.backward()
        self.q_network_1_opt.step()

        self.q_network_2_opt.zero_grad()
        q2_loss.backward()
      #  if self.is_Hyper:
            #self.gradClamp(self.q_network_2.parameters(), grad_clip)
          #  torch.nn.utils.clip_grad_value_(self.q_network_2.parameters(), grad_clip)
       #     torch.nn.utils.clip_grad_norm_(self.q_network_2.last_layer.parameters(), grad_clip)
        self.q_network_2_opt.step()
            
        pi, log_pi, mean, log_std = self.policy_network.sample_action(states)
        q1_pi = self.q_network_1(states,pi)
        q2_pi = self.q_network_2(states,pi)
        min_q_pi = torch.min(q1_pi,q2_pi)

        policy_loss = ((self.alpha * log_pi) - min_q_pi).mean()
        #regularization losses
        #mean_loss = self.reg['mean'] * mean.pow(2).mean()
        #std_loss = self.reg['std'] * log_std.pow(2).mean()
        #policy_loss += mean_loss + std_loss

        self.save_gradient(self.policy_network)

        self.policy_network_opt.zero_grad()
        policy_loss.backward()

        norm = 0
        for p in self.policy_network.parameters():
            param_norm = p.grad.data.norm(2)
            norm += param_norm.item() ** 2
        norm = norm ** (1. / 2)            

        self.policy_network_opt.step()

        # alpha loss
        if self.entropy_tunning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)

        # update target network params
        soft_update(self.target_q_network_1,self.q_network_1, tau)
        soft_update(self.target_q_network_2,self.q_network_2, tau)

        return q1_loss.item(), q2_loss.item(), policy_loss.item(), norm, torch.mean(q1_pi,dim=0).item(), torch.mean(q2_pi,dim=0).item()