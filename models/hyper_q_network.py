import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):

    def __init__(self, layer, ltype="linear"):
        super(ResBlock, self).__init__()

   
        if ltype == "linear":
            self.fc = nn.Sequential(
                                nn.Linear(layer, layer, bias=True),
                                nn.ELU(),
                                nn.Linear(layer, layer, bias=True),
                               )
        elif ltype == "conv1":
            self.fc = nn.Sequential(
                                nn.Conv1d(layer, layer, kernel_size=3,padding=1),
                                nn.ELU(),
                                nn.Conv1d(layer, layer, kernel_size=3,padding=1),
                               )

    def forward(self, x):
        
        h = self.fc(x)
        return F.elu(x + h)

class Reggresor(nn.Module):

    def __init__(self, in_features, out_features, clamp=100):

        super(Reggresor, self).__init__()

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.kaiming_uniform_(self.weight)
        self.clamp=clamp

    def forward(self, x):

        y = torch.mm(x, self.weight)
        w = torch.norm(self.weight, dim=0, keepdim=True)

        x = torch.norm(x, dim=1, keepdim=True)
        
        xw = torch.clamp(x * w, max=self.clamp)
        y = y / (xw)
        
        return y

class Head(nn.Module):

    def __init__(self, latent_dim, output_dim_in, output_dim_out):
        super(Head, self).__init__()
        
        h_layer = 1024
        self.output_dim_in = output_dim_in
        self.output_dim_out = output_dim_out
                
        # Q function net
        self.W = nn.Sequential(
            nn.Linear(h_layer, output_dim_in * output_dim_out)
        )

        self.b = nn.Sequential(
            nn.Linear(h_layer, output_dim_out)
        )
        self.s = nn.Sequential(
            nn.Linear(h_layer, output_dim_out)
        )

        self.init_layers()

    def forward(self, x):

        w = self.W(x).view(-1, self.output_dim_out, self.output_dim_in)
        b = self.b(x).view(-1, self.output_dim_out, 1)
        s = 1. + self.s(x).view(-1, self.output_dim_out, 1)

        return w, b, s
   
    def init_layers(self):
        for b in self.b.modules():
            if isinstance(b, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                torch.nn.init.zeros_(b.weight)  

class State_Embadding(nn.Module):

    def __init__(self, state_dim, z_dim):

        super(State_Embadding, self).__init__()

        f_layer = 512
        self.z_dim = z_dim

        self.hyper = nn.Sequential(
			nn.Linear(state_dim, f_layer, bias=True),
			nn.ELU(),
			ResBlock(f_layer),
			ResBlock(f_layer),
            ResBlock(f_layer),
            ResBlock(f_layer),
            ResBlock(f_layer),
			nn.Linear(f_layer, self.z_dim, bias=True),
            nn.ELU(),
		)

        self.init_layers()

    def forward(self, state):
		
        # f heads
        z = self.hyper(state).view(-1, self.z_dim)
        return z

    def init_layers(self):

        # init f with fanin
        for module in self.hyper.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1. / math.sqrt(fan_in)
                torch.nn.init.uniform_(module.weight, -bound, bound) 


class Hyper_QNetwork(nn.Module):
    # Hyper net that create weights from the state for a net that estimates function Q(S, A)
    def __init__(self,state_dim, action_dim,hidden_dim, num_hidden=3):
        super(Hyper_QNetwork, self).__init__()
	
        self.emb_dim = 64
        self.g_layer = 32
        self.z_dim = 1024
        
        self.hyper = State_Embadding(state_dim, self.z_dim)

        # Q function net
        self.layer1 = Head(self.z_dim,self.emb_dim,self.g_layer)
        self.hidden = nn.ModuleList(Head(self.z_dim,self.g_layer,self.g_layer) for i in range(num_hidden))
        self.last_layer = Head(self.z_dim,self.g_layer,1)

        self.action_emb = nn.Linear(action_dim, self.emb_dim)
    

    def forward(self, state, action):
        
        z = self.hyper(state)

		#action embedding
        emb = torch.tanh(self.action_emb(action).view(-1,self.emb_dim,1))
   
        # g first layer
        w ,b ,s = self.layer1(z)
        out = F.elu(torch.bmm(w, emb) * s + b)
   
        # g hidden layers
        for i, layer in enumerate(self.hidden):
            w, b, s = self.hidden[i](z)
            out = F.elu(torch.bmm(w, out) * s + b)

        # g final layer
        w, b, s = self.last_layer(z)
        out = torch.bmm(w, out) * s + b  

        return out.view(-1, 1)
  

