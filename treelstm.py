import torch
from torch import nn

class TreeLstm(nn.Module):

    # https://arxiv.org/pdf/1503.00075.pdf
    # https://nlp.stanford.edu/pubs/bowman2016spinn.pdf
    
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_l = nn.Linear(self.hidden_size, 5 * self.hidden_size)
        self.W_r = nn.Linear(self.hidden_size, 5 * self.hidden_size)

    def slice_gate(self, gates, i):
        size = self.hidden_size
        pos = i * size
        return gates[:,pos:pos+size]

    def forward(self, lhs, rhs): #there will be a new vector here for rel+ dir
        l_h, l_c = torch.split(lhs, self.hidden_size, dim=1)
        r_h, r_c = torch.split(rhs, self.hidden_size, dim=1)
        gates = self.W_l(l_h) + self.W_r(r_h) #there should be e here
        #need new input vector with relation and direction 
        #there would be a new W
        #the output should be the same as these
        #the input size might change
        #add new W
        #
        i   = torch.sigmoid(self.slice_gate(gates, 0))
        f_l = torch.sigmoid(self.slice_gate(gates, 1))
        f_r = torch.sigmoid(self.slice_gate(gates, 2))
        o   = torch.sigmoid(self.slice_gate(gates, 3))
        g   = torch.tanh(self.slice_gate(gates, 4))
        c_t = f_l * l_c + f_r * r_c + i * g
        h_t = o * torch.tanh(c_t)
        return torch.cat([h_t, c_t], dim=1)