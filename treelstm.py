import torch
from torch import nn

class TreeLstm(nn.Module):

    # https://arxiv.org/pdf/1503.00075.pdf
    # https://nlp.stanford.edu/pubs/bowman2016spinn.pdf
    # adding rel to TreeLSTM: https://www.groundai.com/project/an-enhanced-tree-lstm-architecture-for-sentence-semantic-modeling-using-typed-dependencies/1
    
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_l = nn.Linear(self.hidden_size, 5 * self.hidden_size)
        self.W_r = nn.Linear(self.hidden_size, 5 * self.hidden_size)
        self.W_rel = nn.Linear(self.)

    def slice_gate(self, gates, i):
        size = self.hidden_size
        pos = i * size
        return gates[:,pos:pos+size]

    def forward(self, lhs, rhs, rel_one_hot):
        l_h, l_c = torch.split(lhs, self.hidden_size, dim=1)
        r_h, r_c = torch.split(rhs, self.hidden_size, dim=1)
        print("lh shape: ", l_h.shape)
        print("l_c shape: ", l_c.shape)
        print("r_h shape: ", r_h.shape)
        print("r_c shape: ", r_c.shape)
        gates = self.W_l(l_h) + self.W_r(r_h) #hidden state * 5 to accommodate 5 diff gates?
    

        print("gates shape: ", gates.shape)
        r = 
        i   = torch.sigmoid(self.slice_gate(gates, 0))
        f_l = torch.sigmoid(self.slice_gate(gates, 1))
        f_r = torch.sigmoid(self.slice_gate(gates, 2))
        o   = torch.sigmoid(self.slice_gate(gates, 3))
        g   = torch.tanh(self.slice_gate(gates, 4))
        c_t = f_l * l_c + f_r * r_c + i * g
        h_t = o * torch.tanh(c_t)
        return torch.cat([h_t, c_t], dim=1)