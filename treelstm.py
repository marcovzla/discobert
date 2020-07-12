import torch
from torch import nn
import config

class TreeLstm(nn.Module):

    # https://arxiv.org/pdf/1503.00075.pdf
    # https://nlp.stanford.edu/pubs/bowman2016spinn.pdf
    # another way of adding rel to TreeLSTM: https://www.groundai.com/project/an-enhanced-tree-lstm-architecture-for-sentence-semantic-modeling-using-typed-dependencies/1
    
    def __init__(self, hidden_size, include_relation_embedding, include_direction_embedding, rel_hidden_size, dir_hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_l = nn.Linear(self.hidden_size, 5 * self.hidden_size)
        self.W_r = nn.Linear(self.hidden_size, 5 * self.hidden_size)
        if include_relation_embedding and include_direction_embedding:
            self.W_rel = nn.Linear(rel_hidden_size + dir_hidden_size, 5 * self.hidden_size) 
        elif include_relation_embedding:
            self.W_rel = nn.Linear(rel_hidden_size, 5 * self.hidden_size) 


    def slice_gate(self, gates, i):
        size = self.hidden_size
        pos = i * size
        return gates[:,pos:pos+size]

    def forward(self, lhs, rhs, rel_emb):
        l_h, l_c = torch.split(lhs, self.hidden_size, dim=1)
        r_h, r_c = torch.split(rhs, self.hidden_size, dim=1)
        if rel_emb != None:
            gates = self.W_l(l_h) + self.W_r(r_h) + self.W_rel(rel_emb) #hidden state * 5 to accommodate 5 diff gates
        else:
            gates = self.W_l(l_h) + self.W_r(r_h)
    
        i   = torch.sigmoid(self.slice_gate(gates, 0))
        f_l = torch.sigmoid(self.slice_gate(gates, 1))
        f_r = torch.sigmoid(self.slice_gate(gates, 2))
        o   = torch.sigmoid(self.slice_gate(gates, 3))
        g   = torch.tanh(self.slice_gate(gates, 4))
        c_t = f_l * l_c + f_r * r_c + i * g
        h_t = o * torch.tanh(c_t) 
        return torch.cat([h_t, c_t], dim=1)