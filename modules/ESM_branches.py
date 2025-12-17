import torch
from torch import nn

class ESM_branches(nn.Module):
    def __init__(self, layer_list):
        super(ESM_branches, self).__init__()
        self.embedding_dim = layer_list[0]
        self.layer_list = layer_list

        self.branch1 = self.dnn_branch(layer_list)
        self.branch2 = self.dnn_branch(layer_list)
        self.branch3 = self.dnn_branch(layer_list)

        self.dense_layer = nn.Sequential(
            nn.Linear(layer_list[-1]*3, (layer_list[-1])),
            nn.ReLU())
        
    def dnn_branch(self, layer_list):
        layers = []
        for i in range(len(layer_list) - 1):
            layers.append(nn.Linear(layer_list[i], layer_list[i + 1]))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, inputs):
        '''
        input_dim: (batch, n_graph, walk_length, embedding_dim*3)
        '''
        x1 = inputs[:,:,:,:self.embedding_dim]
        x2 = inputs[:,:,:,self.embedding_dim:2*self.embedding_dim]
        x3 = inputs[:,:,:,2*self.embedding_dim:]
        x1,x2,x3 = self.branch1(x1), self.branch2(x2), self.branch3(x3)
        x = torch.cat((x1, x2, x3), dim=-1)
        x_cat = self.dense_layer(x)
        return x_cat