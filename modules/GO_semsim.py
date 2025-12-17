import dgl
import torch
from torch import nn
from dgl.nn import GraphConv


class GCN_gosemsim(nn.Module):
    def __init__(self, layer_list):
        super(GCN_gosemsim, self).__init__()
        gcn_module_list = []
        for i in range(len(layer_list) - 1):
            gcn_conv = GraphConv(layer_list[i], layer_list[i + 1], activation=nn.ReLU())
            gcn_module_list.append(gcn_conv)
        self.gcn_module_list = nn.ModuleList(gcn_module_list)

        self.transform = nn.Sequential(
            nn.Linear(layer_list[-1], layer_list[-1]),
            nn.BatchNorm1d(layer_list[-1]),
            nn.ReLU())

    def forward(self, graph_list):
        '''
        graph_list dim: (batch, n_graph, walk_lengh, go_num, go_dim)
        '''
        outs = [
            torch.stack([self.Aggregate(n_graph) for n_graph in batch])
            for batch in graph_list
        ]
        return torch.stack(outs)
    
    def Aggregate(self, graph):
        '''
        graph dim (walk_lengh, go_num, go_dim)
        '''
        h = graph.ndata["feat"]
        for module in self.gcn_module_list:
            h = module(graph, h)
        with graph.local_scope():
            graph.ndata['h'] = h 
            emb = dgl.mean_nodes(graph, 'h')
        return self.transform(emb)