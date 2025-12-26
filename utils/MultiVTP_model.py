import torch
from torch import nn
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)
from MultiVTP.modules.PLE_model import PLE
from MultiVTP.modules.GO_semsim import GCN_gosemsim
from MultiVTP.modules.ESM_branches import ESM_branches
from MultiVTP.modules.Graphormer import GraphormerBlock

class MultiVTP(nn.Module):
    def __init__(self):
        super(MultiVTP, self).__init__()
        self.esm_branchs = ESM_branches([2560, 512, 128])
        self.go_aggragate = GCN_gosemsim([768, 1280, 128])
        self.tra_proj = nn.Sequential(nn.Linear(31, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64))

        self.gf = nn.ModuleList([GraphormerBlock(576) for _ in range(1)])
        self.dis_embeding = torch.nn.Embedding(30, 1, dtype=torch.float32)
        self.dis_embeding.weight.data.fill_(0)
        self.clf = PLE(576*3, 256, 256, 11)
    
    def forward(self, batch_data):
        """
        batch_data: {
        esm dim: (batch, n_graph, walk_lenth, 2560*3)
        go dim: (batch, n_graph, walk_lenth, 768)
        tra dim: (batch, n_graph, walk_lenth, 31)
        global dim: (batch, n_graph, walk_lenth, 256)
        local dim: (batch, n_graph, walk_lenth, walk_lenth)
        }
        """ 
        esm = self.esm_branchs(batch_data["seq"])
        go = self.go_aggragate(batch_data["go"])
        tra = self.tra_proj(batch_data["tra"])
        gformer_outs = []
        graph_nums = esm.shape[1]

        for n in range(graph_nums):
            shortest_path = batch_data["local"][:,n,:,:]
            local_topological_property = self.dis_embeding(shortest_path).squeeze(-1)
            multimodal = torch.cat([esm[:,n,:,:], go[:,n,:,:], tra[:,n,:,:]], dim=-1)

            global_topological_property = batch_data["global"][:,n,:,:]
            node_embeding = self.get_node_embeding(multimodal, global_topological_property)

            for layer in self.gf:
                node_embeding = layer(node_embeding, local_topological_property) #(batch, walk_lenth, embeding_dim)
            gformer_outs.append(node_embeding[:,0,:])

        gformer_outs = torch.cat(gformer_outs, dim=-1) #(batch_size, n_graph*embeding_dim)
        outs = self.clf(gformer_outs)
        return outs
    
    def get_node_embeding(self, multimodal, global_encoding):
        node_embeding = torch.cat((multimodal, global_encoding), dim=-1)
        return node_embeding


