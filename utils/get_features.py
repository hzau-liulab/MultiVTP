import sys
import os
import pickle
import torch
import numpy as np
from pathlib import Path
from dgl.dataloading import GraphDataLoader
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)
from MultiVTP.features.functional_encoder import FunctionEncoder
from MultiVTP.features.sequence_encoder import SequenceEncoder
from MultiVTP.features.traditional_encoder import TraEncoder
from MultiVTP.features.network_attributes import PPIEncoder


def _load_ppi_data():
    ppi_config = {
        Path("../input/PPInetwork/subgraphs.pkl"): "graph_sampling",
        Path("../input/PPInetwork/global_topological_properties.pkl"): "get_global_topological_properties",
        Path("../input/PPInetwork/local_topological_properties.pkl"): "get_shortest_path_distance"
    }
    
    ppiencoder = PPIEncoder()
    for file_path, method_name in ppi_config.items():
        if not file_path.exists():
            print(f"{file_path} not exists, starting generation...")
            generate_method = getattr(ppiencoder, method_name)
            generate_method()

    with open(f"../input/PPInetwork/subgraphs.pkl", "rb") as f:
        subgraph = pickle.load(f)
    with open(f"../input/PPInetwork/global_topological_properties.pkl", "rb") as f:
        global_top = pickle.load(f)
    with open(f"../input/PPInetwork/local_topological_properties.pkl", "rb") as f:
        local_top = pickle.load(f)
    return subgraph, global_top, local_top

def load_features(query_nodes, device):
    subgraph_sampling, global_toplogical_properties, local_toplogical_properties = _load_ppi_data()
    functional_encoder = FunctionEncoder()
    sequence_encoder = SequenceEncoder()
    traditinal_encoder = TraEncoder()
    batch_data = {
        "seq": [],
        "go": [],
        "tra": [],
        "global": [],
        "local": []
    }

    for query_node in query_nodes:
        if query_node in subgraph_sampling:
            query_multisubgraph = subgraph_sampling[query_node]
        else:
            query_multisubgraph = [[query_node for _ in range(20)] for _ in range(3)]

        #sequence_embedding, functional_embedding, traditinal_embedding, global_toplogical_properties, local_toplogical_properties
        multisubgraph_seq, multisubgraph_go, multisubgraph_tra, multisubgraph_global, multisubgraph_local = [],[],[],[],[]
        for query_subgraph in query_multisubgraph:
            query_subgraph_seq, query_subgraph_go, query_subgraph_tra, query_subgraph_global = [],[],[],[]
            for node in query_subgraph:

                #get sequence embedding
                sequence = open(f"../input/fasta/{node}.fasta", "r").readlines()[1].strip()
                sequence_embedding = sequence_encoder.get_embeddings(sequence)
                query_subgraph_seq.append(sequence_embedding)

                #get functional graph
                functional_lines = open(f"../input/GO/{node}.txt", "r").readlines()
                functional_dict = {line.split("\t")[0]:line.split("\t")[1].strip() for line in functional_lines}
                go_graph = functional_encoder.get_go_graph(functional_dict)
                query_subgraph_go.append(go_graph)

                #get traditional features
                trafeatures = traditinal_encoder.get_tra_features(node)
                query_subgraph_tra.append(trafeatures)

                #get global toplogical properties
                if query_node in global_toplogical_properties:
                    query_subgraph_global.append(global_toplogical_properties[query_node])
                else:
                    query_subgraph_global.append(global_toplogical_properties["padding"])
            

            #get subgraph shortest path matrix
            query_subgraph_local = []
            for node_start in query_subgraph:
                if node_start not in local_toplogical_properties:
                    dis = [0] * len(query_subgraph)
                else:
                    dis = [
                        local_toplogical_properties[node_start][end_node]
                        for end_node in query_subgraph
                    ]
                query_subgraph_local.append(dis)
                    
            minibatch_go_graph = GraphDataLoader(query_subgraph_go, shuffle=False, batch_size=len(query_subgraph_go))
            minibatch_go_graph = next(iter(minibatch_go_graph))
            multisubgraph_go.append(minibatch_go_graph)

            multisubgraph_seq.append(torch.stack(query_subgraph_seq))
            multisubgraph_tra.append(query_subgraph_tra)
            multisubgraph_global.append(query_subgraph_global)
            multisubgraph_local.append(query_subgraph_local)

    batch_data["seq"].append(torch.stack(multisubgraph_seq))
    batch_data["go"].append(multisubgraph_go)
    batch_data["tra"].append(multisubgraph_tra)
    batch_data["global"].append(multisubgraph_global)
    batch_data["local"].append(multisubgraph_local)

    batch_data["seq"] = torch.stack(batch_data["seq"]).to(torch.float32).to(device)
    batch_data["tra"] = torch.from_numpy(np.array(batch_data["tra"])).to(torch.float32).to(device)
    batch_data["global"] = torch.from_numpy(np.array(batch_data["global"])).to(torch.float32).to(device)
    batch_data["local"] = torch.tensor(batch_data["local"]).to(torch.long).to(device)
    return batch_data
                         


