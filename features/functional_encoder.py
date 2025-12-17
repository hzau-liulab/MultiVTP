import torch
import dgl
import numpy as np
from transformers import AutoTokenizer, AutoModel  
from pygosemsim import graph
from pygosemsim import similarity
from pygosemsim import download


class FunctionEncoder:
    def __init__(self, pubmedbert_path="../pretrain_models/PubMedBERT"):
        self.tokenizer = AutoTokenizer.from_pretrained(pubmedbert_path)  
        self.model = AutoModel.from_pretrained(pubmedbert_path)
        # download.obo("go-basic")
        self.GO = graph.from_resource("go-basic")
        self.similarity_method = similarity.wang

    def get_embeddings(self, go_descriptions):
        if not go_descriptions:
            return torch.zeros(1, 768)
        else:
            functional_embeddings = []
            for description in go_descriptions:
                max_length = len(desciption)
                desciption = [desciption]
                toks = self.tokenizer.batch_encode_plus(desciption, 
                                                    padding="max_length", 
                                                    max_length=max_length, 
                                                    truncation=True,
                                                    return_tensors="pt")
                toks_cuda = {}
                for k,v in toks.items():
                    toks_cuda[k] = v
                embedding = self.model(**toks_cuda)
                functional_embedding = embedding["pooler_output"]
                functional_embeddings.append(functional_embedding)

            functional_embeddings = torch.cat(functional_embeddings)
            return functional_embeddings

    def get_go_similarity_graph(self, goids=None, thrshold=0.3):
        if not goids:
            similarity_matrix = np.ones((1, 1))
        else:
            go_nums = len(goids)
            similarity_matrix = np.zeros((go_nums, go_nums))
            
            for i in range(go_nums):
                query_id = goids[i]
                similarity_matrix[i, i] = 1.0
                for j in range(i + 1, go_nums):
                    target_id = goids[j]
                    try:
                        sim_score = self.similarity_method(self.GO, query_id, target_id)
                        similarity_matrix[i, j] = sim_score
                        similarity_matrix[j, i] = sim_score

                    except:
                        similarity_matrix[i, j] = 0.0
                        similarity_matrix[j, i] = 0.0
        sim_tensor = torch.from_numpy(similarity_matrix)
        go_similarity_graph = torch.where(sim_tensor > thrshold, torch.tensor(1.0), torch.tensor(0.0))
        return go_similarity_graph
    
    def get_go_graph(self, function_dict):
        golist = list(function_dict.keys())
        function_embeddings = self.get_embeddings([function_dict[goid] for goid in golist])
        go_similarity_graph = self.get_go_similarity_graph(golist)
        edge_index = torch.nonzero(go_similarity_graph, as_tuple=True)
        edge_index = torch.stack(edge_index, dim=0)
        go_graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=len(golist))
        go_graph.ndata['feat'] = function_embeddings.detach().float()
        return go_graph


