import numpy as np
import random
import networkx as nx
from node2vec import Node2Vec
import pickle

class Walker:
    def __init__(self, G, p, q):
        self.G = G
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    pos = (prev, cur)
                    next = cur_nbrs[alias_draw(alias_edges[pos][0],
                                            alias_edges[pos][1])]
                    walk.append(next)
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(
                    walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in G.neighbors(dst):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight']
                                  for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}

        for edge in G.edges():
            alias_edges[(edge[0],edge[1])] = self.get_alias_edge(edge[0], edge[1])
            alias_edges[(edge[1],edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return

def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K, dtype=np.float32)
    J = np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

class PPIEncoder:
    def __init__(self, PPI_path="../input/PPInetwork", num_walks = 3, walk_length = 20):
        self.ppi_path = PPI_path
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.G = pickle.load(open(rf"{PPI_path}/hippie_graph.pkl", 'rb'))
    
    def graph_sampling(self):
        walker = Walker(self.G, p=1, q=1)
        walker.preprocess_transition_probs()
        walks = walker.simulate_walks(num_walks=self.num_walks, walk_length=self.walk_length)

        subgraph_dict = {}
        for walk_path in walks:
            start_node = walk_path[0]
            subgraph_dict.setdefault(start_node, []).append(walk_path)
        with open(f"{self.ppi_path}/subgraphs.pkl", "wb")as f:
            pickle.dump(subgraph_dict, f)
        print("graph_sampling done")
    
    def get_global_topological_properties(self, trainset_path="../datasets/virus_species"):
        node2vec = Node2Vec(self.G, dimensions=256, walk_length=100, num_walks=16, workers=4, p=4, q=1)
        model = node2vec.fit(window=16, min_count=10, batch_words=32)
        node_embedding_dict = {str(node): model.wv[node] for node in model.wv.index_to_key}
        trainset = open(f"{trainset_path}/train.txt", 'r').read().split("\n")
        trainset_node_embedding = [node_embedding_dict[node] for node in trainset if node in node_embedding_dict]
        padding_embedding = np.mean(trainset_node_embedding)
        node_embedding_dict["padding"] = padding_embedding
        with open(f"{self.ppi_path}/global_topological_properties.pkl", "wb")as f:
            pickle.dump(node_embedding_dict, f)
        print("global topological properties done")
    
    def get_shortest_path_distance(self):
        shortest_path_length = dict(nx.shortest_path_length(self.G))
        with open(f"{self.ppi_path}/local_topological_properties.pkl", "wb")as f:
            pickle.dump(shortest_path_length, f)
        print("local topological properties done")

    # def get_network_attributes(self):
    #     print("start subgraph sampling...")
    #     random_walks = self.graph_sampling()
    #     print("getting global topological properties...")
    #     global_encoding = self.get_global_topological_properties()
    #     print("getting local topological properties...")
    #     shortest_path_length = self.get_shortest_path_length()
    #     with open(f"{self.ppi_path}/subgraphs.pkl", "wb")as f:
    #         pickle.dump(random_walks, f)
    #     with open(f"{self.ppi_path}/global_topological_properties.pkl", "wb")as f:
    #         pickle.dump(global_encoding, f)
    #     with open(f"{self.ppi_path}/local_topological_properties.pkl", "wb")as f:
    #         pickle.dump(shortest_path_length, f)

