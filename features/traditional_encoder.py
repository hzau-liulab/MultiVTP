import torch
import pandas as pd

class TraEncoder:
    def __init__(self, tradf_path = "../input/tra_z_score_species.csv", trainset_path="../datasets/virus_species"):
        self.traditional_df = pd.read_csv(f"{tradf_path}", index_col=0)
        trainset = open(f"{trainset_path}/train.txt", 'r').read().split("\n")
        valid_indices = self.traditional_df.index.intersection(trainset)
        self.padding = torch.tensor(self.traditional_df.loc[valid_indices].mean(axis=0))

    def get_tra_features(self, uniprotid):
        if uniprotid in self.traditional_df.index.tolist():
            tra_features = self.traditional_df.loc[uniprotid]
            return torch.tensor(tra_features)
        else:
            return self.padding


