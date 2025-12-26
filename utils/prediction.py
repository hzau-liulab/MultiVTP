import pandas as pd
from typing import List, Union
from get_features import load_features
from MultiVTP_model import *
import torch.nn.functional as F
import argparse


virus_order = ['H1N1', 'HIV-1', 'HHV-4', 'ZIKV', 'DENV-2', 'H3N2', 'HHV-8', 'HPV16', 'HPV18', 
               'HHV-1', 'HPV8', 'HPV31', 'HCV-1b', 'MeV', 'HCV', 'HPV6b', 'HHV-8P', 'HPV11', 
               'H5N1', 'LCMV', 'SARS-CoV-2', 'SV40', 'HPV9', 'HPV5', 'HCV-H']
parser = argparse.ArgumentParser()
parser.add_argument('--query_proteins', nargs='+', type=str)
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()
device = torch.device(args.device)

def predict_protein_scores(query_proteins: List[str]):
    features = load_features(query_proteins, device)
    prediction_scores = []
    model = MultiVTP()
    model.to(device)
    model.eval()
    
    with torch.no_grad(): 
        for cv in range(5):
            weight_path = f"../model_weights/MutilVTP_species_weights{cv}.pth"
            model.load_state_dict(torch.load(weight_path, map_location=device))
            out = model(features)
            prediction_scores.append(out)
    
    vtp_score = F.sigmoid(torch.stack(prediction_scores).mean(dim=0))
    vtp_df = pd.DataFrame(vtp_score.cpu().detach().numpy(), index=query_proteins, columns=virus_order)
    vtp_df.to_csv("../outputs/vtp_score.csv")


if __name__ == "__main__":
    QUERY_PROTEINS = args.query_proteins
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result_df = predict_protein_scores(
        query_proteins=QUERY_PROTEINS,
    )
    print("finished")



