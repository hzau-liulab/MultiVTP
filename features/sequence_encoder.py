import torch
import esm

class SequenceEncoder:
    def __init__(self, esm2_path = "../pretrain_model/ESM2/esm2_t36_3B_UR50D.pt"):
        self.esm2_model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(esm2_path)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.esm2_model.eval()

    def get_embeddings(self, sequence):
        _, _, batch_tokens = self.batch_converter([("_" ,sequence)])
        with torch.no_grad():
            embeding36 = self.esm2_model(batch_tokens, repr_layers=[36], need_head_weights=False)["representations"][36][0][1:-1].mean(dim=0)
            embeding35 = self.esm2_model(batch_tokens, repr_layers=[35], need_head_weights=False)["representations"][35][0][1:-1].mean(dim=0)
            embeding34 = self.esm2_model(batch_tokens, repr_layers=[34], need_head_weights=False)["representations"][34][0][1:-1].mean(dim=0)
        return torch.cat([embeding34, embeding35, embeding36], dim=0)


