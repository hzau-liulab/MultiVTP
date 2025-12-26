# Description 
MultiVTP is a multilabel model for virus target protein prediction that integrates graph representation learning with multimodal information. The algorithm first samples subgraphs centered on query proteins via random walks to capture neighborhood contexts. It then extracts both network topological features and multimodal representations for each node. Following extraction, these features are integrated within subgraphs through a graph transformer module. Finally, the Progressive Layered Extraction (PLE) module is introduced to process the embeddings for each virus prediction task, enabling multilabel prediction.

# Usage
## Installation 
1. Clone the repository to your local device.
```shell
git clone https://github.com/hzau-liulab/MultiVTP   
cd MultiVTP
```
2. Install the necessary dependencies.
* Python packages
   ```text
   python                3.9.0    
   Numpy                 1.24.3     
   Pandas                2.2.2              
   fair-esm              2.0.0      
   pytorch               2.2.0    
   DGL                   2.4.0
   scikit-learn          1.3.0    
   pygosemsim            0.1.0
   transformers          4.44.2
   networkx              3.2.1
   node2vec              0.5.0
   ```
   We recommend creating a new conda environment for MultiVTP, and then install the required packages within this environment.
   ```shell
   conda create -n MultiVTP python=3.9.0
   conda activate MultiVTP
   ```
3. Download the pre-trained model weights            
Our pre-trained model weights can be found at [MultiVTP model](https://drive.google.com/drive/folders/1h8jA142_CyDYjMXoiWPJVtuTY9yCz0zq?usp=drive_link)         
ESM-2 [esm2_t36_3B_UR50D](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt)        
PubMedBERT [PubMedBERT](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/tree/main)

Pre-trained model weights need move to the `./model_weights` folder. ESM-2 and PubMedBERT should be placed in the `./pretrain_models/ESM2` and `./pretrain_models/PubMedBERT` directories, respectively.
        

