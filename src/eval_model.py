from RhoDesign import RhoDesignModel
from alphabet import Alphabet
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
from util import load_structure, extract_coords_from_structure, seq_rec_rate
import os
random.seed(1)

_device = 0

class args_class:  # use the same param as esm-if1, waiting to be adjusted...
    def __init__(self, encoder_embed_dim, decoder_embed_dim, dropout):
        self.local_rank = int(os.getenv("LOCAL_RANK", -1))
        self.device_id = [0, 1, 2, 3, 4, 5, 6, 7]
        self.epochs = 100
        self.lr = 1e-5
        self.batch_size = 1
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.dropout = dropout
        self.gvp_top_k_neighbors = 15
        self.gvp_node_hidden_dim_vector = 256
        self.gvp_node_hidden_dim_scalar = 512
        self.gvp_edge_hidden_dim_scalar = 32
        self.gvp_edge_hidden_dim_vector = 1
        self.gvp_num_encoder_layers = 3
        self.gvp_dropout = 0.1
        self.encoder_layers = 3
        self.encoder_attention_heads = 4
        self.attention_dropout = 0.1
        self.encoder_ffn_embed_dim = 512
        self.decoder_layers = 3
        self.decoder_attention_heads = 4
        self.decoder_ffn_embed_dim = 512


def eval(model,pdb_list,_device):
    """
    fpath: path to pdb file
    """
    test_path = './../data/test/'
    test_ss_path = './../data/test_ss/'
    model_path = './../checkpoint/ss_apexp_best.pth'
    
    model_dir=torch.load(model_path) 
    model.load_state_dict(model_dir)  
    model.eval()
    rc = []
    for pdb_name in tqdm(pdb_list):
        pdb_path = test_path + pdb_name
        ss_path = test_ss_path + pdb_name.split('.')[0] + '.npy'
        ss_ct_map = np.load(ss_path)
        pdb = load_structure(pdb_path)
        coords, seq = extract_coords_from_structure(pdb)

        pred_seq = model.sample(coords,ss_ct_map,_device,temperature=1e-5)
        rc_value = seq_rec_rate(seq,pred_seq)
        rc.append(rc_value)
    
    print('recovery rate: ' + str(np.mean(rc)))

if __name__ == '__main__':
    args = args_class(512,512,0.1)
    dictionary = Alphabet(['A','G','C','U','X'])
    model = RhoDesignModel(args, dictionary).cuda(device=_device)
    pdb_list = os.listdir('./../data/test/')
    eval(model,pdb_list,_device)