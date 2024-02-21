from RhoDesign_without2d import RhoDesignModel
from alphabet import Alphabet
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
from util import load_structure, extract_coords_from_structure, seq_rec_rate
import os
import argparse
random.seed(1)



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


def eval(model,pdb_path,save_path,_device,temp=1e-5):
    """
    fpath: path to pdb file
    """

    model_path = './../checkpoint/no_ss_apexp_best.pth'

    name = pdb_path.split('/')[-2]
    
    model_dir=torch.load(model_path) 
    model.load_state_dict(model_dir)  
    model.eval()
    rc = []
    

    pdb = load_structure(pdb_path)
    coords, seq = extract_coords_from_structure(pdb)

    

    pred_seq = model.sample(coords,_device,temperature=temp)
    rc_value = seq_rec_rate(seq,pred_seq)
    rc.append(rc_value)
    with open(os.path.join(save_path,f'{name}_without2d.fasta'),'w') as f:
        f.write(f'>{name}_without2d'+'\n')
        f.write(pred_seq+'\n')
    
    print('sequence: ' + pred_seq)
    print('recovery rate: ' + str(np.mean(rc)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('-pdb', '--pdb_file', type=str,help='path to the pdb file',required=True)
    parser.add_argument('-save', '--save_path', type=str, help='path to the save directory',required=True)
    parser.add_argument('-device', '--device', default=0, type=int, help='Assign the device to run the model')
    parser.add_argument('-temp', '--temperature', default=1, type=float, help='temperature for sampling')
    args = parser.parse_args()

    pdb = args.pdb_file
    save_path = args.save_path
    _device = args.device
    temp = args.temperature

    model_args = args_class(512,512,0.1)
    dictionary = Alphabet(['A','G','C','U','X'])
    model = RhoDesignModel(model_args, dictionary).cuda(device=_device)
    eval(model,pdb,save_path,_device,temp)