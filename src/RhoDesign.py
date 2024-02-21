from typing import Any, Dict, List, Optional, Tuple, NamedTuple
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from gvp_transformer_encoder import GVPTransformerEncoder
from transformer_src.transformer_decoder import TransformerDecoder
from util import rotate, CoordBatchConverter, load_structure


class RhoDesignModel(nn.Module):
    """
    Architecture: Geometric GVP-GNN as initial layers, followed by
    sequence-to-sequence Transformer encoder and decoder.
    """

    def __init__(self, args, alphabet):
        super().__init__()
        encoder_embed_tokens = self.build_embedding(
            args,
            alphabet,
            args.encoder_embed_dim,
        )
        decoder_embed_tokens = self.build_embedding(
            args,
            alphabet,
            args.decoder_embed_dim,
        )
        encoder = self.build_encoder(args, alphabet, encoder_embed_tokens)
        decoder = self.build_decoder(args, alphabet, decoder_embed_tokens)
        self.args = args
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = GVPTransformerEncoder(args, src_dict, embed_tokens)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
        )
        return decoder

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.padding_idx
        emb = nn.Embedding(num_embeddings, embed_dim, padding_idx)
        nn.init.normal_(emb.weight, mean=0, std=embed_dim**-0.5)
        nn.init.constant_(emb.weight[padding_idx], 0)
        return emb

    def forward(
        self,
        coords,
        adjunct_coords,
        ss_ct_map,
        padding_mask,
        confidence,
        prev_output_tokens,
        return_all_hiddens: bool = False,
        features_only: bool = False,
    ):
        encoder_out = self.encoder(
            coords,
            adjunct_coords,
            ss_ct_map,
            padding_mask,
            confidence,
            return_all_hiddens=return_all_hiddens,
        )

        logits, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            return_all_hiddens=return_all_hiddens,
        )
        return logits, extra

    def sample(
        self,
        coords,
        ss_ct_map,
        _device,
        partial_seq=None,
        temperature=1.0,
        confidence=None,
    ):
        L = len(coords)
        # Convert to batch format
        batch_converter = CoordBatchConverter(self.decoder.dictionary)
        batch_coords, confidence, _, _, padding_mask, ss_ct_map = batch_converter(
            [(coords, confidence, None, ss_ct_map)]
        )
        batch_coords, confidence, padding_mask, ss_ct_map = (
            batch_coords.cuda(device=_device),
            confidence.cuda(device=_device),
            padding_mask.bool().cuda(device=_device),
            ss_ct_map.cuda(device=_device),
        )

        c = batch_coords[:, :, [0, 1, 2], :]  # the four backbone atoms
        adc = batch_coords[
            :, :, :, :
        ]  # eight atoms which are used to compute dihedral angles

        # Start with prepend token
        mask_idx = self.decoder.dictionary.get_idx("<mask>")
        sampled_tokens = torch.full((1, 1 + L), mask_idx, dtype=int)
        sampled_tokens = sampled_tokens.cuda(device=_device)
        sampled_tokens[0, 0] = self.decoder.dictionary.get_idx("<cath>")
        if partial_seq is not None:
            for i, c in enumerate(partial_seq):
                sampled_tokens[0, i + 1] = self.decoder.dictionary.get_idx(c)

        # Save incremental states for faster sampling
        incremental_state = dict()

        # Run encoder only once
        encoder_out = self.encoder(c, adc, ss_ct_map, padding_mask, confidence)

        # Decode one token at a time
        for i in range(1, L + 1):
            if sampled_tokens[0, i] != mask_idx:
                continue
            logits, _ = self.decoder(
                sampled_tokens[:, :i],
                encoder_out,
                incremental_state=incremental_state,
            )
            logits = logits[0].transpose(0, 1)
            logits /= temperature
            probs = F.softmax(logits, dim=-1)
            sampled_tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)
        sampled_seq = sampled_tokens[0, 1:]

        # Convert back to string via lookup
        all_data = [self.decoder.dictionary.get_tok(a) for a in sampled_seq]

        seq_data = []

        for i in all_data:
            if i not in ["A", "U", "C", "G"]:
                seq_data.append("X")
                continue
            seq_data.append(i)

        return "".join(seq_data)
