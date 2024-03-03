import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from dataclasses import dataclass
from typing import Optional


# ---------------- Model Arguments ---------------- #

@dataclass
class ModelArgs:
    dim:int=4096
    n_layers:int=32
    # Number of heeads for the queries
    n_heads:int=32
    # Number of heads for K and V
    vocab_size:int=-1
    
    n_kv_heads:Optional[int]=None
    # Set during loading tokenizer
    multiple_of:int=256 
    ffn_dim_multiplier:Optional[float]=None
    norm_eps:float=1e-5

    # V cache
    max_batch_size:int=32 
    max_seq_len:int=2048 
    device:str=None


# ------------------------------------------------- #


# ----------------   Transformer   ---------------- #
    
class Transformer(nn.Module):

    def __init__(self, args:ModelArgs) -> None:
        super().__init__()

        ## Verify that vocabulary size is set
        assert args.vocab_size!=-1, "Vocabulary size must be set"

        self.args=args
        self.vocab_size=args.vocab_size
        self.n_layers=args.n_layers
        self.tok_embeddings=nn.Embedding(self.vocab_size,args.dim)

        self.layers=nn.ModuleList()

        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm=RMSNorm(args.dim,eps=args.norm_eps)

        self.output=nn.Linear(args.dim,self.vocab_size,bias=False)

        self.freqs_complex=precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len*2,device=self.args.device)

    def forward(self,tokens:torch.Tensor, start_pos:int):
        # (B, Seq_Len)
        batch_size,seq_len=tokens.shape
        assert seq_len==1, "One token to process at a time"

        


# ------------------------------------------------- #