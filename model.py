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