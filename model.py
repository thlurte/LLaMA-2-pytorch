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




def precompute_theta_pos_frequencies(head_dim:int,seq_len:int,device:str,theta:float=10000.0):
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"

    # based on the formula theta_i = 10000^(-2(i-1)/dim) for i = [1,2,3,... dim/2]
    # theta paramets
    # Shape: (head_dim / 2)
    theta_numerator=torch.arange(0,head_dim,2).float()
    # Shape: (head_dim / 2)
    theta=1.0/(theta**(theta_numerator/head_dim)).to(device)
    # Construct the positions
    # Shape: (seq_len)
    m=torch.arange(seq_len,device=device)
    # multiply each theta by each position using the outer product
    # Shape: (seq_len) outer_product (head_dim/2) -> (seq_lem,head_dim / 2) 
    freqs=torch.outer(m,theta).float()
    # polar form c = R * exp(i*m*theta)
    # (seq_len,head_dim/2) -> (seq_len,head_dim/2)
    freqs_complex=torch.polar(torch.ones_like(freqs),freqs)
    return freqs_complex

def apply_rotary_embeddings(x:torch.tensor, freqs_complex: torch.tensor, device: str):
    # (B, seq_len, H, head_dim) -> (B, seq_len, H, head_dim/2)
    x_complex=torch.view_as_complex(x.float().reshape(*x.shape[:-1],-1,2))
    # (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
    freqs_complex=freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B, seq_len, H, head_dim/2) * (1, seq_len, 1, head_dim/2) = (B, seq_len, H, head_dim/2) 
    x_rotated=x_complex*freqs_complex
    # (B, seq_len, H, head_dim/2) -> (B, seq_len, H, head_dim/2, 2)
    x_out=torch.view_as_real(x_rotated)
    # (B, seq_len, H, head_dim/2, 2) -> (B, seq_len, H, head_dim/2)
    x_out=x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

# ----------------  Encoder Block  ---------------- #

class EncoderBlock(nn.Module):
    def __init__(self,args:ModelArgs) -> None:
        super().__init__()
        self.n_heads=args.n_heads
        self.dim=args.dim
        self.head_dim=args.dim//args.n_heads

        self.attention=SelfAttention(args)
        self.feed_forward=FeedForward(args)

        # Normalization before self-attention 
        self.attention=RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization attention 
        self.ffn_norm=RMSNorm(args.dim,eps=args.norm_eps)

    def forward(self, x:torch.Tensor, start_pos:int, freq_complex:torch.Tensor):
        # (B, seq_len, dim)  + (B, seq_len, dim) --> (B, seq_len, dim) 
        h=x+self.attention.forward(self.attention_nom(x),start_pos,freq_complex) 
        out=h+self.feed_forward.forward(self.ffn_norm(h))
        return out 


# ------------------------------------------------- #

# ----------------     RMSNorm     ---------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps: float=1e-6) -> None:
        super().__init__()
        self.eps=eps
        # gamma parameter
        self.weight=nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, seq_len, dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)
    
    def forward(self,x:torch.Tensor):
        # (dim) * (B, seq_len, dim) -> (B, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)

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
        # (B, seq_len)
        batch_size,seq_len=tokens.shape
        assert seq_len==1, "One token to process at a time"

        # (B, seq_len) --> (B, seq_len, dim)
        h=self.tok_embeddings(tokens)

        # Retrieve the pairs (m,theta) corresponding to the positions [start_pos, start_pos+seq_len]
        freq_complex=self.freqs_complex[start_pos:start_pos+seq_len]

        # Apply encoder layers consectuively
        for layer in self.layers:
            h=layer(h,start_pos,freq_complex)

        h=self.norm(h)
        output=self.output(h).float()

        return output

# ------------------------------------------------- #