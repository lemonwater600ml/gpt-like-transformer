from dataclasses import dataclass
from torch import nn as nn
from torch.nn import functional as f





class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # Q! # huggingface is using Conv1D for the FC 
        # https://github.com/huggingface/transformers/blob/44f6fdd74f84744b159fa919474fd3108311a906/src/transformers/models/gpt2/modeling_gpt2.py#L568
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        # Q! # Why no activation here
        
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
        

class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.attn(self.ln_2(x))
        return x
    
    
@dataclass
class GPT2Config:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384


class GPT2(nn.Module): 
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        
    
        # Q! why putting lm_head here instead of inside the transformer ModuleDict?
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    