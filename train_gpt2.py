from dataclasses import dataclass
import inspect
import torch
import math
from torch import nn as nn
from torch.nn import functional as F
import tiktoken 
import time
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not a "bias", more of a mask. (future masking)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        # TODO need to delete the buffer
        
        
    def forward(self, x):
        B, T, C = x.size() # batch_sz, sequence_length, n_embd
        # nh: number of head. hs: head size. 
        # C: number of channels = nh * hs
        # GPT-2 124M, n_head=12, hs=64, nh*hs=C=768 channels
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # making nh to the batch dim so it 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # parallelizable
        # attention (materializes the large (T, T) matrix for all the queries and keys)
        
        
        # -- attention
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) 
        # # Q! k.size(-1): "scaled" dot product, scaling factor. the sqrt of k's dim
        # # Q! @: __imatmul__
        # # Q! .transpose(-2, -1);    .transpose(1, 2) vs .transpose(2, 1)
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # use -inf because the softmax will zero out it
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) matmul (B, nh, T, hs) -> (B, nh, T, hs)
        
        # -- flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
                
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs
        y = self.c_proj(y)
        # Q! can ignore this output project?
        return y     
        

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # Q! # huggingface is using Conv1D for the FC 
        # diff between conv1d & vanilla nn
        # https://github.com/huggingface/transformers/blob/44f6fdd74f84744b159fa919474fd3108311a906/src/transformers/models/gpt2/modeling_gpt2.py#L568
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # Q! # Why no activation here 
        # Q! does output project do anything
        
    
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
        x = x + self.mlp(self.ln_2(x))
        return x
    
    
@dataclass
class GPT2Config:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50k BPE merges + 256 bytes tokens + 1 <|endoftext|> # tokenizer video
    n_layer: int = 12 # 
    n_head: int = 12
    n_embd: int = 768


class GPT2(nn.Module): 
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd), # block_size is sequence length (1024)
            h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        
    
        # Q! why putting lm_head here instead of inside the transformer ModuleDict?
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) 

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight # copy the data_ptr (reference) and replace
                                                            # the old wte.weight value will become orphant and abandon

        # init params
        self.apply(self._init_weights) # apply the function to all submodule    
        # Q! apply where? submodule used in the forward? __init__? or any submodule in the class
        
        
    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            # Q! diff between torch.nn.init.normal and torch.nn.init.normal_
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif(isinstance(module, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)        
                

    def forward(self, idx, target=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape(T); range in pytorch. create the indices
        # Q! is torch.arange returning tensor? is setting device necessary for creation of each tensor
        tok_embd = self.transformer.wte(idx)    # token embeddings of shape (B, T, n_embd)
        pos_embd = self.transformer.wpe(pos)    # position embeddings of shape (T, n_embd)
        x = tok_embd + pos_embd
        
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1)) 
            # .view(-1, logits.size) the fst -1 is B * T, calculating automatically
        return logits, loss
    
    # cross entropy loss = train initial test
        """
using device: mps
tensor(10.9361, device='mps:0', grad_fn=<NllLossBackward0>)
torch.Size([4, 32, 50257])
at the initialization, we're hoping each token is getting a rough uniform probability
We'd wish the probability of each token be 1/50257 in this case
Sanity check the loss
-ln(1/50257) = 10.8249051197 -> the expected initial loss
        """
        
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPT2Config(**config_args)
        model = GPT2(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    
    def configure_optimization(self, weight_decay, learning_rate, device): 
        param_dict = { pn: p for pn, p in self.named_parameters() }
        param_dict = { pn: p for pn, p in param_dict.items() if p.requires_grad }
        
        decay_params = [ p for pn, p in param_dict.items() if p.dim() >= 2]
        non_decay_params = [p for pn, p in param_dict.items() if p.dim() < 2]
        
        optim_group = [
            {"params": decay_params, "weight_decay": 0.01},
            {"params": non_decay_params, "weight_decay": 0}
            ]
        
        num_decay_params = sum( p.numel() for p in decay_params )
        num_non_decay_params = sum( p.numel() for p in non_decay_params )
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(non_decay_params)}, with {num_non_decay_params:,} parameters")
        
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        print(f"Using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(params=optim_group, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -------------------------------- Devices && DDP  ------------------------------------------
#

# DDP launch for 8 GPU
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# set up DDP
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1) != -1)
if ddp:
    assert torch.cuda.is_available(), "ddp required cuda"
    ddp_rank = int(os.environ.get("RANK"))
    ddp_local_rank = int(os.environ.get("LOCAL_RANK"))
    ddp_world_size = int(os.environ.get("WORLD_SIZE"))
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)
    master_process = ddp_rank == 0 # for logging purpose and checkpointing
else: 
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = "mps"

# DEVICE = "cpu"
print(f"using device: {DEVICE}")

# Testing DDP
# print("I am GPU", ddp_rank)
# if ddp:
#     destroy_process_group()
# import sys; sys.exit(0)

# --------------------------------- Test Pretrain --------------------------------------

# # Test the loading pretrainined weight 
# model = GPT2.from_pretrained('gpt2')
# print("didn't crash yay!")

# --------------------------------- DataLoader --------------------------------------

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        
        with open("input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"number of tokens {len(self.tokens)}")
        # print(f"1 epoch = {len(self.tokens) // (B * T * self.num_processes)} batches")
        self.curr_pos = self.B * self.T * self.process_rank
        
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.curr_pos:self.curr_pos + (B * T) + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        
        # Advance the position in the tensor
        self.curr_pos += B * T * self.num_processes
        # if loading the next batch would be our of bounds, reset
        if self.curr_pos + (B * T * self.num_processes + 1) >= len(self.tokens):
            self.curr_pos = B * T * self.num_processes
        return x, y

# --------------------------------- DataLoader --------------------------------------

def synchronize(DEVICE):
    if DEVICE == "cpu":
        torch.cpu.synchronize()
    elif DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()
        

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


# --------------------------------- Gradient accumulation ---------------------------------
total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 2 # micro batch size
T = 1024 # 
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# --------------------------------- ---------------------------------


torch.set_float32_matmul_precision("high") # highest: FP32, high: TP32

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)
# B=2 for mps is about 1000 ms for a step

# get logits & loss
model = GPT2(GPT2Config(vocab_size=50304))
model.to(DEVICE)
# model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# Learning rate scheduling
MAX_LR = 6e-4
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 10
MAX_STEPS = 50
def get_lr(it):
    if it < WARMUP_STEPS:
        return MAX_LR * (it + 1) / WARMUP_STEPS
    if it > MAX_STEPS:
        return MIN_LR
    decay_ratio = (it - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    assert 0 <= decay_ratio and decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


# optimization
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)  # 3e-4 is a good starting point at early debugging stage
optimizer = raw_model.configure_optimization(weight_decay=0.1, learning_rate=6e-4, device=DEVICE)
# AdamW is Adam fixing a bug-like thing
# first moment is like momentum. second moment is like RMSProp


for step in range(MAX_STEPS):
    t0 = time.time()
    optimizer.zero_grad() # always remember to start with zero gradient
    loss_accum = 0.0 # grad_accum
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        # with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        #     logits, loss = model(x, y)
            # import code; code.interact(local=locals())
        
        """ 
        we have to scale the loss to account for gradient accumulation
        because the gradient just add on each successive backward()
        addition of gradient corresponds to a SUM  in the object, but
        instead of a SUM, we want MEAN. Scale the loss here so it comes out right
        """
        logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach() # detach this from tensor because we need value only
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    lr = get_lr(step)
    for param in optimizer.param_groups:
        param['lr'] = lr
    
    optimizer.step() # one step to optimize the weight
    synchronize(DEVICE) # wait for the gpu to finish finish work
    t1 = time.time()
    dt = (t1 - t0)
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():<9.6f} | lr: {lr:4e} | norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}") 

# Q! data loader lit
# loss drop from 11 to 6
# expect the loss to come down but not to much
# in the 50257 tokens, many of them never occur in the dataset
# easy gain to make to the optimization
# for example. take the bias of all the logits that never occur and driving them to negative infinity.
# All these unicode, they never occur. So the probability should be very low.
# The loss gain we're seeing are alone the line of the leading usage of tokens that never occur 
# This is probably the most gain that we're seeing.
# Q! in each batch, the model is driving all the logits of tokens that is not seeing in this batch to negative infinity?
# because they are not seeing in this batch. their probability is low.
if ddp:
    destroy_process_group()
    
import sys; sys.exit(0) 



# --------------------------------- parameter sharing wte and lm_head (weight sharing) --------------------------------- 
# print(sd_hf["lm_head.weight"].shape)            # 50257, 768. head. take the 768 channel and upscale to 50257 to get the 
#                                                 # logit of next token
# print(sd_hf["transformer.wte.weight"].shape)    # 50257, 768. Give the token embedding in the bottom

# # Trick: pytorch's pointer 
# print(sd_hf ["lm_head weight"].data_ptr())         # 140650503360576
# print(sd_hf ["transformer-wte.weight"].data_ptr())  # 140650503360576


# common weight tieing scheme: (weight sharing scheme) from [Attention is all you need paper] (softmax section), 
# it cited from [using the output embedding to improve language model]
# two matrixes are shared, tied, and they're the same matrix
# the original paper code simply use wte again to get the logit


# Trick!
# Q! trick!!!!
# wish the two matrices to behave similar in the following sense
# if two token are similar. presumebly we expect they are nearby in the token embedding space
# if two token are the same, expect it has the same probability at the output of the transformer because they are semantically similar
# similar token should have similar embeddings or similar weights
# in the [Using the output embedding to improve language model] paper, they observed performance increasing


# The gradient flow go to lm_head (wte) when starting backprop 
# And it goes to lm_head (wte) again when it reaches the bottom of the transformer


# --------------------------------- train with small batch for debugging purpose -----------------------------------------

# enc = tiktoken.get_encoding('gpt2')

# with open('input.txt', 'r') as f:
#     text = f.read()
# text = text[:1000]
# tokens = enc.encode(text)
# B, T = 4, 32
# buf = torch.tensor(tokens[:B*T + 1], device=DEVICE) # tensor should match the device
# # the other way to move tensor to the device, notice the equal
# # buf = torch.tensor(tokens[:B*T + 1])
# # buf = buf.to(device) # if not equal, it's just stateful
# x = buf[:-1].view(B, T)
# y = buf[1:].view(B, T)

# # get logits & loss
# model = GPT2(GPT2Config)
# model.to(DEVICE)
# # logits, loss = model(x, y)

# # optimization
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)  # 3e-4 is a good starting point at early debugging stage
# # AdamW is Adam fixing a bug-like thing
# # first moment is like momentum. second moment is like RMSProp
# for i in range(50):
#     optimizer.zero_grad() # always remember to start with zero gradient
#     logits, loss = model(x, y)
#     loss.backward()
#     optimizer.step() # one step to optimize the weight
#     print(f"step {i}, loss: {loss.item()}") # loss is a 1-dim tensor living on gpu
#                                             # loss.item() makes pytorch to ship it to cpu and memory
#                                             # as a float that we can just print

# # print(loss)
# # print(logits.shape)
# import sys; sys.exit(0) 
                    
# ------------------------- Test the generation -----------------------------------
# Generate the sample generation output
# num_return_sequences = 5
# max_length = 30

# model = GPT2.from_pretrained('gpt2')      
# # model = GPT2(GPT2Config())
# model.eval() # when there is no need for back prop
# model.to(device)
# # print("shipped model but didn't crash yay!")

# import tiktoken 
# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'm a language model, ")
# tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
# x = tokens.to(device)
# # print("shipped tensor")

# # generate! right now x is (B, T) where B = 5, T = 8
# # set the seed to 42
# torch.manual_seed(42)
# if device == "cuda":
#     torch.cuda.manual_seed(42)    
# elif device == "mps":
#     torch.mps.manual_seed(42)

# while x.size(1) < max_length:
#     # forward the model to get the logits
#     with torch.no_grad():
#         logits = model(x) # (B, T, vocab_size)
#         # take the logits at the last position
#         logits = logits[:, -1, :] # (B, vocab_size) # inefficient implementation of sample
#         # get the probabilities
#         probs = F.softmax(logits, dim=-1)
#         # do top-k sampling of 50 (huggingface pipeline default)
#         # topk_probs here becomes (5, 50), topk_indices is (5, 50)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#         # select a token from the top-k probabilities
#         ix = torch.multinomial(topk_probs, 1) # (B, 1)
#         # gather the corresponding indices
#         xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
#         # append to the sequence
#         x = torch.cat((x, xcol), dim=1)


# # print the generated text
# for i in range(num_return_sequence
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)


# ------------------------ model initialization: std 0.02, residual init ------------------------
""" 
nn: normal distribution: 0.02 stddev
bias, 0 (pytorch default bias initialization is uniform)
"""

# Q! check Xavier initialization
# Xavier initialization: one of the sqrt of number of features that are incoming into this layer
# 768: 1/sqrt(768): 0.03
# 1600: 1/sqrt(1600): 0.025 (dimension of the biggest model in gpt2)

# trick
# it's not completely crazy to be hard coding 0.02 here
# something that is able to grow with the model size instead 


wte: 0.02
pos: 0.01

# layernorm needs initialization and has parameter: torch default layernorm scale:1 offset: 0
# ------------------------------------ residual init ------------------------------------------------
"""
Every residual link contribute some amount and gets added.
The variance of the the activation in the residual stream grows. 
Scaling down by 1 over (/) square root of number of layer
This is a way to control the growth of activation inside the residual stream in the forward path

# Q! this is controlling the growth of initialization
# should we control the forward as well? e.g. implement this in the forward
# Would it hurt the result of residual?

# Q! why only deal with the residual of MLP but not residual in attention

torch.randn: normal distribution zero mean one standard deviation

# example
x = torch.zeros(768)
n = 100
for i in range(n):
    x += torch.randn(768)
print(x.std()) # ensor(10.2252)


The scaling factor used here exactly compensates for that growth

# example
x = torch.zeros(768)
n = 100
for i in range(n):
    x += n**(-0.5) * torch.randn(768)
print(x.std()) # ensor(1.0082)

"""
# ----------------------------------- Section2 ------------------------------------------------
# ----------------------------------- let's make it fast, mixed precision 1000 ms-----------------------------------
"""
# get an interactive console
import code; code.interact(local=locals())
default: torch.float32
"""

"""
Deep neural network can tolerate small precision. but not all workflow can tolerate it

sparcity is not widely used so look at non-sparcity 

int8 is used for inference. not training. it has uniform spacing 
Need float so we get a better match to the normal distribution that occur during training of NN
Both activation and weights are distributed as a normal distribution.
float point are important to match that representation

If the tensor has smaller bits. it's easier to move around

Many of NN workflow are memory-bound. Size and bandwidth (speed) is precious
most of the tensor cores, most of time are idle becuase we cannot feed data fast enough from memory 
if getting utilization of 60% is good.

half of time of the well-tuned application, the tensor cores are not doing multiplies because the dat is not available.

Trick!
Q! mixed precision optimization?
"""
# ----------------------------------- Tensor cores, timing the code, TF32 precision, 333 ms-------------------------------------------------------------
"""
tensor cores is an instruction in the a100 architecture

everything is broken up into the 4*4 instructions because it's the fastest way to multiply matrices
Most of work are nn.Linear
768 - matrix multiplication -> 5020267 dominates anything happend in the network (output classifier)

A100 - white paper (NVIDIA A100 Tensor Core GPU Architecture)

# Q!
Diff between FP32 and TF32, INPUT OPERANDS Accumulator

FT32
internal representation within the instruction when accumulation occurs
the intermediate plus equals (+=) of the intermediate little vector multiplies that all happens in FT32


IF can tolerate a little precision fudge, the speed up is free.
TF32 is internally cropped 13 mantissa bits whe calculation. But the input, output, etc are still 32 bits
8X (8 times) faster and it's a bit more approximate


By default, max out the batch size that fits the gpu





"""

# ----------------------------------- precision -----------------------------------
"""
     sign bit   Range (exponent)    Precision (mantissa)
FP32 1          e8                  M23
TF32 1          e8                  m10
FP16 1             e5               m23
BF16 1          e8                  m7
"""
# ----------------------------------- sec 2 - optimization: enable TF32 -----------------------------------
"""
crop out some of the precision inside the operation it self
is TF32 available on mps? Nope! it's something on Ampere architecture (A100)

step 48, loss: 6.862908363342285, dt: 965.22ms, tok/sec: 2121.79559913023
step 49, loss: 6.957481384277344, dt: 953.73ms, tok/sec: 2147.367690144732
== after applying change malmul FP32 to TF32 ==
step 47, loss: 6.962921619415283, dt: 980.61ms, tok/sec: 2088.49
step 48, loss: 6.862908363342285, dt: 962.69ms, tok/sec: 2127.37
step 49, loss: 6.957481384277344, dt: 958.45ms, tok/sec: 2136.79


[Google colab with T4]
step 0, loss: 10.892457008361816, dt: 911.94ms, tok/sec: 2245.75
step 1, loss: 9.65002155303955, dt: 839.31ms, tok/sec: 2440.09
step 2, loss: 9.463541030883789, dt: 835.93ms, tok/sec: 2449.98
...
tep 47, loss: 6.962922096252441, dt: 885.82ms, tok/sec: 2311.99
step 48, loss: 6.862907886505127, dt: 891.89ms, tok/sec: 2296.26
step 49, loss: 6.957481861114502, dt: 882.76ms, tok/sec: 2320.00
== activate TF32
step 0, loss: 11.089818954467773, dt: 871.80ms, tok/sec: 2349.16
step 1, loss: 9.81151294708252, dt: 845.50ms, tok/sec: 2422.23
step 2, loss: 9.313061714172363, dt: 838.16ms, tok/sec: 2443.44
...
step 47, loss: 6.679123878479004, dt: 913.00ms, tok/sec: 2243.16
step 48, loss: 6.602997779846191, dt: 910.33ms, tok/sec: 2249.73
step 49, loss: 6.542574405670166, dt: 907.41ms, tok/sec: 2256.98



3x for Andrej's 8 A100 cluster

even if multiplication is TF32. it's still FP32 shipping around the memory system. 
It's just costing us way too much time to shuttle around all this data.
even though we made the multiply itself mush faster. we're memory bound and we're not actually seeing the full benefit.
"""
# ----------------------------------- sec2 - opti - bfloat16 -----------------------------------
"""
still memory bound. still moving around all these float
decrease the amount of stuff that we're moving around. 

BF16
the range of 
few possibilities within that range because we are truncating the mantissa

FP32->FP16, the range changed. need gradient scalers (a whole video by itself)
FP16 came first. In Volta series which is before Amper
Everyone needed to use the gradient scaler at the time. additional source of state and complexity.
The reason is that the range is reduced in FP16 (IEEE FP16 sepc)
And NVIDIA made BF16 on Ampere. truncate the mantissa instead of exponent.  
Still the same range so no gradient scaler. everything is much simpler

# Trick!
For Mixed Precision on pytorch, recommend the article of "PyTorch Recipes > Automatic Mixed Precision"
Ignore gradient scaler. Look at torch.autocast


* "autocasting" (in "torch.amp")
    do not call half() or bfloat16(). 
* only surround the forward pass of the model and the loss calculation. 
  Leave the loss.backward() and optimizer.step() alone
    with torch.autocast(device_type=DEVICE):
            output = model(input)
            loss = loss_fn(output, target)

# Q! why bfloat16 cannot be used at backward

* pytorch doesn't support bfloat16 on mps yet 
* take super long time on cpu 


step 0, loss: 10.882240295410156, dt: 320835.29ms, tok/sec: 3.19
step 1, loss: 9.616708755493164, dt: 323324.63ms, tok/sec: 3.17
step 2, loss: 9.276605606079102, dt: 318698.48ms, tok/sec: 3.21

[colab with L4, 4 batches, 1024]
number of tokens 338025, 1 epoch = 82 batches
step 0, loss: 10.909079551696777, dt: 955.17ms, tok/sec: 4288.26
step 1, loss: 9.649524688720703, dt: 877.77ms, tok/sec: 4666.35
step 2, loss: 9.432360649108887, dt: 876.19ms, tok/sec: 4674.78
=== apply autocast on bfloat16
step 0, loss: 10.909454345703125, dt: 734.52ms, tok/sec: 5576.44
step 1, loss: 9.65005111694336, dt: 617.70ms, tok/sec: 6631.06
step 2, loss: 9.434700012207031, dt: 632.62ms, tok/sec: 6474.66


>>> logits.dtype
torch.bfloat16

logits is changed to bfloat16
but not everything changed. for example, 
>>> model.trainsformer.wte
Embedding(50257, 768)
>>> model.trainsformer.wte.weight
tensor([[9.9743e-05, .....]]) still torch.float32

Mixed precision
on the autocast page [CUDA Ops that can autocast to float16]
only matrix multiply like operations get converted to float16. 
a lot of them might not be able to get converted
softmax, layernorm, log?  
operation that are susceptible to precision changes will remain at float32

Andrej: 333 ms -> 300 ms; 50k token/sec -> 55k token/sec
empirically in many cases. this is a worth it tradeoff.
because it allows running fast. we can train longer to make up the precision decrease
https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/
"""
# ----------------------------------- torch compile -----------------------------------
"""
mps batch:2 T: 1025; FP32 high
step 0, loss: 10.892457008361816, dt: 1645.70ms, tok/sec: 1244.45
step 1, loss: 9.65002155303955, dt: 1359.61ms, tok/sec: 1506.31
step 2, loss: 9.463541030883789, dt: 1335.48ms, tok/sec: 1533.53
==
NO SUPORT

cpu batch:@ T:1025 FP32 high
step 0, loss: 10.892457008361816, dt: 4414.26ms, tok/sec: 463.95
step 1, loss: 9.65002155303955, dt: 3852.65ms, tok/sec: 531.58
step 2, loss: 9.463541030883789, dt: 3750.96ms, tok/sec: 545.99
...
step 47, loss: 6.962922096252441, dt: 4313.06ms, tok/sec: 474.84
step 48, loss: 6.862907886505127, dt: 3741.33ms, tok/sec: 547.40
step 49, loss: 6.9574809074401855, dt: 3719.97ms, tok/sec: 550.54
===
step 0, loss: 10.892455101013184, dt: 26375.43ms, tok/sec: 77.65
step 1, loss: 9.650020599365234, dt: 3551.08ms, tok/sec: 576.73
step 2, loss: 9.463541984558105, dt: 3498.62ms, tok/sec: 585.37
...
step 47, loss: 6.962923049926758, dt: 3808.85ms, tok/sec: 537.70
step 48, loss: 6.862907886505127, dt: 4257.64ms, tok/sec: 481.02
step 49, loss: 6.9574809074401855, dt: 4243.64ms, tok/sec: 482.60

"""

""" Remove python interpreter
speedup mainly comes from reducing python overhead and GPU read/writes.

The python interpreter use eager mode. It materializes all the operations layer-by-layer as it goes through.
The calculation are dispatched and run in this order. 
The python interpreter doesn't know what kind of operations are going to happen later

Torch compile sees the entire code at the same time. Is able to know what operation you intend to run.
It will optimize the process.
Take out of the python interpreter. Takes the model and compiles it as a single object with no python interpreter involved

"""


""" GPU kernel fusion / GPU read-write
Reduce the round trip between gpu and gpu memory

class TanhGELU(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


read .pow first
Ask to dispatch a kernel, that takes the input, raise it to the 3rd power, 
when the kernel is running, the gpu memory stores the input,

The input goes from gpu memory to the gpu. To all the cores, caches, and registers on the chip of GPU
Then, save the result to the memory.
It's this travel time that causes a lot of issues

when multiply with the constant 0.044715, it's another travel. Dispatch another kernel.


without torch.compile, it doesn't have context and process sequentially.
with torch.compile, it knows that data can stay on gpu until all computations are completed. 
only one travel needed. so everything get speed up. 
don't need to pay several memory bandwidth cost.


# Q!
# trick!
NVIDIA: https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
Ampere architecture blog 2000: https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
"""
# ----------------------------------- flash attention -----------------------------------
""" FlashAttention: Fast and Memory-Efficient Exact Attention

# Q! what's "materialize"
N*N is never materialized

# trick
torch compile was not able the 

for each batch element, we get t by t attention. the attention is never materialized.
use only softmax trick. 
incrementally evaluate a softmax without having to realize all of the

Original: Online normalizer calculation for softmax ->
FlashAttention

fuse all operations with the online softmax update into a single fused kernel flash attention
be aware of memory hierarchy.
flop don't really matter. 
The entire memory access pattern matters
torch compiler is amazing but there are still optimization available to us that torch compile cannot find.

scaled_dot_product_attention() : compound operation in pytorch


MPS
1570 tok/s -> 1800 tok/s 
"""

# ----------------------------------- Ugly number -----------------------------------

""" 
torch.compile cannot find now. 4% increase 

increase the ugly odd number to the multiple of 4, 8, 16, 64, or even 128
wte layer: increase the vocab size doesn't hurt the the result at the wte layer
classifier layer (lm_head): wte weight is reused here. there are additional tokens 
    predicting for tokens that are never presenting in the training data
    the NN network has to learn that these probability needs to be driven to zero
    the logit we're producing has to drive the output of the dimension to negative inf
    
    that's no different from all the other tokens that are not in the dataset.
    we're just introducing a few more token that need to be driven to zero in probability
    
functionally not breaking anything. 
taking some extra memory. more flops needed. but it's harmless operation. we're adding calcuation 
bue we're running faster.

In cuda, kernels use block tiles. block tiles are usually nice numbers. power of 2.
calculation are done in chunk of 32, 64.
when the desired calculation doesn't neatly fit into those block tiles.

There are kinds of boundary kernels that can kick into do the last part.
In a lot of kernels, they will chunk it up your input and they will do the nice part(number) first
There are second phase that they come back and process the remaining part.
The kernels for that could be very inefficient. spinning up all these extra computes.

pytorch 2.3.1 or earlier will see 30% improvement

"""
# ------------------------------------------------------------------------------------------------
# ----------------------------------- Sec 3 Algorithmic optimization -----------------------------
# ------------------------------------------------------------------------------------------------
# -----------------------------------  -----------------------------------
""" 
gpt3 has a wider context window (2048), bigger model, some changes of hyperparams

AdamW 
    beta=(0.9, 0.95) (beta2 default is 0.99)
    eps=1e-8 (10^-8) same to default


norm = torch.nn.utils.clip_grad_norm_(modle.parameters(), 1.0)
gradient clip
    clip the global norm of gradient at 1.0 after the calculation the gradient (loss.backward())
    
    norm (length): calculating the global norm of the gradients on parameters.
    square and add it all up (sum) all single gradient on all the parameters. Take a big square root of that
    Make sure that all the length are smaller than 1.0. If more, clip it
    
    In some data batch, the loss might be hight. Therefore the gradient is high as well. This could shock the model
    and shock the optimization. Use gradient clip to prevent the model from too big of shocks in terms of gradient
    magnitude. It upper bounds it in this way.
    
    hacky solution. a patch solution of a deeper issue.
    
    good to always visualize the norm. (print out the number)
    dropping: well behaved
    climbing: bad. destablelizing during training
    spike: some issue of instability.
    # Q! how to address the instability.
    # Q! what are the deep issue
    
    it's not uncommon that norm will be high for the very first few stages
    the model is random. a ton of learning happens in the early stage of training
    mostly learning the biases of the output tokens
    unstable time but it usually stabilized in a very few iterations
    
    for Andrej's norm: 28 -> 6 -> 2 -> 11 -> 6. not insane but a little bit funky
"""


# ----------------------------------- learning rate scheduling -----------------------------------
""" learning rate schedule
cosine decay
300B tokens 
warmup 3.75M tokens
cosine decay until 260B
drop to 10% for some over some horizons. (260-300B)




learning rate scheduler is still an active research area
# Q! what's warmup, 
# Q! why cosine
# Q! why cosine decay multiply decay_ratio with pi
"""
# ----------------------------------- Increase the batch size -----------------------------------
""" 
Andrej skips it for
    - complicate the things because change the number of tokens that are processing at each step of 
        optimization 
    - Complicate the number of tokens that feeds to the model.
    - Not a major improvement. A more system optimize for speed up. not algorithmic. 
    
    early stage
    model is in atypical settings
    most learn to ignore tokens that don't come up in the training set very often.
    Learning very simple biases
    every single token is basically telling use these token not those missing tokens.
    The gradient of every single exmple are extremely highly correlated.
    they all look roughly the same in the original optimization
    
    if the gradient are similar and are highly correlated. no need to do gradients in big 
    batch size.
    
    gradient from batch size of 32k is extremely similar (or exact same gradient) to the 
    batch of 1M early on in the training
    
    Later in the optimization
    One the model learned the simple stuff. that's where the gradients become more decorrelated
    per example. that's where they actually offer you sort of statistical power in some sense.
"""
# ----------------------------------- Data are sampled without replacement during training -----------------------------------
""" until an epoch boundary is reached to minimize overfitting 
exhausting a pool instead of taking and returning to the pool. until the next epoch """
# ----------------------------------- Weight decay of 0.1 -----------------------------------
"""
weight decay: tensor involved in matrix multiplication, embeddings
usually not weight decay: 1-d tensors, laynorm, scale, biases

regularization
pulling down the the weights forces the optimization to 
    1. use more of the weights.
    2. Not allowing any one of the weights individually to be way too large
force the network to kind of distribute the work across more channels because there's sort of 
like a pull of gravity on the weights themselves

# Q! is the attention gets regularized? (weight decay)
"""
""" fused AdamW
instead of iterating in a for loop over all the parameter tensors and updating them.
that will launch a lot of kernels.
fused means all these kernels are fused into a single kernel.
    get rid of a lot of overhead
    a single time on all the parameters call a kernel that updates them
    kernel fusion for adamW update, instead of iterating over all tensors
# Q! fused adam documentation
# Q! Is there any guideline for the tuning optimization parameters?
# Q! Andrej: The relationship between weight_decay, learning_rate, batch_size, the adam parameter 
        beta1, beta2, the episilon. these are complicated mathmatical relationship in the optimization
        literature.
        This could be a whole video
"""

# ----------------------------------- Gradient Accumulation -----------------------------------
""" 
gpt3 paper table 2.1
smaller model: bigger learning rate. vice versa
large model: bigger model, bigger batch size, vice versa

the batch is not directly B in the code.
the "batch" on the paper is amount of tokens.
converting to the "batch" in our code will be 0.5e6 / 1024 ~= 488 batch size

batch size is correlated with all the other optimization parameters and learning rate etc


The loss (cross entropy loss) is 

# Q! cross entropy loss reduction = mean?
the cross entropy is supposed to be the log loss? where is the mean and sum coming from 
https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss


# Trick!
the default: torch.function.cross_entropy's reduction='mean"
So if simply loss.backward for each micro_step, we're scaling down with a wrong factor.
We can correct it by applying the missing factor


# Q! loss.detach()?
detach the tensor from the grad. keep tracking values. (making these leaf node when adding them to loss_accum)

"""
# ----------------------------------- Distributed data parallel -----------------------------------
""" 
DistributedDataParallel (legacy : DataParallel is not recommended)

The forward is still the same. 
for backward, once all gpu complete backward. proceed "all reduce" to average the gradients across all 
the rank (gpus) of their gradient. deposite that average on every single rank. every single rank will 
end up with the average on it. The communication is just synchronizes and averages the gradients.

DDP has some mechanism to improve the efficiency. As the backward pass through the layers of the 
transformer. It actually can dispatch communication for the gradient while the backward pass is 
still happening. there is overlap of the communication of the gradient and the synchronization of them
and the backward pass

forward is same. backward is mostly unchanged. we're tacking on this average as we'll see


DDP with gradient accumulation:
loss.backward(): backward pass and synchronize the gradient
with grad_accum_step loos, we don't actually want to the synchronization after 
every single loss.backward()
Because we're just depositing gradients. and we're doing that serially
We just them to be adding up. we dont' want to synchronize every single time. wasteful
only at the last step that we want to do the all reduce to average the gradient
# Q! is it because the synchronization is time-expensive?


official is with no_sync context manager. Andrej directly toggle with the flag


"""
# ----------------------------------- Dataset used -----------------------------------
""" 
gpt3 used webtext dataset which is never released
open webtext project.
outbound link from reddit, 45M links
Common Crawl: a mess
C4: similar to commoncrawl but processed differently

# Q! Trick
Fineweb edu. how it's processed? hg tutorial page 
sample-10BT


SlimPajama: cleaned RedPajama



"""
# -----------------------------------  -----------------------------------
# -----------------------------------  -----------------------------------
# -----------------------------------  -----------------------------------
# -----------------------------------  -----------------------------------

# ------------------------------------------------------------------------------------------------

