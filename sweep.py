import os
import sys

with open(sys.argv[0]) as f:
    code = f.read()  # read the code of this file ASAP, for logging
import uuid
import time
import contextlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.attention.flex_attention import BlockMask, flex_attention  # KoszarskyB
from bayes_opt import BayesianOptimization
from bayes_opt.event import DEFAULT_EVENTS, Events
import json
import math
import random
import numpy as np


# -----------------------------------------------------------------------------
# Muon optimizer

@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Use the Frobenius norm of (X @ X.T)^2 computed during first NS iteration to ensure spectral norm
    # is below 1, as suggested by Johan Sokrates Wind @johanwind
    # https://github.com/KellerJordan/modded-nanogpt/discussions/23#discussioncomment-11293594
    A = X @ X.T
    A2 = A @ A
    A2_norm = A2.norm() + 1e-28
    X /= A2_norm ** 0.25  # ensure top singular value <= 1
    A /= A2_norm ** 0.5
    A2 /= A2_norm
    X = a * X + (b * A + c * A2) @ X

    # Perform the remaining NS iterations
    for _ in range(steps - 1):
        A = X @ X.T
        B = b * A + c * A @ A  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.rank = int(os.environ['RANK'])
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)
        assert all(isinstance(p, torch.Tensor) for p in params)
        sizes = {p.numel() for p in params}
        param_groups = [
            {
                'params': [p for p in params if p.numel() == size],
                'update_buffer': [
                    torch.empty(size, device='cuda', dtype=torch.bfloat16)
                    for _ in range(self.world_size)
                ],
            }
            for size in sizes
        ]
        super().__init__(param_groups, defaults)

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            update_buffers = group['update_buffer']
            # generate weight updates in distributed fashion
            params = group['params']
            assert len(params) % self.world_size == 0
            handle = None
            params_world = None

            def update_prev():
                if params_world is None:
                    return
                assert handle is not None
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffers):
                    p_world.data.add_(
                        g_world.view_as(p_world),
                        alpha=-lr * max(1, p_world.size(0) / p_world.size(1)) ** 0.5,
                    )

            for base_i in range(len(params))[::self.world_size]:
                p = params[base_i + self.rank]
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.lerp_(g, 1 - momentum)
                g = g.lerp_(buf, momentum) if nesterov else buf
                g = zeropower_via_newtonschulz5(g, steps=ns_steps).flatten()
                update_prev()
                handle = dist.all_gather(update_buffers, g, async_op=True)
                params_world = params[base_i: base_i + self.world_size]
            update_prev()


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))

class Rotary(nn.Module):
    def __init__(self, dim, max_seq_len=65536):
        super().__init__()
        inv_freq = (1 / 1024) ** torch.linspace(0.0, 1.0, steps=dim // 4, dtype=torch.float32)
        inv_freq = torch.cat([inv_freq, inv_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i, j -> ij", t, inv_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x: Tensor):
        cos, sin = self.cos[None, :x.size(-3), None, :], self.sin[None, :x.size(-3), None, :]
        x1, x2 = x.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(dim // num_heads) # dim // num_heads = head_dim
        self.c_proj = CastedLinear(dim, dim)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x, vi, block_mask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q = self.c_q(x).view(B, T, self.num_heads, -1)
        k = self.c_k(x).view(B, T, self.num_heads, -1)
        v = self.c_v(x).view(B, T, self.num_heads, -1)
        v = self.lambdas[0] * v + self.lambdas[1] * vi.view_as(v) # @KoszarskyB & @Grad62304977
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, enable_gqa=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.c_fc   = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config: "GPTConfig", layer_idx: int):
        super().__init__()
        self.skip_attn_idx = config.num_layers // 2 + 1
        if layer_idx != self.skip_attn_idx:
            self.attn = CausalSelfAttention(config.model_dim, config.num_heads)
        self.mlp = MLP(config.model_dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))
        self.layer_idx = layer_idx

    def forward(self, x, vi, x0, block_mask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.layer_idx != self.skip_attn_idx:
            x = x + self.attn(norm(x), vi, block_mask)
        x = x + self.mlp(norm(x))
        return x

class ValueEmbedding(nn.Module):
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.embed = nn.ModuleList([
            nn.Embedding(config.vocab_size, config.model_dim)
            for _ in range(6)
        ])

    def forward(self, inputs) -> "list[torch.Tensor]":
        ve = [emb(inputs) for emb in self.embed]
        ve += reversed(ve)
        return ve

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    num_layers : int = 12
    num_heads : int = 6 # head dim 128 suggested by @Grad62304977
    model_dim : int = 768

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.num_layers = config.num_layers

        # U-net design by @brendanh0gan
        self.num_encoder_layers = config.num_layers // 2  # Half of the layers for encoder
        self.num_decoder_layers = config.num_layers - self.num_encoder_layers  # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.embed = nn.Embedding(config.vocab_size, config.model_dim)
        self.blocks = nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.num_layers)])
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual learning
        # U-net structure on token value embeddings by @leloykun
        self.value_embeds = ValueEmbedding(config)
        self.lm_head = CastedLinear(config.model_dim, config.vocab_size)
        self.lm_head.weight.data.zero_()  # @Grad62304977

        # Initialize token loss tracking
        self.register_buffer('token_loss_counts', torch.zeros(config.vocab_size))
        self.register_buffer('token_loss_sums', torch.zeros(config.vocab_size))
        self.register_buffer('token_loss_sum_squares', torch.zeros(config.vocab_size))
        self.register_buffer('token_loss_avgs', torch.ones(config.vocab_size))
        self.register_buffer('token_loss_stds', torch.zeros(config.vocab_size))
        # Buffers for interval-based updates
        self.register_buffer('interim_counts', torch.zeros(config.vocab_size))
        self.register_buffer('interim_sums', torch.zeros(config.vocab_size))
        self.register_buffer('interim_sum_squares', torch.zeros(config.vocab_size))
        # Track recently touched tokens
        self.recent_tokens = [set() for _ in range(args.recent_token_memory)]
        self.current_interval_idx = 0

    @torch._dynamo.disable()
    @torch.no_grad()
    def _gpu_accumulate_stats(self, losses, targets, step):
        # Vectorized approach, all on GPU
        flat_targets = targets.view(-1).long()
        flat_losses = losses.view(-1).bfloat16()

        unique_tokens, inverse_indices = flat_targets.unique(return_inverse=True)

        counts = torch.zeros_like(unique_tokens, dtype=torch.bfloat16)
        sums = torch.zeros_like(unique_tokens, dtype=torch.bfloat16)
        sum_squares = torch.zeros_like(unique_tokens, dtype=torch.bfloat16)

        ones_for_counts = torch.ones_like(flat_losses, dtype=torch.bfloat16)
        counts.index_add_(0, inverse_indices, ones_for_counts)
        sums.index_add_(0, inverse_indices, flat_losses)
        sum_squares.index_add_(0, inverse_indices, flat_losses ** 2)

        self.interim_counts.index_add_(0, unique_tokens, counts)
        self.interim_sums.index_add_(0, unique_tokens, sums)
        self.interim_sum_squares.index_add_(0, unique_tokens, sum_squares)

        # --- Store these tokens into the current interval set ---
        # Move to CPU so we can stuff them into a Python set
        unique_tokens_cpu = unique_tokens.detach().cpu().tolist()
        self.recent_tokens[self.current_interval_idx].update(unique_tokens_cpu)

        # Every N steps, call _update_global_stats
        if step % args.stats_update_interval == (args.stats_update_interval - 1):
            self._update_global_stats()
            # After stats update, advance to the NEXT buffer slot (ring buffer)
            self.current_interval_idx = (self.current_interval_idx + 1) % args.recent_token_memory
            # Clear that next slot, because we’re about to start writing into it
            self.recent_tokens[self.current_interval_idx].clear()

    @torch._dynamo.disable()
    @torch.no_grad()
    def _update_global_stats(self):
        all_recent_tokens = set()
        for rt in self.recent_tokens:
            if rt is not None:
                all_recent_tokens.update(rt)

        if not all_recent_tokens:
            return  # nothing to update

        update_mask = torch.zeros_like(self.token_loss_counts, dtype=torch.bool)
        update_mask[list(all_recent_tokens)] = True

        # Update global counts
        self.token_loss_counts[update_mask] += self.interim_counts[update_mask]
        self.token_loss_sums[update_mask] += self.interim_sums[update_mask]
        self.token_loss_sum_squares[update_mask] += self.interim_sum_squares[update_mask]

        # Compute means and std
        counts = self.token_loss_counts[update_mask]
        means = self.token_loss_sums[update_mask] / counts.clamp(min=1)

        variances = (self.token_loss_sum_squares[update_mask] / counts.clamp(min=1)) - (means ** 2)
        stds = torch.sqrt(torch.clamp(variances, min=1e-6))

        self.token_loss_avgs[update_mask] = means
        self.token_loss_stds[update_mask] = stds

        # Clear interim buffers
        self.interim_counts.zero_()
        self.interim_sums.zero_()
        self.interim_sum_squares.zero_()

    @torch._dynamo.disable()
    @torch.no_grad()
    def get_token_weights(self, targets, step):
        """
        Avoid any branch that changes shapes. Inductor sees a single path:
          - If valid_mask has no True, we define global_mean=0, global_std=1 => weights=1
          - If step < warmup => we set progress=0 => weights=1
          - Otherwise => normal weighting
        Everything is float32. Return shape == [# of tokens in the batch].
        """
        # 0) Flatten & ensure int64 for indexing, float32 for final
        flat_targets = targets.view(-1).to(torch.int64)
        device = flat_targets.device

        # We'll build final_weights in float32
        final_weights = torch.ones_like(flat_targets, dtype=torch.float32, device=device)

        # 1) Compute progress factor in [0,1]
        warmup = args.token_loss_warmup_iters
        total_steps = args.num_iterations - warmup
        if total_steps <= 0:
            progress = 0.0
        else:
            step_after_warmup = max(0, step - warmup)
            progress = float(min(max(step_after_warmup / total_steps, 0.0), 1.0))

        # 2) Compute max/min weight in float32
        max_weight = 1.0 + (args.token_loss_max_weight - 1.0) * progress
        min_weight = 1.0 + (args.token_loss_min_weight - 1.0) * progress
        max_weight_t = torch.tensor(max_weight, dtype=torch.float32, device=device)
        min_weight_t = torch.tensor(min_weight, dtype=torch.float32, device=device)

        # 3) Gather stats from global buffers
        counts = self.token_loss_counts[flat_targets].float()  # shape [batch]
        means = self.token_loss_avgs[flat_targets].float()  # shape [batch]

        valid_counts = (self.token_loss_counts > 0).to(torch.int8)
        has_valid = valid_counts.sum() > 0

        global_mean = torch.where(
            has_valid,
            (self.token_loss_avgs * valid_counts.float()).sum() / (valid_counts.float().sum() + 1e-8),
            torch.tensor(0.0, dtype=torch.float32, device=device)
        )
        global_std = torch.where(
            has_valid,
            (self.token_loss_stds * valid_counts.float()).sum() / (valid_counts.float().sum() + 1e-8),
            torch.tensor(1.0, dtype=torch.float32, device=device)
        )

        # 4) Compute z-scores
        z_scores = (means - global_mean) / (global_std + 1e-6)

        # Base weighting from z-scores => in [-1, +1], scaled to [1, max_weight]
        tanh_vals = torch.tanh(z_scores)  # in [-1, +1]
        diff = max_weight_t - 1.0
        raw_weights = 1.0 + tanh_vals * diff  # shape [batch]

        # --- Apply new flags to clamp or disable certain directions ---
        # if we're NOT up-weighting high-loss tokens => clamp any raw_weights > 1.0 down to 1.0
        if not args.enable_high_loss_upweighting:
            raw_weights = torch.minimum(raw_weights, torch.tensor(1.0, device=device))

        # if we're NOT down-weighting low-loss tokens => clamp any raw_weights < 1.0 up to 1.0
        if not args.enable_low_loss_downweighting:
            raw_weights = torch.maximum(raw_weights, torch.tensor(1.0, device=device))

        # if we *are* down-weighting low-loss tokens => enforce a minimum of token_loss_min_weight
        if args.enable_low_loss_downweighting:
            # Typically min_weight <= 1.0.  If user sets it above 1.0, it’s on them.
            raw_weights = torch.clamp(raw_weights, min=min_weight_t.item(), max=9999.0)

        # 5) Rare-token logic
        rare_token_mask = (counts < args.min_token_samples)
        sample_factor = (counts / args.min_token_samples).clamp_(0, 1)
        rare_token_weights = 1.0 + args.rare_token_epsilon * sample_factor
        final_weights.copy_(raw_weights)  # fill with raw_weights
        final_weights[rare_token_mask] = rare_token_weights[rare_token_mask]

        # 6) If progress=0 => we get raw_weights=1 => done

        # 7) Normalize so mean=1 (only if non-empty)
        target_mean = 1.0
        if final_weights.numel() > 0:
            mean_val = final_weights.mean()
            scale_factor = target_mean / (mean_val + 1e-8)
            # Don’t over-correct
            scale_factor = torch.clamp(scale_factor, 0.5, 2.0)
            final_weights *= scale_factor

        return final_weights

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        sliding_window_num_blocks: torch.Tensor,
        step: int,
        mode: str = 'train'
    ):
        BLOCK_SIZE = 128
        seq_len = len(inputs)
        assert seq_len % BLOCK_SIZE == 0
        total_num_blocks = seq_len // BLOCK_SIZE
        assert inputs.ndim == 1
        docs = (inputs == 50256).cumsum(0)
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_mask: torch.Tensor):
            num_blocks = dense_mask.sum(dim=-1, dtype=torch.int32)
            indices = dense_mask.argsort(dim=-1, descending=True, stable=True).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        def create_doc_swc_block_mask(sliding_window_num_blocks: torch.Tensor):
            kv_idx = block_idx = torch.arange(total_num_blocks, dtype=torch.int32, device="cuda")
            q_idx = block_idx[:, None]
            causal_bm = q_idx >= kv_idx
            causal_full_bm = q_idx > kv_idx
            window_bm = q_idx - kv_idx < sliding_window_num_blocks
            window_full_bm = window_bm
            # document_bm = (docs_low[q_idx] <= docs_high[kv_idx]) & (docs_low[kv_idx] <= docs_high[q_idx])
            document_bm = (docs_low[:, None] <= docs_high) & (docs_low <= docs_high[:, None])
            document_full_bm = (docs_low[:, None] == docs_high) & (docs_low == docs_high[:, None])
            nonzero_bm = causal_bm & window_bm & document_bm
            full_bm  = causal_full_bm & window_full_bm & document_full_bm
            kv_num_blocks, kv_indices = dense_to_ordered(nonzero_bm ^ full_bm)
            full_kv_num_blocks, full_kv_indices = dense_to_ordered(full_bm)
            return BlockMask.from_kv_blocks(
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )

        block_mask = create_doc_swc_block_mask(sliding_window_num_blocks)

        # forward the GPT model itself
        x = self.embed(inputs[None]) # token embeddings of shape (b, t, model_dim)
        x = norm(x) # @Grad62304977
        x0 = x
        ve = self.value_embeds(inputs)
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]

        # Store outputs for U-Net skip connections
        skip_connections = []
        # Encoder pass - process only the first half of the blocks
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, ve_enc[i], x0, block_mask)
            skip_connections.append(x)
        # Decoder pass - process the remaining blocks with weighted skip connections
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            # U-net structure on token value embeddings by @leloykun
            x = self.blocks[self.num_encoder_layers + i](x, ve_dec[i], x0, block_mask)

        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30) # @Grad62304977
        logits = logits.float()

        if mode == 'val':
            # Return plain cross-entropy
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        # Calculate initial losses per token
        losses = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
        # GPU-based stats accumulation
        self._gpu_accumulate_stats(losses.detach(), targets.detach(), step)
        # Weighting
        token_weights = self.get_token_weights(targets.detach(), step)
        weighted_losses = losses * token_weights
        return weighted_losses.mean()

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(file: Path):
    # only reads the header, returns header data
    # header is 256 int32
    header = torch.from_file(f"{file}", False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    return int(header[2]) # number of tokens (claimed)

def _load_data_shard(path: Path, num_tokens):
    with path.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, seq_len, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.seq_len = seq_len

        # glob files that match the pattern
        self.files = sorted(Path.cwd().glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        self.files_num_tokens = [_peek_data_shard(file) for file in self.files]
        assert min(self.files_num_tokens) >= num_processes * seq_len + 1
        self.total_num_tokens = sum(self.files_num_tokens)

        self.reset()

    def reset(self):
        self.current_shard = -1
        self.advance()

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.seq_len
        self.tokens = _load_data_shard(self.files[self.current_shard], self.files_num_tokens[self.current_shard])

    def next_batch(self):
        batch_size = self.seq_len * self.num_processes
        buf = self.tokens[self.current_position:self.current_position+self.seq_len+1]
        # host side async is sufficient;
        # no performance improvement was observed when introducing a separate stream.
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # inputs
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # targets
        # advance current position and load next shard if necessary
        self.current_position += batch_size
        if self.current_position + batch_size + 1 >= len(self.tokens):
            self.advance()
        return inputs, targets


# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin: str = 'data/fineweb10B/fineweb_train_*.bin'  # input .bin to train on
    input_val_bin: str = 'data/fineweb10B/fineweb_val_*.bin'  # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size: int = 8  # batch size, in sequences, across all devices
    sequence_length: int = 19 * 1024  # sequence length, in tokens
    num_iterations: int = 1490  # number of iterations to run
    warmup_iters: int = 100
    cooldown_iters: int = 600  # number of iterations of linear warmup/cooldown for triangular or trapezoidal schedule
    weight_decay: float = 0
    # Token loss weighting hyperparameters
    token_loss_warmup_iters: int = 201  # Steps before starting to apply token-specific weighting
    token_loss_max_weight: float = 1.014  # Maximum multiplier for high-loss tokens
    token_loss_min_weight: float = 0.8726  # Minimum multiplier for low-loss tokens
    min_token_samples: int = 10  # Minimum samples needed for full weighting
    rare_token_epsilon: float = 0.1  # Max extra weight for rare tokens
    stats_update_interval: int = 19  # How often to update token statistics
    recent_token_memory: int = 9  # How many update intervals to keep track of touched tokens
    enable_high_loss_upweighting: bool = True  # If True, tokens with mean > global_mean are up-weighted
    enable_low_loss_downweighting: bool = False  # If True, tokens with mean < global_mean are down-weighted
    # evaluation and logging hyperparams
    val_loss_every: int = 0  # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens: int = 3112960*40 #1638400*7 #3112960*4 #10485760  # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every: int = 0  # every how many steps to save the checkpoint? 0 for only at the end
    fixed_seed: int = 1337  # None to disable


args = Hyperparameters()

# set up DDP (distributed data parallel). torchrun sets this env variable
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
assert torch.cuda.is_available()
device = torch.device(f'cuda:{ddp_local_rank}')
torch.cuda.set_device(device)
print(f'using device: {device}')
dist.init_process_group(backend='nccl', device_id=device)
dist.barrier()
master_process = (ddp_rank == 0)  # this process will do logging, checkpointing etc.


def set_global_seed(seed: int):
    # Have rank 0 create a fresh random seed
    if ddp_rank == 0:
        # rank 0 is the "root" - broadcast to everyone
        seeder = torch.tensor([seed], device='cuda', dtype=torch.long)
    else:
        seeder = torch.zeros((1,), device='cuda', dtype=torch.long)
    # Make sure everyone has reached this point
    dist.barrier()
    # Broadcast that seed to all ranks
    dist.broadcast(seeder, src=0)

    # Python and NumPy
    random.seed(seed + ddp_rank)
    np.random.seed(seed + ddp_rank)

    # PyTorch
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed + ddp_rank)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + ddp_rank)
        torch.cuda.manual_seed_all(seed + ddp_rank)
        # Make cuDNN determinstic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Another barrier so that all processes finish seeding before continuing
    dist.barrier()


# begin logging
logfile = None
if master_process:
    run_id = uuid.uuid4()
    Path('logs').mkdir(exist_ok=True)
    logdir = Path('logs') / f'{run_id}'
    logdir.mkdir()
    logfile = Path('logs') / f'{run_id}.txt'
    print(logfile.stem)
    # create the log file
    with logfile.open('w') as f:
        # begin the log by printing this file (the Python code)
        print(code, file=f)
        print('=' * 100, file=f)


def print0(s, logonly=False):
    if master_process:
        with logfile.open('a') as f:
            if not logonly:
                print(s)
            print(s, file=f)


# log information about the hardware/software environment this is running on
# and print the full `nvidia-smi` to file
print0(f'Running python {sys.version}')
#print0(f'Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:')
#import subprocess

#result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#print0(f'{result.stdout}', logonly=True)
#print0('=' * 100, logonly=True)

# calculate the number of steps to take in the val loop.
assert args.val_tokens % (args.sequence_length * ddp_world_size) == 0
val_steps = args.val_tokens // (args.sequence_length * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (ddp_world_size) == 0
train_accumulation_steps = args.batch_size // ddp_world_size


# learning rate decay scheduler (linear warmup and cooldown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it + 1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.cooldown_iters:
        return 1.0
    # 3) linear cooldown
    else:
        decay_ratio = (args.num_iterations - it) / args.cooldown_iters
        return decay_ratio


best_score = -float("inf")
target_loss = 3.925
val_loss_every = 0 #args.val_loss_every
dry_run = True
FIXED_SEED = 1337
def objective(token_loss_max_weight, stats_update_interval, recent_token_memory):
    global best_score, val_loss_every, dry_run
    args.token_loss_max_weight = round(token_loss_max_weight, 3)
    args.stats_update_interval = int(round(stats_update_interval / 100.0))
    args.recent_token_memory = int(round(recent_token_memory / 100.0))

    set_global_seed(FIXED_SEED)

    args.val_loss_every = val_loss_every
    best_val_loss = float('inf')
    best_step = 0
    if dry_run:
        print("\n\n!!!!!!\nDRY RUN\n!!!!!!\n\n")
    else:
        print(f"\n\n!!!!!!\nTESTING:{args.token_loss_max_weight}, {args.stats_update_interval}, {args.recent_token_memory}\n!!!!!!\n\n")

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, args.sequence_length, ddp_rank, ddp_world_size)
    val_loader = DistributedDataLoader(args.input_val_bin, args.sequence_length, ddp_rank, ddp_world_size)
    print0(
        f"Training DataLoader: total number of tokens: {train_loader.total_num_tokens} across {len(train_loader.files)} files")
    print0(
        f"Validation DataLoader: total number of tokens: {val_loader.total_num_tokens} across {len(val_loader.files)} files")
    print0('=' * 100, logonly=True)
    inputs_train, targets_train = train_loader.next_batch()

    # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
    # this originates from Karpathy's experiments.
    num_vocab = 50304
    model = GPT(GPTConfig(vocab_size=num_vocab, num_layers=12, num_heads=6, model_dim=768))
    model = model.cuda().bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    config.coordinate_descent_tuning = True  # suggested by @Chillee
    model = torch.compile(model)
    # here we wrap model into DDP container
    model = DDP(model, device_ids=[ddp_local_rank], broadcast_buffers=False, gradient_as_bucket_view=True)
    raw_model = model.module  # always contains the "raw" unwrapped model

    # init the optimizer(s)
    embed_params = [*raw_model.embed.parameters(), *raw_model.value_embeds.parameters()]
    optimizer1 = torch.optim.Adam(embed_params, lr=0.6, betas=(0.8, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight], lr=0.008, betas=(0.8, 0.95), fused=True)
    params = list(raw_model.blocks.parameters())
    matrix_params = [p for p in params if p.ndim == 2]
    scalar_params = [p for p in params if p.ndim < 2] + [raw_model.skip_weights]
    optimizer3 = torch.optim.AdamW(scalar_params, lr=0.04, betas=(0.8, 0.95), fused=True, weight_decay=0.025)
    optimizer4 = Muon(matrix_params, lr=0.05, momentum=0.95)
    optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]

    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

    sliding_window_num_blocks = torch.tensor(1, dtype=torch.int32, device="cuda")
    sw_num_blocks_prev = 1
    # Start training loop
    training_time_ms = 0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # begin training
    for step in range(args.num_iterations + 1):
        last_step = (step == args.num_iterations)
        # This effectively ignores timing first 10 steps, which are slower for weird reasons.
        # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
        # steps with dummy data first, and then re-initialize the model and reset the loader.
        if step == 10:
            training_time_ms = 0
            t0 = time.perf_counter()
        timed_steps = float('nan') if step <= 11 else (step - 10) + 1  # <= 11 to avoid bug in val

        # Linearly increase the sliding window size over training in chunks of 64 from 64 -> 1792. By @fernbear.bsky.social
        frac_done = step / args.num_iterations  # training progress
        sw_num_blocks = int(((1 - frac_done) * 64 + frac_done * 1792 + 64) // 128)
        if sw_num_blocks != sw_num_blocks_prev:
            sliding_window_num_blocks.copy_(sw_num_blocks, non_blocking=True)
            sw_num_blocks_prev = sw_num_blocks

        if dry_run and step == 350:
            dry_run = False
            print("Quitting dry run.")
            return -999999

        # once in a while evaluate the validation dataset
        if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0) or step in []):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)

            # run validation batches
            model.eval()
            val_loader.reset()
            val_loss = 0.0
            for _ in range(val_steps):
                with torch.no_grad():
                    inputs_val, targets_val = val_loader.next_batch()
                    val_loss += model(inputs_val, targets_val, sliding_window_num_blocks, step, mode="val")
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss /= val_steps
            #if val_loss < best_val_loss:
            best_val_loss = val_loss.item()
            best_step = step
            # log val loss to console and to logfile
            print0(
                f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / (timed_steps - 1):.2f}ms')
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            # if val_loss < target_loss and step >= 1200:
            #    break
            if math.isnan(val_loss):
                return -999999

        if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            # save the state of the training process
            #log = dict(step=step, code=code, model=raw_model.state_dict(),
            #           optimizers=[opt.state_dict() for opt in optimizers])
            #torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        # bit confusing: we want to make sure to eval on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        for i in range(1, train_accumulation_steps + 1):
            with contextlib.ExitStack() as stack:
                if i < train_accumulation_steps:  # there's no need to sync gradients every accumulation step
                    stack.enter_context(model.no_sync())
                # Layer step-up causes recompilations, so we have to disable this optimization
                #if step >= 5:
                #    stack.enter_context(torch.compiler.set_stance(skip_guard_eval_unsafe=True))
                model(inputs_train, targets_train, sliding_window_num_blocks, step).backward()
                inputs_train, targets_train = train_loader.next_batch()
        if train_accumulation_steps != 1:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= train_accumulation_steps
        if step < 301:
            # momentum warmup for Muon
            frac = min(step / 300, 1)
            for group in optimizer4.param_groups:
                group['momentum'] = (1 - frac) * 0.85 + frac * 0.95
        #if step == 1250:
        #    args.val_loss_every = 8
        # step the optimizers and schedulers
        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()
        # null the gradients
        model.zero_grad(set_to_none=True)
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.
        approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
        print0(
            f"step:{step + 1}/{args.num_iterations} train_time:{approx_time:.0f}ms step_avg:{approx_time / timed_steps:.2f}ms")

    #score = -training_time_ms - (best_step * 3.0) - (best_val_loss * 400.0) if best_val_loss <= target_loss else -999999
    score = -best_val_loss
    print(f"\n\n!!!!!!\nSCORE:{score}\n!!!!!!\n\n")
    if score > best_score:
        best_score = score
    update_best_params(args.token_loss_max_weight, args.stats_update_interval, args.recent_token_memory, best_score == score, best_step, best_val_loss, training_time_ms, score)
    return score


def update_best_params(token_loss_max_weight, stats_update_interval, recent_token_memory, is_best, step, val_loss, training_time_ms, score):
    with open('best_params.json', 'a') as f:
        json.dump({
            'token_loss_max_weight': token_loss_max_weight,
            'stats_update_interval': stats_update_interval,
            'recent_token_memory': recent_token_memory,
            'is_best': is_best,
            'step': step,
            'val_loss': val_loss,
            'training_time_ms': training_time_ms,
            'score': score,
        }, f)
        f.write('\n')

# Bayesian Optimization
pbounds = {
    'token_loss_max_weight': (1.018, 1.025), #(1.0, 1.1),
    # These are *100 (they're divided before rounding)
    'stats_update_interval': (19_00, 19_00.1), #(1_00, 50_00)
    'recent_token_memory': (9_00, 9_00.1), #(1_00, 10_00),
}

optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42,
)
# Dry run for timing (big ouch but necessary)
objective(args.token_loss_max_weight, args.stats_update_interval * 100, args.recent_token_memory * 100)

best_score = -float("inf")
optimizer.maximize(
    init_points=7,
    n_iter=35,
)

print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()
