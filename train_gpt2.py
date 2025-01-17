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
        self.rotary = Rotary(dim // num_heads)
        self.c_proj = CastedLinear(dim, dim)
        self.c_proj.weight.data.zero_()

    def forward(self, x, ve, block_mask):
        B, T = x.size(0), x.size(1)
        assert B == 1, 'Must use batch size = 1 for FlexAttention'
        q = self.c_q(x).view(B, T, self.num_heads, -1)
        k = self.c_k(x).view(B, T, self.num_heads, -1)
        v = self.c_v(x).view(B, T, self.num_heads, -1)

        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        else:
            v = self.lambdas[0] * v

        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)

        y = flex_attention(q.transpose(1, 2),
                           k.transpose(1, 2),
                           v.transpose(1, 2),
                           block_mask=block_mask)
        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class FixedBlock(nn.Module):
    """
    Used for the first X blocks (the "input region") and last Y blocks (the "output region").
    May apply value embeddings (ve) and can be integrated into skip connections.
    """
    def __init__(self, model_dim, num_heads, use_attn=True):
        super().__init__()
        self.use_attn = use_attn
        self.attn = CausalSelfAttention(model_dim, num_heads) if use_attn else None
        self.mlp = MLP(model_dim)

        # We keep this for possible skip weighting. If you do not want these
        # extra parameters, you could remove them; but this keeps parity with
        # the old design that had self.lambdas.
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, ve, x0, block_mask):
        # minimal approach, adopting the old code's "lambdas" usage
        x = self.lambdas[0] * x + self.lambdas[1] * x0

        if self.use_attn:
            x = x + self.attn(norm(x), ve, block_mask)
        x = x + self.mlp(norm(x))
        return x


import math
import torch.nn.functional as F
from torch import nn

class Router(nn.Module):
    """
    The Router class governs how each middle block decides which block to visit next, effectively enabling dynamic sequencing in the model’s middle region.
    At a high level, each middle block has 𝑀+1 possible routes: indices 0 to 𝑀-1 for the other middle blocks, plus index 𝑀 for an “exit route” that leads to the output blocks.
    To encourage a natural left-to-right progression, each block is assigned a “forward route,” typically the block with index current_block_idx+1, unless we are at the last middle block, in which case the “forward route” is the exit (𝑀).
    At initialization, the router sets a higher probability on that forward route, so the “default” path traverses the middle blocks in ascending order and finally exits.
    During training, the router’s logit parameters update to reflect learned routes—some blocks may skip forward or route to earlier blocks if that proves more optimal.

    In operation, each middle block receives two special numerical features—block_pass_count (how many times we have visited a middle block so far) and max_block_count (the maximum allowed visits)—which are concatenated to a summary of the block’s token representations.
    This combined input is passed through a Gumbel-Softmax layer to produce a distribution over all possible next routes.
    If training, the code may print out statements like “Routing from 0 to 1” or “Routing from 9 to M (10), exiting to output” or "Max block count reached, routing from 4 to M (10) instead of 5" depending on which path is chosen at each step.
    Early in training, we expect the router to follow its initialization bias, mostly stepping block-by-block through the middle region;
    over time, as the model learns, the router might diverge from this simple route—skipping blocks, revisiting earlier blocks, or exiting earlier—if that yields better performance.
    """
    def __init__(self, model_dim, M, gumbel_temp, initial_forward_prob, this_block_idx):
        super().__init__()
        self.M = M
        self.gumbel_temp = gumbel_temp
        self.this_block_idx = this_block_idx

        # We'll produce logits from a vector in R^(model_dim+2):
        #    x_mean (size model_dim) + [block_pass_count, max_block_count]
        self.weight = nn.Parameter(torch.zeros(M+1, model_dim+2))
        self.bias = nn.Parameter(torch.zeros(M+1))

        # Initialize weight with small random normal distribution
        nn.init.normal_(self.weight, mean=0.0, std=1e-3)

        # The route that is "forward" from this_block_idx is:
        # if i < M-1 => next block = i+1
        # if i == M-1 => exit (index = M)
        self.forward_route_idx = min(self.this_block_idx + 1, M)

        # Bias the router so that self.forward_route_idx has initial_forward_prob
        # and all other routes share the remainder evenly.
        forward_bias = math.log(initial_forward_prob)

        # There are M+1 total routes, so we distribute (1 - initial_forward_prob)
        # among the other M routes. But one of them is also exit route = M.
        # We do not exclude "self route" or anything; it's a design choice whether
        # block i can route to i, or skip multiple blocks, etc.
        # For simplicity, we just treat them all as "other routes" here.
        remain_count = (M+1) - 1  # excluding the forward_route
        if remain_count > 0:
            remain_prob = (1.0 - initial_forward_prob) / remain_count
        else:
            remain_prob = 0.0  # edge case if M=0
        other_bias = math.log(remain_prob) if remain_prob > 0 else -1e8  # so we don't get NaNs

        with torch.no_grad():
            for i in range(M+1):
                if i == self.forward_route_idx:
                    self.bias[i] = forward_bias
                else:
                    self.bias[i] = other_bias

    def forward(self, x, block_pass_count, max_block_count):
        """
        x: shape (B=1, T, D)
        We will take the mean across T to get x_mean in (B=1, D).
        Then we concat [block_pass_count, max_block_count] as two extra features.
        """
        # 1) mean over the time dimension
        x_mean = x.mean(dim=1)  # shape = (1, D)

        # 2) create a (1, 2) tensor for [block_pass_count, max_block_count]
        bc_info = torch.tensor(
            [float(block_pass_count), float(max_block_count)],
            dtype=x_mean.dtype,
            device=x_mean.device
        ).unsqueeze(0)  # shape = (1, 2)

        # 3) concat => shape (1, D+2)
        x_cat = torch.cat([x_mean, bc_info], dim=-1)

        # 4) compute logits => shape (1, M+1)
        logits = F.linear(x_cat, self.weight, self.bias)

        # Gumbel-Softmax distribution. We'll keep it soft but pick argmax for the next route.
        # This allows the gradient to flow, and leaves us open to explore MCTS or other path search.
        route_prob = F.gumbel_softmax(logits, tau=self.gumbel_temp, hard=False)
        return route_prob


class MiddleBlock(nn.Module):
    def __init__(self, model_dim, num_heads, M, gumbel_temp, initial_forward_prob, this_block_idx):
        super().__init__()
        self.this_block_idx = this_block_idx
        self.attn = CausalSelfAttention(model_dim, num_heads)
        self.mlp = MLP(model_dim)

        # Pass the block index into the Router so it knows which route
        # is the "forward" route for this specific block.
        self.router = Router(
            model_dim=model_dim,
            M=M,
            gumbel_temp=gumbel_temp,
            initial_forward_prob=initial_forward_prob,
            this_block_idx=this_block_idx
        )

    def forward(self, x, block_mask, block_pass_count, max_block_count):
        # Standard middle-block processing (no VE usage):
        x = x + self.attn(norm(x), None, block_mask)
        x = x + self.mlp(norm(x))

        # Now get the routing distribution:
        route_prob = self.router(x, block_pass_count, max_block_count)
        return x, route_prob

class ValueEmbedding(nn.Module):
    """
    Constructor receives X and M.  We create X embedding modules, and in forward()
    produce a list of length X+M+X: [ X embeddings, M None, X embeddings ].
    """
    def __init__(self, vocab_size: int, model_dim: int, X: int, M: int):
        super().__init__()
        self.X = X
        self.M = M
        self.embed_list = nn.ModuleList([
            nn.Embedding(vocab_size, model_dim) for _ in range(X)
        ])

    def forward(self, inputs):
        # Generate one embedding output per input-block embedding
        emb_outs = [emb(inputs).bfloat16() for emb in self.embed_list]  # length X
        # Insert M None entries in the middle, then re-append the same X embeddings
        ve = emb_outs + [None] * self.M + emb_outs
        return ve


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        X=1,
        M=10,
        Y=1,
        num_heads=8,
        model_dim=768,
        gumbel_temp=1.0,
        alpha=0.5,
        initial_forward_prob=0.95,
    ):
        """
        GPT with:
          - X fixed input blocks
          - M middle blocks (unordered)
          - Y fixed output blocks
          - We require X == Y
        """
        super().__init__()
        assert X == Y, "We require X == Y for matching input/output block pattern"
        self.X = X
        self.M = M
        self.Y = Y
        self.gumbel_temp = gumbel_temp
        self.alpha = alpha
        self.initial_forward_prob = initial_forward_prob

        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = ValueEmbedding(vocab_size, model_dim, X=X, M=M)

        # Build the input blocks
        self.input_blocks = nn.ModuleList([
            FixedBlock(model_dim, num_heads, use_attn=True)
            for _ in range(X)
        ])

        # Build the M middle blocks, each with a router
        self.middle_blocks = nn.ModuleList([
            MiddleBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                M=M,
                gumbel_temp=self.gumbel_temp,
                initial_forward_prob=self.initial_forward_prob,
                this_block_idx=i
            )
            for i in range(M)
        ])

        # Build the output blocks
        self.output_blocks = nn.ModuleList([
            FixedBlock(model_dim, num_heads, use_attn=True)
            for _ in range(Y)
        ])

        # Final language modeling head
        self.lm_head = CastedLinear(model_dim, vocab_size)
        self.lm_head.weight.data.zero_()

        # For a U-Net-like skip from X => Y blocks:
        # We'll store skip outputs after each input block, then re-add them in output blocks.
        # We'll give each output block a skip weight.
        self.skip_weights = nn.Parameter(torch.ones(Y))

    def forward(self, inputs, targets, sliding_window_num_blocks, max_block_count=10):
        """
        inputs: shape (T,) of token IDs (must be 1D).
        targets: shape (T,) of token IDs (same length).
        sliding_window_num_blocks: for block_mask creation
        max_block_count: the maximum number of times to loop in the middle region
        """
        BLOCK_SIZE = 128
        seq_len = len(inputs)
        assert seq_len % BLOCK_SIZE == 0
        assert inputs.ndim == 1

        total_num_blocks = seq_len // BLOCK_SIZE
        docs = (inputs == 50256).cumsum(0)  # same doc-splitting logic

        # We create the doc_swc block mask as in the original
        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_mask):
            num_blocks = dense_mask.sum(dim=-1, dtype=torch.int32)
            indices = dense_mask.argsort(dim=-1, descending=True, stable=True).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()

        def create_doc_swc_block_mask(sliding_window_num_blocks):
            kv_idx = block_idx = torch.arange(total_num_blocks, dtype=torch.int32, device='cuda')
            q_idx = block_idx[:, None]
            causal_bm = q_idx >= kv_idx
            causal_full_bm = q_idx > kv_idx
            window_bm = (q_idx - kv_idx) < sliding_window_num_blocks
            window_full_bm = window_bm
            document_bm = (docs_low[:, None] <= docs_high) & (docs_low <= docs_high[:, None])
            document_full_bm = (docs_low[:, None] == docs_high) & (docs_low == docs_high[:, None])

            nonzero_bm = causal_bm & window_bm & document_bm
            full_bm = causal_full_bm & window_full_bm & document_full_bm

            kv_num_blocks, kv_indices = dense_to_ordered(nonzero_bm & ~full_bm)
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

        # embed input tokens
        x0 = norm(self.embed(inputs[None]).bfloat16())  # shape (1, T, D)
        x = x0

        # get the value embeddings (which is a list of 12 segments, but we only need X + Y)
        ve_all = self.value_embeds(inputs)  # length 12
        # We'll slice out the first X for the input blocks and the last Y for the output blocks
        # The middle blocks do not use any ve.
        ve_input = ve_all[: self.X]
        ve_output = ve_all[-self.Y :] if self.Y > 0 else []

        # 1) input (X) pass
        skip_connections = []
        for i in range(self.X):
            x = self.input_blocks[i](x, ve_input[i], x0, block_mask)
            # store skip
            skip_connections.append(x)

        # 2) dynamic middle pass
        current_idx = 0
        block_pass_count = 0

        # We have M middle blocks; the router picks among 0..(M-1) or M => exit
        while block_pass_count < max_block_count:
            # pass x through the current middle block
            x, route_prob = self.middle_blocks[current_idx](x, block_mask, block_pass_count, max_block_count)
            # route_prob: shape (1, M+1)
            next_idx = route_prob.argmax(dim=-1)[0]  # pick the route with highest prob
            if bool(next_idx.eq(self.M)):
                print(f"Routing from {current_idx} to M ({self.M}), exiting to output")
                # means route to output region
                break
            if (block_pass_count+1) >= max_block_count:
                print(f"Max block count reached, routing from {current_idx} to M ({self.M}) instead of {int(next_idx)}")
            else:
                print(f"Routing from {current_idx} to {int(next_idx)}")
            current_idx = int(next_idx)
            block_pass_count += 1


        # 3) output (Y) pass
        # For a U-Net style, we re-add skip from the input blocks
        for j in range(self.Y):
            x = x + self.skip_weights[j] * skip_connections.pop()
            x = self.output_blocks[j](x, ve_output[j], x0, block_mask)

        # finalize
        x = norm(x)
        logits = self.lm_head(x)
        # soft cap
        logits = 15 * torch.tanh(logits / 15)
        logits = logits.float()

        # compute cross-entropy
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets)

        # add usage penalty for the middle region
        if self.training:
            usage_penalty = self.alpha * (block_pass_count / float(self.M if self.M > 0 else 1))
            loss = loss + usage_penalty

        return loss


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
    sequence_length: int = 10 * 1024  # sequence length, in tokens
    num_iterations: int = 1490  # number of iterations to run
    warmup_iters: int = 0
    cooldown_iters: int = 600  # number of iterations of linear warmup/cooldown for triangular or trapezoidal schedule
    weight_decay: float = 0
    # Unordered Blocks
    X: int = 1
    M: int = 10
    Y: int = 1
    gumbel_temp: float = 1.0
    route_count_alpha: float = 0.5
    initial_forward_prob: float = 0.95
    # evaluation and logging hyperparams
    val_loss_every: int = 0  # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens: int = 1638400*7 #1638400*7 #3112960*4 #10485760  # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
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
if args.fixed_seed is not None:
    set_global_seed(args.fixed_seed)


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
model = GPT(
    vocab_size=50304,
    X=args.X,
    M=args.M,
    Y=args.Y,
    num_heads=6,
    model_dim=768,
    gumbel_temp=args.gumbel_temp,
    alpha=args.route_count_alpha,
    initial_forward_prob=args.initial_forward_prob,
)
model = model.cuda().bfloat16()
for m in model.modules():
    if isinstance(m, CastedLinear):
        m.float()
config.coordinate_descent_tuning = True  # suggested by @Chillee
model = torch.compile(model)
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank], broadcast_buffers=False, gradient_as_bucket_view=True, find_unused_parameters=True)
raw_model = model.module  # always contains the "raw" unwrapped model

# init the optimizer(s)
hidden_matrix_params = []
for blk in raw_model.input_blocks:
    hidden_matrix_params.extend(p for p in blk.parameters() if p.ndim == 2)
for blk in raw_model.middle_blocks:
    hidden_matrix_params.extend(p for p in blk.parameters() if p.ndim == 2)
for blk in raw_model.output_blocks:
    hidden_matrix_params.extend(p for p in blk.parameters() if p.ndim == 2)

# embed_params = the main embedding plus value embeds
embed_params = [raw_model.embed.weight, *raw_model.value_embeds.parameters()]
# LM head params
head_params = [raw_model.lm_head.weight]
# scalar_params = everything else that is < 2D
scalar_params = [
    p for p in raw_model.parameters()
    if p.ndim < 2
]
# Define the two-optimizer scheme:
optimizer1 = torch.optim.Adam(
    [
        dict(params=embed_params, lr=0.6),
        dict(params=head_params, lr=0.008),
        dict(params=scalar_params, lr=0.04),
    ],
    betas=(0.8, 0.95),
    fused=True
)
optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95)
optimizers = [optimizer1, optimizer2]


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
        # log val loss to console and to logfile
        print0(
            f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / (timed_steps - 1):.2f}ms')
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

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
            # This doesn't work with token-weighted loss (or needs to be set > args.token_loss_warmup_iters?)
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
        for group in optimizer2.param_groups:
            group['momentum'] = (1 - frac) * 0.85 + frac * 0.95
    #if step == (args.num_iterations - (args.cooldown_iters // 2)):
    #    args.val_loss_every = 15
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

print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()
