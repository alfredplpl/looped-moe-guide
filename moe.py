"""Decoder-only MoE with optional depth recurrence over its middle blocks."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class MoEConfig:
    vocab_size: int = 16_384
    context_length: int = 512
    layers: int = 12
    width: int = 512
    heads: int = 8
    experts: int = 18
    experts_per_token: int = 2
    expert_width: int = 2_944
    balance_weight: float = 0.01
    loop_repeats: int = 1

    def __post_init__(self):
        if min(self.vocab_size, self.context_length, self.layers, self.width,
               self.heads, self.expert_width) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("Model width must be divisible by the number of heads")
        if not 1 <= self.experts_per_token <= self.experts:
            raise ValueError("Experts per token must be between 1 and the expert count")
        if self.loop_repeats < 1:
            raise ValueError("The middle blocks must run at least once")
        if self.loop_repeats > 1 and self.layers < 5:
            raise ValueError("Recurrence requires two entry blocks, a middle block and two exit blocks")

    @property
    def block_order(self) -> tuple[int, ...]:
        if self.loop_repeats == 1:
            return tuple(range(self.layers))
        entry = tuple(range(2))
        middle = tuple(range(2, self.layers - 2))
        exit_blocks = tuple(range(self.layers - 2, self.layers))
        return entry + middle * self.loop_repeats + exit_blocks

    @property
    def logical_layers(self) -> int:
        return self.layers + (self.loop_repeats - 1) * max(self.layers - 4, 0)


class ExpertLayer(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.count = config.experts
        self.top_k = config.experts_per_token
        self.router = nn.Linear(config.width, self.count, bias=False)
        self.gate_up = nn.Parameter(
            torch.empty(self.count, config.width, 2 * config.expert_width)
        )
        self.down = nn.Parameter(
            torch.empty(self.count, config.expert_width, config.width)
        )
        nn.init.normal_(self.gate_up, std=0.02)
        nn.init.normal_(self.down, std=0.02 / (2 * config.logical_layers) ** 0.5)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = hidden.flatten(0, 1)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            scores = F.linear(tokens.float(), self.router.weight.float())
        top_scores, expert_ids = scores.topk(self.top_k, dim=-1)
        weights = top_scores.softmax(dim=-1)

        # Pack equally sized batches for the GPU. Pad to the busiest expert so
        # that every selected route is processed, even if routing is uneven.
        sorted_experts, order = expert_ids.flatten().sort()
        counts = torch.zeros(self.count, device=tokens.device, dtype=torch.long)
        counts.scatter_add_(0, sorted_experts, torch.ones_like(sorted_experts))
        capacity = int(counts.max().item())
        starts = counts.cumsum(0) - counts
        rank = torch.arange(order.numel(), device=tokens.device) - starts[sorted_experts]
        slots = sorted_experts * capacity + rank
        token_ids = order // self.top_k
        packed = tokens.new_zeros(self.count * capacity, tokens.size(-1))
        packed = packed.index_copy(0, slots, tokens[token_ids])
        gate, up = torch.bmm(packed.view(self.count, capacity, -1), self.gate_up).chunk(2, dim=-1)
        output = torch.bmm(F.silu(gate) * up, self.down).flatten(0, 1)
        routed = output[slots].float() * weights.flatten()[order, None]
        combined = torch.zeros_like(tokens, dtype=torch.float32).index_add(0, token_ids, routed)

        assigned_fraction = counts.float() / order.numel()
        balance_loss = self.count * (assigned_fraction * scores.softmax(dim=-1).mean(dim=0)).sum()
        return combined.to(hidden.dtype).view_as(hidden), balance_loss


class TransformerBlock(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.heads = config.heads
        self.attn_norm = nn.RMSNorm(config.width)
        self.expert_norm = nn.RMSNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.attn_out = nn.Linear(config.width, config.width, bias=False)
        self.experts = ExpertLayer(config)
        nn.init.normal_(self.attn_out.weight, std=0.02 / (2 * config.logical_layers) ** 0.5)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, width = hidden.shape
        qkv = self.qkv(self.attn_norm(hidden)).view(batch, length, 3, self.heads, -1)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        hidden = hidden + self.attn_out(attended.transpose(1, 2).reshape(batch, length, width))
        output, balance_loss = self.experts(self.expert_norm(hidden))
        return hidden + output, balance_loss


class MoELanguageModel(nn.Module):
    def __init__(self, config: MoEConfig = MoEConfig(), *, activation_checkpointing: bool = False):
        super().__init__()
        self.config = config
        self.activation_checkpointing = activation_checkpointing
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.context_length, config.width)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.layers))
        self.block_order = config.block_order
        self.norm = nn.RMSNorm(config.width)
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.position.weight, std=0.02)

    def parameter_counts(self) -> tuple[int, int]:
        """Count stored weights and active parameter applications, including repeated uses."""
        config = self.config
        total = sum(parameter.numel() for parameter in self.parameters())
        block_total = sum(parameter.numel() for parameter in self.blocks[0].parameters())
        inactive_per_block = 3 * config.width * config.expert_width * (config.experts - config.experts_per_token)
        shared = total - config.layers * block_total
        return total, shared + config.logical_layers * (block_total - inactive_per_block)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return next-token logits and the mean router load-balancing loss."""
        if tokens.ndim != 2 or tokens.numel() == 0 or tokens.size(1) > self.config.context_length:
            raise ValueError("Expected a nonempty [batch, sequence] within the configured context length")
        positions = torch.arange(tokens.size(1), device=tokens.device)
        hidden = self.embedding(tokens) + self.position(positions)
        balance_losses = []
        # Reuse the same module and parameters on each visit to a middle block.
        for index in self.block_order:
            block = self.blocks[index]
            if self.training and self.activation_checkpointing:
                hidden, balance_loss = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden, balance_loss = block(hidden)
            balance_losses.append(balance_loss)
        logits = F.linear(self.norm(hidden), self.embedding.weight)
        return logits, torch.stack(balance_losses).mean()

    def training_loss(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, balance_loss = self(tokens)
        language_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return language_loss + self.config.balance_weight * balance_loss
