"""Small synthetic model used to develop the architecture Skill."""

from __future__ import annotations

from typing import Any


class Module:
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, hidden_states: Any) -> Any:
        return hidden_states


class LayerNorm(Module):
    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size

    def forward(self, hidden_states: Any) -> Any:
        return hidden_states


class Embedding(Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def forward(self, input_ids: Any) -> Any:
        return input_ids


class SimpleAttention(Module):
    def __init__(self, hidden_size: int) -> None:
        self.qkv_proj = Linear(hidden_size, hidden_size * 3)
        self.out_proj = Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: Any) -> Any:
        qkv = self.qkv_proj(hidden_states)
        return self.out_proj(qkv)


class SimpleMLP(Module):
    def __init__(self, hidden_size: int) -> None:
        self.gate_up_proj = Linear(hidden_size, hidden_size * 2)
        self.down_proj = Linear(hidden_size * 2, hidden_size)

    def forward(self, hidden_states: Any) -> Any:
        hidden_states = self.gate_up_proj(hidden_states)
        return self.down_proj(hidden_states)


class SimpleMoE(Module):
    def __init__(self, hidden_size: int, num_experts: int) -> None:
        self.router = Linear(hidden_size, num_experts)
        self.experts = Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: Any) -> Any:
        router_logits = self.router(hidden_states)
        return self.experts((hidden_states, router_logits))


class SimpleDecoderLayer(Module):
    def __init__(
        self,
        hidden_size: int,
        use_moe: bool,
        num_experts: int = 4,
    ) -> None:
        self.input_layernorm = LayerNorm(hidden_size)
        self.self_attn = SimpleAttention(hidden_size)
        self.post_attention_layernorm = LayerNorm(hidden_size)

        if use_moe:
            self.mlp = SimpleMoE(hidden_size, num_experts)
        else:
            self.mlp = SimpleMLP(hidden_size)

    def forward(self, hidden_states: Any) -> Any:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SimpleForCausalLM(Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        use_moe: bool = False,
    ) -> None:
        self.embed_tokens = Embedding(vocab_size, hidden_size)
        self.layer = SimpleDecoderLayer(hidden_size, use_moe)
        self.norm = LayerNorm(hidden_size)
        self.lm_head = Linear(hidden_size, vocab_size)

    def forward(self, input_ids: Any) -> Any:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)
