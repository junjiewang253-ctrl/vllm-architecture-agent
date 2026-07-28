"""Generic vLLM source-pattern categorization for v2.0 Source Context.

The functions in this module deliberately return candidate categories only.
They do not choose diagram pages, final architecture nodes, edges, or layout.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PatternMatch:
    """A conservative category match for a symbol seen in source."""

    category: str
    confidence: float
    reason: str


_CONSTRUCTOR_PATTERNS: tuple[tuple[str, str], ...] = (
    ("Embedding", "embedding"),
    ("Attention", "attention"),
    ("QKV", "projection"),
    ("ColumnParallel", "projection"),
    ("MergedColumnParallel", "projection"),
    ("RowParallel", "projection"),
    ("Linear", "projection"),
    ("Norm", "normalization"),
    ("LayerNorm", "normalization"),
    ("RMSNorm", "normalization"),
    ("MLP", "ffn"),
    ("FeedForward", "ffn"),
    ("FusedMoE", "moe"),
    ("MoE", "moe"),
    ("Gate", "router"),
    ("Router", "router"),
    ("LMHead", "output_head"),
    ("LogitsProcessor", "logits_processor"),
    ("Pool", "pooling"),
    ("Vision", "multimodal_encoder"),
    ("Encoder", "multimodal_encoder"),
    ("Mamba", "state_space"),
    ("SSM", "state_space"),
    ("Recurrent", "recurrent"),
    ("Loader", "loader"),
)

_CAPABILITY_TERMS: dict[str, tuple[str, ...]] = {
    "tensor_parallel": (
        "ParallelLinear",
        "ParallelEmbedding",
        "ParallelLMHead",
        "tensor_model_parallel",
        "tp_group",
    ),
    "pipeline_parallel": (
        "SupportsPP",
        "make_layers",
        "IntermediateTensors",
        "PPMissingLayer",
        "is_pp_missing_parameter",
    ),
    "expert_parallel": ("get_ep_group", "ep_rank", "ep_size", "ExpertParallel", "expert parallel group", "EPLB"),
    "moe": ("FusedMoE", "MixtureOfExperts", "num_experts", "experts"),
    "multimodal": ("MultiModal", "Vision", "Image", "MultiModalProcessor"),
    "pooling": ("Pooler", "PoolingType", "SupportsPooling", "pooling", "pooler"),
    "classification": ("Classification", "Score", "classifier"),
    "speculative": ("speculative", "Speculative", "draft"),
    "recurrent_or_ssm": ("Mamba", "SSM", "Recurrent"),
    "attention_free": ("attention_free", "AttentionFree"),
    "custom_attention": ("Attention(", "QKVParallelLinear", "attention backend", "custom_attention"),
    "custom_weight_loading": ("load_weights", "weight_loader", "AutoWeightsLoader"),
    "lora": ("SupportsLoRA", "LoRA"),
    "quantization": ("quant_config", "QuantizationConfig", "quantization"),
    "torch_compile": ("support_torch_compile", "torch.compile", "compile"),
}


def categorize_constructor(symbol: str) -> PatternMatch:
    """Return a broad constructor category without inferring final semantics."""

    for term, category in _CONSTRUCTOR_PATTERNS:
        if term in symbol:
            return PatternMatch(
                category=category,
                confidence=0.75,
                reason=f"constructor name contains {term!r}",
            )
    return PatternMatch(
        category="unknown",
        confidence=0.2,
        reason="constructor did not match a generic vLLM category",
    )


def detect_capabilities(text: str) -> dict[str, dict[str, object]]:
    """Detect candidate model capabilities from source text and symbols."""

    result: dict[str, dict[str, object]] = {}
    for capability, terms in _CAPABILITY_TERMS.items():
        matched = [term for term in terms if term in text]
        result[capability] = {
            "detected": bool(matched),
            "confidence": min(0.95, 0.35 + 0.15 * len(matched)) if matched else 0.0,
            "evidence_terms": matched,
            "reason": "matched generic source terms" if matched else "not detected",
        }
    return result


def classify_model_source(class_names: list[str], source_text: str) -> list[str]:
    """Return broad model category candidates for Agent review."""

    categories: list[str] = []
    joined_names = " ".join(class_names)
    if "ForCausalLM" in joined_names or "CausalLM" in joined_names:
        categories.append("decoder_text_generation")
    if "Embedding" in joined_names or "Pooling" in joined_names or "pooling" in source_text:
        categories.append("embedding_or_pooling")
    if "Classification" in joined_names or "classifier" in source_text:
        categories.append("classification")
    if any(term in source_text for term in ("MultiModal", "Vision", "Image")):
        categories.append("multimodal")
    if any(term in source_text for term in ("speculative", "Speculative")):
        categories.append("speculative")
    if any(term in source_text for term in ("Mamba", "SSM", "Recurrent")):
        categories.append("hybrid_or_recurrent")
    if "Attention" not in source_text and any(
        term in source_text for term in ("Mamba", "SSM", "attention_free")
    ):
        categories.append("attention_free")
    if "Transformers" in source_text or "AutoModel" in source_text:
        categories.append("transformers_backend")
    if not categories:
        categories.append("helper_or_shared_module")
    return categories
