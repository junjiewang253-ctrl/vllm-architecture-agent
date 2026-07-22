#!/usr/bin/env python3
"""Build Architecture IR from source-analysis JSON.

The builder stays conservative: it consumes source-analysis facts, emits a
two-page Architecture IR, and leaves unsupported or external facts unresolved.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ARCHITECTURE_IR_VERSION = "0.3"
SOURCE_ANALYSIS_VERSION = "0.2"

TP_SYMBOLS = {"QKVParallelLinear", "RowParallelLinear", "VocabParallelEmbedding", "ParallelLMHead"}
PP_SYMBOLS = {"get_pp_group", "make_layers", "PPMissingLayer"}
EP_SYMBOLS = {"get_ep_group", "FusedMoE"}
CONFIG_EXPR_PATTERN = re.compile(r"\bconfig\.[A-Za-z_][A-Za-z0-9_]*\b")


def _slug(value: str | None, fallback: str) -> str:
    text = value or fallback
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or fallback


def _line(record: dict[str, Any] | None) -> int | None:
    if not record:
        return None
    value = record.get("line")
    return value if isinstance(value, int) else None


def _lines(records: list[dict[str, Any] | None]) -> list[int]:
    return sorted({line for line in (_line(record) for record in records) if line})


def _evidence(
    evidence_type: str,
    records: list[dict[str, Any] | None] | None = None,
    *,
    lines: list[int] | None = None,
    note: str | None = None,
) -> list[dict[str, Any]]:
    """Create compact source evidence while preserving line numbers."""
    all_lines = sorted(set(lines or []) | set(_lines(records or [])))
    item: dict[str, Any] = {"type": evidence_type}
    if len(all_lines) == 1:
        item["line"] = all_lines[0]
    elif all_lines:
        item["lines"] = all_lines
    if note:
        item["note"] = note
    return [item] if all_lines or note else []


def _class_by_name(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["name"]: item
        for item in data.get("classes", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }


def _methods(class_record: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not class_record:
        return {}
    return {
        item["name"]: item
        for item in class_record.get("methods", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }


def _assignments(
    data: dict[str, Any],
    *,
    owner_class: str | None = None,
    attribute: str | None = None,
    constructor: str | None = None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in data.get("module_assignments", []):
        if not isinstance(item, dict):
            continue
        if owner_class is not None and item.get("owner_class") != owner_class:
            continue
        if attribute is not None and item.get("attribute") != attribute:
            continue
        if constructor is not None and item.get("constructor") != constructor:
            continue
        result.append(item)
    return result


def _first_assignment(
    data: dict[str, Any],
    *,
    owner_class: str | None,
    attributes: set[str] | None = None,
    constructor_suffixes: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    if not owner_class:
        return None
    for item in _assignments(data, owner_class=owner_class):
        attribute = item.get("attribute")
        constructor = item.get("constructor")
        if attributes is not None and attribute not in attributes:
            continue
        if constructor_suffixes and not (
            isinstance(constructor, str) and constructor.endswith(constructor_suffixes)
        ):
            continue
        return item
    return None


def _forward_flow(data: dict[str, Any], class_name: str | None) -> dict[str, Any] | None:
    if not class_name:
        return None
    for item in data.get("forward_flows", []):
        if isinstance(item, dict) and item.get("class") == class_name:
            return item
    return None


def _forward_control_flow(data: dict[str, Any], class_name: str | None) -> dict[str, Any] | None:
    if not class_name:
        return None
    for item in data.get("forward_control_flows", []):
        if isinstance(item, dict) and item.get("class") == class_name:
            return item
    return None


def _walk_control(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        result.append(item)
        if item.get("type") == "if":
            result.extend(_walk_control(item.get("then", [])))
            result.extend(_walk_control(item.get("else", [])))
        elif item.get("type") == "for":
            result.extend(_walk_control(item.get("body", [])))
            result.extend(_walk_control(item.get("else", [])))
    return result


def _control_call_record(
    data: dict[str, Any],
    class_name: str | None,
    target: str,
    *,
    resolved_collection: str | None = None,
) -> dict[str, Any] | None:
    flow = _forward_control_flow(data, class_name)
    if not flow:
        return None
    for item in _walk_control(flow.get("body", [])):
        value = item.get("value")
        if not isinstance(value, dict) or value.get("type") != "call":
            continue
        if value.get("target") != target:
            continue
        if resolved_collection and value.get("resolved_collection") != resolved_collection:
            continue
        return {"line": value.get("line"), "source": value.get("source")}
    return None


def _call_line(data: dict[str, Any], class_name: str | None, target: str) -> int | None:
    flow = _forward_flow(data, class_name)
    if not flow:
        return None
    for call in flow.get("calls", []):
        if isinstance(call, dict) and call.get("target") == target:
            return _line(call)
    return None


def _parallel_symbols_by_context(data: dict[str, Any]) -> dict[str, set[str]]:
    by_context: dict[str, set[str]] = {}
    for hint in data.get("parallelism_hints", []):
        if not isinstance(hint, dict):
            continue
        context = hint.get("context")
        symbol = hint.get("symbol")
        if isinstance(context, str) and isinstance(symbol, str):
            by_context.setdefault(context, set()).add(symbol)
    return by_context


def _badges_for_contexts(data: dict[str, Any], contexts: list[str], *, include_nested: bool = False) -> list[str]:
    symbols: set[str] = set()
    for context, context_symbols in _parallel_symbols_by_context(data).items():
        for wanted in contexts:
            if context == wanted or (include_nested and context.startswith(wanted)):
                symbols.update(context_symbols)
    badges: list[str] = []
    if symbols & TP_SYMBOLS:
        badges.append("TP")
    if symbols & PP_SYMBOLS:
        badges.append("PP")
    if symbols & EP_SYMBOLS:
        badges.append("EP")
    return badges


def _find_top_level_class(data: dict[str, Any], unresolved: list[dict[str, Any]]) -> str | None:
    classes = _class_by_name(data)
    candidates: list[tuple[int, str, list[int]]] = []
    for class_name, class_record in classes.items():
        if not class_name.endswith("ForCausalLM"):
            continue
        score = 3
        lines = [_line(class_record)]
        attrs = {
            item.get("attribute")
            for item in _assignments(data, owner_class=class_name)
            if item.get("assignment_kind") == "submodule"
        }
        score += len(attrs & {"model", "lm_head", "logits_processor"})
        flow = _forward_flow(data, class_name)
        if flow:
            lines.append(_line(flow))
            if any(isinstance(call, dict) and call.get("target") == "self.model" for call in flow.get("calls", [])):
                score += 3
        candidates.append((score, class_name, [line for line in lines if line]))

    if not candidates:
        unresolved.append({"item": "top_level_for_causal_lm", "reason": "No ForCausalLM class found.", "evidence": []})
        return None
    candidates.sort(reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        unresolved.append(
            {
                "item": "top_level_for_causal_lm",
                "reason": "Multiple ForCausalLM candidates had the same score.",
                "candidates": [name for _, name, _ in candidates],
                "evidence": _evidence("derived", lines=sorted({line for _, _, lines in candidates for line in lines})),
            }
        )
        return None
    return candidates[0][1]


def _find_layer_factory(data: dict[str, Any], owner_class: str | None) -> dict[str, Any] | None:
    if not owner_class:
        return None
    for item in data.get("layer_factories", []):
        if not isinstance(item, dict) or item.get("owner_class") != owner_class:
            continue
        targets = item.get("targets")
        if isinstance(targets, list) and "self.layers" in targets:
            return item
    return None


def _first_mlp_assignment(items: Any) -> dict[str, Any] | None:
    if not isinstance(items, list):
        return None
    for item in items:
        if isinstance(item, dict) and item.get("attribute") == "mlp":
            return item
    return None


def _inverse_condition(condition: str) -> str:
    if "<" in condition and ">=" not in condition:
        return condition.replace("<", ">=", 1)
    return f"not ({condition})"


def _find_dense_moe_variants(data: dict[str, Any], decoder_class: str | None) -> tuple[list[dict[str, Any]], list[dict[str, Any] | None]]:
    if not decoder_class:
        return [], []
    for condition in data.get("conditions", []):
        if not isinstance(condition, dict):
            continue
        if condition.get("owner_class") != decoder_class or condition.get("method") != "__init__":
            continue
        text = condition.get("condition")
        if not isinstance(text, str) or "layer_idx" not in text or "first_k_dense_replace" not in text:
            continue
        true_assignment = _first_mlp_assignment(condition.get("true_assignments", []))
        false_assignment = _first_mlp_assignment(condition.get("false_assignments", []))
        variants: list[dict[str, Any]] = []
        if true_assignment and isinstance(true_assignment.get("constructor"), str):
            variants.append({"condition": text, "component": true_assignment["constructor"], "phase": "construction"})
        if false_assignment and isinstance(false_assignment.get("constructor"), str):
            variants.append({"condition": _inverse_condition(text), "component": false_assignment["constructor"], "phase": "construction"})
        return variants, [condition, true_assignment, false_assignment]
    return [], []


def _find_add_record(data: dict[str, Any], class_name: str | None) -> dict[str, Any] | None:
    flow = _forward_control_flow(data, class_name)
    if not flow:
        return None
    for item in _walk_control(flow.get("body", [])):
        value = item.get("value")
        if isinstance(value, dict) and value.get("type") == "add":
            source = value.get("source")
            if isinstance(source, str) and "residual" in source:
                return {"line": item.get("line") or value.get("line"), "source": item.get("source") or source}
    return None


def _assignment_by_attribute(data: dict[str, Any], owner_class: str | None, attribute: str) -> dict[str, Any] | None:
    return _first_assignment(data, owner_class=owner_class, attributes={attribute})


def _collect_config_unresolved(unresolved: list[dict[str, Any]], records: list[dict[str, Any] | None]) -> None:
    expressions: dict[str, set[int]] = {}
    for record in records:
        if not record:
            continue
        for chunk in (record.get("source"), record.get("condition"), record.get("repeat_expression")):
            if not isinstance(chunk, str):
                continue
            for match in CONFIG_EXPR_PATTERN.findall(chunk):
                expressions.setdefault(match, set())
                line = _line(record)
                if line:
                    expressions[match].add(line)
    if expressions:
        unresolved.append(
            {
                "item": "external_config_values",
                "reason": "Configuration values are symbolic in source-analysis and were not inferred concretely.",
                "expressions": sorted(expressions),
                "evidence": _evidence("direct", lines=sorted({line for lines in expressions.values() for line in lines})),
            }
        )


def _model_name(data: dict[str, Any], top_level_class: str | None) -> str:
    source_file = data.get("source_file")
    if isinstance(source_file, str) and source_file:
        return Path(source_file).stem.replace("_", "-")
    return top_level_class or "unknown-model"


def _display(label: str | None = None, subtitle: str | None = None, *, show_badges: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {"show_badges": show_badges}
    if label is not None:
        result["label"] = label
    if subtitle is not None:
        result["subtitle"] = subtitle
    return result


def _node(
    *,
    node_id: str,
    label: str,
    kind: str,
    evidence: list[dict[str, Any]],
    subtitle: str | None = None,
    phase: str = "runtime",
    scope: str | None = None,
    parent_id: str | None = None,
    badges: list[str] | None = None,
    repetition: dict[str, Any] | None = None,
    variants: list[dict[str, Any]] | None = None,
    display: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": node_id,
        "label": label,
        "subtitle": subtitle,
        "kind": kind,
        "phase": phase,
        "scope": scope,
        "parent_id": parent_id,
        "badges": badges or [],
        "evidence": evidence,
    }
    if repetition is not None:
        result["repetition"] = repetition
    if variants is not None:
        result["variants"] = variants
    if display is not None:
        result["display"] = display
    return result


def _edge(
    *,
    edge_id: str,
    source: str,
    target: str,
    kind: str,
    evidence: list[dict[str, Any]],
    phase: str = "runtime",
    scope: str | None = None,
    label: str | None = None,
    condition: str | None = None,
    source_port: str | None = None,
    target_port: str | None = None,
    display: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": edge_id,
        "source": source,
        "target": target,
        "kind": kind,
        "phase": phase,
        "scope": scope,
        "label": label,
        "condition": condition,
        "source_port": source_port,
        "target_port": target_port,
        "evidence": evidence,
    }
    if display is not None:
        result["display"] = display
    return result


def _detect_core(data: dict[str, Any], unresolved: list[dict[str, Any]]) -> dict[str, Any]:
    classes = _class_by_name(data)
    top_level_class = _find_top_level_class(data, unresolved)
    model_assignment = _first_assignment(data, owner_class=top_level_class, attributes={"model"})
    model_class = model_assignment.get("constructor") if isinstance(model_assignment, dict) else None
    if top_level_class and not isinstance(model_class, str):
        unresolved.append({"item": "self.model", "reason": "self.model target class was not identified.", "evidence": _evidence("derived", [classes.get(top_level_class)])})
        model_class = None

    layer_factory = _find_layer_factory(data, model_class)
    decoder_class = layer_factory.get("layer_constructor") if isinstance(layer_factory, dict) and isinstance(layer_factory.get("layer_constructor"), str) else None
    return {
        "classes": classes,
        "top_level_class": top_level_class,
        "model_assignment": model_assignment,
        "model_class": model_class,
        "embedding_assignment": _first_assignment(data, owner_class=model_class, attributes={"embed_tokens", "embedding", "wte"}, constructor_suffixes=("Embedding", "VocabParallelEmbedding")),
        "layer_factory": layer_factory,
        "decoder_class": decoder_class,
        "final_norm_assignment": _first_assignment(data, owner_class=model_class, attributes={"norm", "final_layernorm", "ln_f"}, constructor_suffixes=("Norm",)),
        "lm_head_assignment": _first_assignment(data, owner_class=top_level_class, attributes={"lm_head"}),
        "logits_assignment": _first_assignment(data, owner_class=top_level_class, attributes={"logits_processor"}),
        "attention_assignment": _first_assignment(data, owner_class=decoder_class, attributes={"self_attn", "attention", "attn"}, constructor_suffixes=("Attention",)),
    }


def _build_overview_page(data: dict[str, Any], core: dict[str, Any], variants: list[dict[str, Any]]) -> dict[str, Any]:
    classes: dict[str, dict[str, Any]] = core["classes"]
    top_level_class = core["top_level_class"]
    model_class = core["model_class"]
    model_assignment = core["model_assignment"]
    embedding_assignment = core["embedding_assignment"]
    layer_factory = core["layer_factory"]
    decoder_class = core["decoder_class"]
    final_norm_assignment = core["final_norm_assignment"]
    lm_head_assignment = core["lm_head_assignment"]
    logits_assignment = core["logits_assignment"]

    top_id = _slug(top_level_class, "top_level_model")
    model_id = _slug(model_class, "model")
    embedding_id = _slug(embedding_assignment.get("constructor") if embedding_assignment else None, "embedding")
    decoder_id = _slug(decoder_class, "decoder_layers")
    final_add_id = "final_residual_add"
    final_norm_id = _slug(final_norm_assignment.get("attribute") if final_norm_assignment else None, "final_norm")
    top_scope = f"{top_level_class}.forward" if top_level_class else None
    model_scope = f"{model_class}.forward" if model_class else None
    logits_scope = f"{top_level_class}.compute_logits" if top_level_class else None

    final_add_record = _find_add_record(data, model_class)
    logits_method = _methods(classes.get(top_level_class)).get("compute_logits") if top_level_class else None
    layer_call = _control_call_record(data, model_class, "layer", resolved_collection="self.layers")

    nodes = [
        _node(
            node_id=top_id,
            label=str(top_level_class),
            subtitle="ForCausalLM wrapper",
            kind="container",
            scope=top_scope,
            evidence=_evidence("direct", [classes.get(top_level_class)]),
            display=_display(str(top_level_class), "Causal language-model wrapper"),
        ),
        _node(
            node_id=model_id,
            label=str(model_class),
            subtitle="self.model",
            kind="container",
            scope=model_scope,
            parent_id=top_id,
            badges=_badges_for_contexts(data, [str(model_class)], include_nested=True),
            evidence=_evidence("direct", [classes.get(model_class), model_assignment]),
            display=_display(str(model_class), "Pipeline-parallel transformer body"),
        ),
        _node(
            node_id="input",
            label="Input IDs / Embeds",
            kind="input",
            scope=model_scope,
            parent_id=model_id,
            evidence=_evidence("derived", [_forward_control_flow(data, model_class)]),
            display=_display("Input", "Token IDs or embedded inputs"),
        ),
    ]
    if embedding_assignment:
        nodes.append(
            _node(
                node_id=embedding_id,
                label=str(embedding_assignment.get("constructor")),
                subtitle=f"self.{embedding_assignment.get('attribute')}",
                kind="embedding",
                scope=model_scope,
                parent_id=model_id,
                badges=_badges_for_contexts(data, [f"{model_class}.__init__"]),
                evidence=_evidence("direct", [embedding_assignment]),
                display=_display("Token Embedding", "Vocabulary-parallel embedding"),
            )
        )
    if layer_factory and decoder_class:
        decoder_badges = _badges_for_contexts(data, [f"{model_class}.__init__", f"{model_class}.forward"])
        if any(isinstance(item.get("component"), str) and "MoE" in item["component"] for item in variants) and "EP" not in decoder_badges:
            decoder_badges.append("EP")
        nodes.append(
            _node(
                node_id=decoder_id,
                label=decoder_class,
                subtitle="self.layers",
                kind="repeated_block",
                scope=model_scope,
                parent_id=model_id,
                badges=decoder_badges,
                repetition={
                    "count_expression": layer_factory.get("repeat_expression") or "<unresolved>",
                    "local_start": "self.start_layer",
                    "local_end": "self.end_layer",
                },
                variants=variants,
                evidence=_evidence("derived", [classes.get(decoder_class), layer_factory, layer_call]),
                display=_display("N x Decoder Layers", "Pipeline-local transformer stack"),
            )
        )
    if final_add_record:
        nodes.append(
            _node(
                node_id=final_add_id,
                label="Add Residual",
                subtitle="hidden_states + residual",
                kind="add",
                scope=model_scope,
                parent_id=model_id,
                evidence=_evidence("direct", [final_add_record]),
                display=_display("Add Residual", "Final hidden-state merge"),
            )
        )
    if final_norm_assignment:
        nodes.append(
            _node(
                node_id=final_norm_id,
                label=str(final_norm_assignment.get("constructor")),
                subtitle=f"self.{final_norm_assignment.get('attribute')}",
                kind="normalization",
                scope=model_scope,
                parent_id=model_id,
                evidence=_evidence("direct", [final_norm_assignment, _control_call_record(data, model_class, "self.norm")]),
                display=_display("Final RMSNorm"),
            )
        )
    if lm_head_assignment:
        nodes.append(
            _node(
                node_id="lm_head",
                label=str(lm_head_assignment.get("constructor")),
                subtitle="self.lm_head",
                kind="head",
                scope=logits_scope,
                parent_id=top_id,
                badges=_badges_for_contexts(data, [f"{top_level_class}.__init__"]),
                evidence=_evidence("direct", [lm_head_assignment]),
                display=_display("LM Head", "Tensor-parallel output head"),
            )
        )
    if logits_assignment:
        nodes.append(
            _node(
                node_id="logits_processor",
                label=str(logits_assignment.get("constructor")),
                subtitle="self.logits_processor",
                kind="logits_processor",
                scope=logits_scope,
                parent_id=top_id,
                evidence=_evidence("direct", [logits_assignment, logits_method]),
                display=_display("Logits Processor"),
            )
        )

    edges = [
        _edge(
            edge_id="top_invokes_model",
            source=top_id,
            target=model_id,
            kind="invocation",
            scope=top_scope,
            label="self.model",
            evidence=_evidence("direct", [model_assignment], lines=[line for line in [_call_line(data, top_level_class, "self.model")] if line]),
            display={"visible": False, "label": None},
        )
    ]
    if embedding_assignment:
        edges.append(_edge(edge_id="input_to_embedding", source="input", target=embedding_id, kind="runtime", scope=model_scope, source_port="input_ids", target_port="input_ids", evidence=_evidence("direct", [embedding_assignment, _control_call_record(data, model_class, "self.embed_input_ids")]), display={"visible": True, "label": "tokens"}))
    if layer_factory and decoder_class:
        edges.append(_edge(edge_id="embedding_to_decoder_layers", source=embedding_id, target=decoder_id, kind="runtime", scope=model_scope, evidence=_evidence("derived", [embedding_assignment, layer_factory, layer_call]), display={"visible": True, "label": "hidden states"}))
    if final_add_record and layer_factory and decoder_class:
        edges.append(_edge(edge_id="decoder_to_final_add", source=decoder_id, target=final_add_id, kind="runtime", scope=model_scope, condition="last PP rank", evidence=_evidence("derived", [layer_factory, final_add_record]), display={"visible": True, "label": "hidden states"}))
        edges.append(_edge(edge_id="residual_to_final_add", source=decoder_id, target=final_add_id, kind="residual", scope=model_scope, target_port="residual", evidence=_evidence("direct", [final_add_record]), display={"visible": True, "label": "residual"}))
    if final_norm_assignment:
        edges.append(_edge(edge_id="final_add_to_norm", source=final_add_id if final_add_record else decoder_id, target=final_norm_id, kind="runtime", scope=model_scope, evidence=_evidence("direct", [final_add_record, _control_call_record(data, model_class, "self.norm")]), display={"visible": True, "label": "hidden states"}))
    if final_norm_assignment and logits_assignment:
        edges.append(_edge(edge_id="model_hidden_states_to_logits_processor", source=final_norm_id, target="logits_processor", kind="summary", scope=logits_scope, label="hidden_states", source_port="hidden_states", target_port="hidden_states", evidence=_evidence("derived", [final_norm_assignment, logits_method]), display={"visible": True, "label": "hidden states"}))
    if lm_head_assignment and logits_assignment:
        edges.append(_edge(edge_id="lm_head_to_logits_processor", source="lm_head", target="logits_processor", kind="dependency", scope=logits_scope, label="lm_head", source_port="lm_head", target_port="lm_head", evidence=_evidence("direct", [lm_head_assignment, logits_method]), display={"visible": True, "label": "lm_head"}))

    return {"id": "overview", "title": "Model Overview", "nodes": nodes, "edges": edges}


def _build_decoder_detail_page(data: dict[str, Any], core: dict[str, Any], variants: list[dict[str, Any]]) -> dict[str, Any]:
    classes: dict[str, dict[str, Any]] = core["classes"]
    decoder_class = core["decoder_class"]
    attention_assignment = core["attention_assignment"]
    input_norm = _assignment_by_attribute(data, decoder_class, "input_layernorm")
    post_norm = _assignment_by_attribute(data, decoder_class, "post_attention_layernorm")
    mlp_call = _control_call_record(data, decoder_class, "self.mlp")
    self_attn_call = _control_call_record(data, decoder_class, "self.self_attn")
    input_norm_call = _control_call_record(data, decoder_class, "self.input_layernorm")
    post_norm_call = _control_call_record(data, decoder_class, "self.post_attention_layernorm")
    decoder_forward = _forward_control_flow(data, decoder_class)
    scope = f"{decoder_class}.forward" if decoder_class else None
    variant_records = []
    for condition in data.get("conditions", []):
        if isinstance(condition, dict) and condition.get("owner_class") == decoder_class and "first_k_dense_replace" in str(condition.get("condition")):
            variant_records.append(condition)

    nodes = [
        _node(node_id="decoder_input", label="hidden_states + residual", kind="input", scope=scope, evidence=_evidence("derived", [decoder_forward]), display=_display("Hidden States + Residual")),
        _node(node_id="input_layernorm", label=str(input_norm.get("constructor") if input_norm else "RMSNorm"), subtitle="self.input_layernorm", kind="normalization", scope=scope, evidence=_evidence("direct", [input_norm, input_norm_call]), display=_display("Input RMSNorm", "Fused residual handoff")),
        _node(node_id="self_attention", label=str(attention_assignment.get("constructor") if attention_assignment else "HYV3Attention"), subtitle="self.self_attn", kind="attention", scope=scope, badges=_badges_for_contexts(data, [f"{attention_assignment.get('constructor')}.__init__"] if attention_assignment else [], include_nested=True), evidence=_evidence("direct", [classes.get(str(attention_assignment.get("constructor"))) if attention_assignment else None, attention_assignment, self_attn_call]), display=_display("Self Attention", str(attention_assignment.get("constructor") if attention_assignment else "HYV3Attention"))),
        _node(node_id="attention_residual", label="Residual Handoff", kind="add", scope=scope, evidence=_evidence("derived", [post_norm_call], note="RMSNorm returns hidden_states and residual; this is a derived handoff abstraction."), display=_display("Residual Handoff")),
        _node(node_id="post_attention_layernorm", label=str(post_norm.get("constructor") if post_norm else "RMSNorm"), subtitle="self.post_attention_layernorm", kind="normalization", scope=scope, evidence=_evidence("direct", [post_norm, post_norm_call]), display=_display("Post-Attention RMSNorm", "Fused residual handoff")),
        _node(node_id="ffn_stage", label="Feed-Forward Stage", kind="container", scope=scope, variants=variants, evidence=_evidence("derived", variant_records + [mlp_call]), display=_display("Feed-Forward Stage")),
        _node(node_id="dense_ffn", label="HYV3FeedForward", kind="ffn", phase="construction", scope=f"{decoder_class}.__init__" if decoder_class else None, parent_id="ffn_stage", evidence=_evidence("direct", [record for record in variant_records] + [item for item in _assignments(data, owner_class=decoder_class, attribute="mlp", constructor="HYV3FeedForward")]), display=_display("Dense FFN", "HYV3FeedForward")),
        _node(node_id="moe_ffn", label="HYV3MoEFused", kind="moe", phase="construction", scope=f"{decoder_class}.__init__" if decoder_class else None, parent_id="ffn_stage", badges=["EP"], evidence=_evidence("direct", [record for record in variant_records] + [item for item in _assignments(data, owner_class=decoder_class, attribute="mlp", constructor="HYV3MoEFused")]), display=_display("Mixture of Experts", "HYV3MoEFused")),
        _node(node_id="ffn_residual", label="Residual Handoff", kind="add", scope=scope, evidence=_evidence("derived", [mlp_call], note="The layer returns hidden_states and residual; this is a derived handoff abstraction."), display=_display("Residual Handoff")),
        _node(node_id="decoder_output", label="hidden_states + residual", kind="output", scope=scope, evidence=_evidence("derived", [decoder_forward]), display=_display("Hidden States + Residual")),
    ]

    edges = [
        _edge(edge_id="decoder_input_to_input_layernorm", source="decoder_input", target="input_layernorm", kind="runtime", scope=scope, evidence=_evidence("direct", [input_norm_call]), display={"visible": True, "label": "hidden states"}),
        _edge(edge_id="input_layernorm_to_self_attention", source="input_layernorm", target="self_attention", kind="runtime", scope=scope, evidence=_evidence("direct", [input_norm_call, self_attn_call]), display={"visible": True, "label": "hidden states"}),
        _edge(edge_id="self_attention_to_attention_residual", source="self_attention", target="attention_residual", kind="runtime", scope=scope, evidence=_evidence("derived", [self_attn_call, post_norm_call]), display={"visible": True, "label": "attention output"}),
        _edge(edge_id="decoder_input_residual_to_attention_residual", source="decoder_input", target="attention_residual", kind="residual", scope=scope, evidence=_evidence("derived", [post_norm_call]), display={"visible": True, "label": "residual"}),
        _edge(edge_id="attention_residual_to_post_attention_layernorm", source="attention_residual", target="post_attention_layernorm", kind="runtime", scope=scope, evidence=_evidence("direct", [post_norm_call]), display={"visible": True, "label": "hidden states"}),
        _edge(edge_id="post_attention_layernorm_to_ffn_stage", source="post_attention_layernorm", target="ffn_stage", kind="runtime", scope=scope, evidence=_evidence("direct", [post_norm_call, mlp_call]), display={"visible": True, "label": "hidden states"}),
        _edge(edge_id="ffn_stage_to_ffn_residual", source="ffn_stage", target="ffn_residual", kind="runtime", scope=scope, evidence=_evidence("direct", [mlp_call]), display={"visible": True, "label": "ffn output"}),
        _edge(edge_id="attention_residual_to_ffn_residual", source="attention_residual", target="ffn_residual", kind="residual", scope=scope, evidence=_evidence("derived", [post_norm_call, mlp_call]), display={"visible": True, "label": "residual"}),
        _edge(edge_id="ffn_residual_to_decoder_output", source="ffn_residual", target="decoder_output", kind="runtime", scope=scope, evidence=_evidence("derived", [mlp_call, decoder_forward]), display={"visible": True, "label": "hidden states"}),
    ]
    return {"id": "decoder_layer_detail", "title": "HYV3DecoderLayer Detail", "nodes": nodes, "edges": edges}


def _add_builder_unresolved(unresolved: list[dict[str, Any]], core: dict[str, Any], variants: list[dict[str, Any]]) -> None:
    checks = [
        ("top_level_for_causal_lm", core.get("top_level_class"), "Top-level ForCausalLM class was not uniquely identified."),
        ("self.model", core.get("model_class"), "self.model target class was not identified."),
        ("embedding", core.get("embedding_assignment"), "Embedding submodule was not identified."),
        ("decoder_layers", core.get("layer_factory"), "make_layers repeated decoder factory was not identified."),
        ("final_norm", core.get("final_norm_assignment"), "Final norm submodule was not identified."),
        ("lm_head", core.get("lm_head_assignment"), "LM Head submodule was not identified."),
        ("logits_processor", core.get("logits_assignment"), "LogitsProcessor submodule was not identified."),
        ("decoder_attention", core.get("attention_assignment"), "Decoder-layer attention submodule was not identified."),
        ("dense_moe_variants", variants, "Dense/MoE construction variants were not identified."),
    ]
    for item, value, reason in checks:
        if not value:
            unresolved.append({"item": item, "reason": reason, "evidence": []})


def build_architecture_ir(data: dict[str, Any]) -> dict[str, Any]:
    """Build a two-page Architecture IR from source-analysis data."""
    if data.get("schema_version") != SOURCE_ANALYSIS_VERSION:
        raise ValueError(f"source-analysis schema_version must be {SOURCE_ANALYSIS_VERSION!r}")

    unresolved: list[dict[str, Any]] = []
    core = _detect_core(data, unresolved)
    variants, variant_records = _find_dense_moe_variants(data, core.get("decoder_class"))
    config_records = [
        core.get("embedding_assignment"),
        core.get("layer_factory"),
        core.get("final_norm_assignment"),
        core.get("lm_head_assignment"),
        core.get("logits_assignment"),
        *variant_records,
    ]
    _collect_config_unresolved(unresolved, config_records)
    _add_builder_unresolved(unresolved, core, variants)

    return {
        "schema_version": ARCHITECTURE_IR_VERSION,
        "model_name": _model_name(data, core.get("top_level_class")),
        "detail_level": "overview",
        "pages": [
            _build_overview_page(data, core, variants),
            _build_decoder_detail_page(data, core, variants),
        ],
        "unresolved": unresolved,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Architecture IR from source-analysis JSON.")
    parser.add_argument("input", type=Path, help="source-analysis JSON file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output Architecture IR JSON path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists() or not args.input.is_file():
        print(f"error: file does not exist: {args.input}", file=sys.stderr)
        return 2
    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: unable to read source-analysis JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(data, dict):
        print("error: source-analysis root must be an object", file=sys.stderr)
        return 2
    try:
        result = build_architecture_ir(data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: unable to write Architecture IR: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote Architecture IR draft to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
