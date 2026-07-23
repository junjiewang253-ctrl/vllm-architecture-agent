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

ARCHITECTURE_IR_VERSION = "0.5"
SOURCE_ANALYSIS_VERSION = "0.2"

TP_SYMBOLS = {"QKVParallelLinear", "RowParallelLinear", "VocabParallelEmbedding", "ParallelLMHead"}
PP_SYMBOLS = {"get_pp_group", "make_layers", "PPMissingLayer"}
EP_SYMBOLS = {"get_ep_group", "FusedMoE"}
CONFIG_EXPR_PATTERN = re.compile(r"\bconfig\.[A-Za-z_][A-Za-z0-9_]*\b")
DISPLAY_ROUTES = {"direct", "top_lane", "bottom_lane", "horizontal_lane", "vertical_branch", "local_branch", "cache_write", "cache_read", "weight_mapping", "hidden_semantic"}


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


def _import_record(data: dict[str, Any], name: str) -> dict[str, Any] | None:
    for item in data.get("imports", []):
        if not isinstance(item, dict):
            continue
        if item.get("name") == name or item.get("asname") == name:
            return item
    return None


def _class_record(data: dict[str, Any], class_name: str | None) -> dict[str, Any] | None:
    if not class_name:
        return None
    return _class_by_name(data).get(class_name)


def _class_attribute(data: dict[str, Any], owner_class: str | None, name: str) -> dict[str, Any] | None:
    for item in data.get("class_attributes", []):
        if isinstance(item, dict) and item.get("owner_class") == owner_class and item.get("name") == name:
            return item
    return None


def _parallel_hint(data: dict[str, Any], symbol: str) -> dict[str, Any] | None:
    for item in data.get("parallelism_hints", []):
        if isinstance(item, dict) and item.get("symbol") == symbol:
            return item
    return None


def _weight_mapping_kind(data: dict[str, Any], mapping_kind: str) -> dict[str, Any] | None:
    for item in data.get("weight_mappings", []):
        if isinstance(item, dict) and item.get("mapping_kind") == mapping_kind:
            return item
    return None


def _call_record_any(data: dict[str, Any], class_name: str | None, targets: list[str]) -> dict[str, Any] | None:
    for target in targets:
        record = _control_call_record(data, class_name, target)
        if record:
            return record
    return None


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


def _edge_display(
    *,
    visible: bool = True,
    label: str | None = None,
    show_label: bool | None = None,
    route: str = "direct",
) -> dict[str, Any]:
    if route not in DISPLAY_ROUTES:
        raise ValueError(f"invalid edge display route: {route}")
    result: dict[str, Any] = {"visible": visible, "route": route}
    result["show_label"] = bool(show_label) if show_label is not None else False
    if label is not None:
        result["label"] = label
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
            badges=[],
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
                badges=["TP"],
                evidence=_evidence("direct", [embedding_assignment]),
                display=_display("Token Embedding", "Vocabulary-parallel embedding"),
            )
        )
    if layer_factory and decoder_class:
        decoder_badges = ["TP", "PP"]
        if any(isinstance(item.get("component"), str) and "MoE" in item["component"] for item in variants):
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
                badges=["TP"],
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
            display=_edge_display(visible=False),
        )
    ]
    if embedding_assignment:
        edges.append(_edge(edge_id="input_to_embedding", source="input", target=embedding_id, kind="runtime", scope=model_scope, source_port="input_ids", target_port="input_ids", evidence=_evidence("direct", [embedding_assignment, _control_call_record(data, model_class, "self.embed_input_ids")]), display=_edge_display()))
    if layer_factory and decoder_class:
        edges.append(_edge(edge_id="embedding_to_decoder_layers", source=embedding_id, target=decoder_id, kind="runtime", scope=model_scope, evidence=_evidence("derived", [embedding_assignment, layer_factory, layer_call]), display=_edge_display()))
    if final_add_record and layer_factory and decoder_class:
        edges.append(_edge(edge_id="decoder_to_final_add", source=decoder_id, target=final_add_id, kind="runtime", scope=model_scope, condition="last PP rank", evidence=_evidence("derived", [layer_factory, final_add_record]), display=_edge_display()))
        edges.append(_edge(edge_id="residual_to_final_add", source=decoder_id, target=final_add_id, kind="residual", scope=model_scope, target_port="residual", evidence=_evidence("direct", [final_add_record]), display=_edge_display(label="residual", show_label=True, route="top_lane")))
    if final_norm_assignment:
        edges.append(_edge(edge_id="final_add_to_norm", source=final_add_id if final_add_record else decoder_id, target=final_norm_id, kind="runtime", scope=model_scope, evidence=_evidence("direct", [final_add_record, _control_call_record(data, model_class, "self.norm")]), display=_edge_display()))
    if final_norm_assignment and logits_assignment:
        edges.append(_edge(edge_id="model_hidden_states_to_logits_processor", source=final_norm_id, target="logits_processor", kind="summary", scope=logits_scope, label="hidden_states", source_port="hidden_states", target_port="hidden_states", evidence=_evidence("derived", [final_norm_assignment, logits_method]), display=_edge_display(label="hidden states", show_label=True)))
    if lm_head_assignment and logits_assignment:
        edges.append(_edge(edge_id="lm_head_to_logits_processor", source="lm_head", target="logits_processor", kind="dependency", scope=logits_scope, label="lm_head", source_port="lm_head", target_port="lm_head", evidence=_evidence("direct", [lm_head_assignment, logits_method]), display=_edge_display(label="lm_head", show_label=True)))

    return {"id": "overview", "title": "Model Overview", "page_type": "overview", "nodes": nodes, "edges": edges}


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
        _edge(edge_id="decoder_input_to_input_layernorm", source="decoder_input", target="input_layernorm", kind="runtime", scope=scope, evidence=_evidence("direct", [input_norm_call]), display=_edge_display()),
        _edge(edge_id="input_layernorm_to_self_attention", source="input_layernorm", target="self_attention", kind="runtime", scope=scope, evidence=_evidence("direct", [input_norm_call, self_attn_call]), display=_edge_display()),
        _edge(edge_id="self_attention_to_attention_residual", source="self_attention", target="attention_residual", kind="runtime", scope=scope, evidence=_evidence("derived", [self_attn_call, post_norm_call]), display=_edge_display()),
        _edge(edge_id="decoder_input_residual_to_attention_residual", source="decoder_input", target="attention_residual", kind="residual", scope=scope, evidence=_evidence("derived", [post_norm_call]), display=_edge_display(label="residual", show_label=True, route="top_lane")),
        _edge(edge_id="attention_residual_to_post_attention_layernorm", source="attention_residual", target="post_attention_layernorm", kind="runtime", scope=scope, evidence=_evidence("direct", [post_norm_call]), display=_edge_display()),
        _edge(edge_id="post_attention_layernorm_to_ffn_stage", source="post_attention_layernorm", target="ffn_stage", kind="runtime", scope=scope, evidence=_evidence("direct", [post_norm_call, mlp_call]), display=_edge_display()),
        _edge(edge_id="ffn_stage_to_ffn_residual", source="ffn_stage", target="ffn_residual", kind="runtime", scope=scope, evidence=_evidence("direct", [mlp_call]), display=_edge_display()),
        _edge(edge_id="attention_residual_to_ffn_residual", source="attention_residual", target="ffn_residual", kind="residual", scope=scope, evidence=_evidence("derived", [post_norm_call, mlp_call]), display=_edge_display(label="residual", show_label=True, route="bottom_lane")),
        _edge(edge_id="ffn_residual_to_decoder_output", source="ffn_residual", target="decoder_output", kind="runtime", scope=scope, evidence=_evidence("derived", [mlp_call, decoder_forward]), display=_edge_display()),
    ]
    return {"id": "decoder_layer_detail", "title": "HYV3DecoderLayer Detail", "page_type": "decoder_detail", "nodes": nodes, "edges": edges}


def _build_attention_detail_page(data: dict[str, Any], core: dict[str, Any]) -> dict[str, Any]:
    attention_class = "HYV3Attention"
    scope = f"{attention_class}.forward"
    qkv_assignment = _assignment_by_attribute(data, attention_class, "qkv_proj")
    o_proj_assignment = _assignment_by_attribute(data, attention_class, "o_proj")
    attn_assignment = _assignment_by_attribute(data, attention_class, "attn")
    hpc_assignment = _assignment_by_attribute(data, attention_class, "hpc_rope_norm")
    rotary_assignment = _assignment_by_attribute(data, attention_class, "rotary_emb")
    q_norm_assignment = _assignment_by_attribute(data, attention_class, "q_norm")
    k_norm_assignment = _assignment_by_attribute(data, attention_class, "k_norm")
    forward = _forward_control_flow(data, attention_class)
    qkv_call = _control_call_record(data, attention_class, "self.qkv_proj")
    split_call = _control_call_record(data, attention_class, "qkv.split")
    hpc_call = _control_call_record(data, attention_class, "self.hpc_rope_norm")
    q_norm_call = _control_call_record(data, attention_class, "self.q_norm")
    k_norm_call = _control_call_record(data, attention_class, "self.k_norm")
    rotary_call = _control_call_record(data, attention_class, "self.rotary_emb")
    attn_call = _control_call_record(data, attention_class, "self.attn")
    o_proj_call = _control_call_record(data, attention_class, "self.o_proj")
    condition = next(
        (
            item
            for item in data.get("conditions", [])
            if isinstance(item, dict)
            and item.get("owner_class") == attention_class
            and "hpc_rope_norm" in str(item.get("condition"))
        ),
        None,
    )

    nodes = [
        _node(node_id="attention_input", label="hidden_states", kind="input", scope=scope, evidence=_evidence("derived", [forward]), display=_display("Hidden States")),
        _node(node_id="qkv_projection", label="QKVParallelLinear", kind="attention", scope=scope, badges=["TP"], evidence=_evidence("direct", [qkv_assignment, qkv_call]), display=_display("QKV Projection", "QKVParallelLinear")),
        _node(node_id="split_qkv", label="Split Q / K / V", kind="note", scope=scope, evidence=_evidence("derived", [split_call, qkv_call]), display=_display("Split Q / K / V")),
        _node(node_id="attention_path", label="Position & QK Processing", kind="container", scope=scope, evidence=_evidence("derived", [condition, hpc_assignment, rotary_assignment]), display=_display("Position & QK Processing")),
        _node(node_id="hpc_rope_norm", label="HpcRopeNorm", kind="normalization", scope=scope, parent_id="attention_path", evidence=_evidence("direct", [hpc_assignment, hpc_call]), display=_display("HPC Fused Path", "QK Norm + RoPE + KV Cache Write")),
        _node(node_id="fallback_qk_norm", label="Q/K RMSNorm", kind="normalization", scope=scope, parent_id="attention_path", evidence=_evidence("direct", [q_norm_assignment, k_norm_assignment, q_norm_call, k_norm_call]), display=_display("Optional Q/K RMSNorm")),
        _node(node_id="rotary_embedding", label="Rotary Embedding", kind="embedding", scope=scope, parent_id="attention_path", evidence=_evidence("direct", [rotary_assignment, rotary_call]), display=_display("Rotary Embedding")),
        _node(node_id="attention_core", label="Attention", kind="attention", scope=scope, evidence=_evidence("direct", [attn_assignment, attn_call]), display=_display("vLLM Attention", "Paged KV-cache attention")),
        _node(node_id="output_projection", label="RowParallelLinear", kind="head", scope=scope, badges=["TP"], evidence=_evidence("direct", [o_proj_assignment, o_proj_call]), display=_display("Output Projection", "RowParallelLinear")),
        _node(node_id="attention_output", label="Attention Output", kind="output", scope=scope, evidence=_evidence("derived", [o_proj_call, forward]), display=_display("Attention Output")),
    ]
    hpc_condition = str(condition.get("condition")) if isinstance(condition, dict) else "self.hpc_rope_norm is not None"
    edges = [
        _edge(edge_id="attention_input_to_qkv_projection", source="attention_input", target="qkv_projection", kind="runtime", scope=scope, evidence=_evidence("direct", [qkv_call]), display=_edge_display()),
        _edge(edge_id="qkv_projection_to_split_qkv", source="qkv_projection", target="split_qkv", kind="runtime", scope=scope, evidence=_evidence("derived", [qkv_call, split_call]), display=_edge_display()),
        _edge(edge_id="split_qkv_to_hpc_path", source="split_qkv", target="hpc_rope_norm", kind="conditional_true", scope=scope, condition=hpc_condition, evidence=_evidence("direct", [condition, hpc_call]), display=_edge_display(label="HPC supported", show_label=True, route="top_lane")),
        _edge(edge_id="hpc_path_to_attention_core", source="hpc_rope_norm", target="attention_core", kind="runtime", scope=scope, evidence=_evidence("direct", [hpc_call, attn_call]), display=_edge_display(route="top_lane")),
        _edge(edge_id="split_qkv_to_fallback_path", source="split_qkv", target="fallback_qk_norm", kind="conditional_false", scope=scope, condition=f"not ({hpc_condition})", evidence=_evidence("derived", [condition, q_norm_call, k_norm_call]), display=_edge_display(label="fallback", show_label=True, route="bottom_lane")),
        _edge(edge_id="fallback_qk_norm_to_rotary_embedding", source="fallback_qk_norm", target="rotary_embedding", kind="runtime", scope=scope, evidence=_evidence("direct", [q_norm_call, k_norm_call, rotary_call]), display=_edge_display(route="bottom_lane")),
        _edge(edge_id="rotary_embedding_to_attention_core", source="rotary_embedding", target="attention_core", kind="runtime", scope=scope, evidence=_evidence("direct", [rotary_call, attn_call]), display=_edge_display(route="bottom_lane")),
        _edge(edge_id="attention_core_to_output_projection", source="attention_core", target="output_projection", kind="runtime", scope=scope, evidence=_evidence("direct", [attn_call, o_proj_call]), display=_edge_display()),
        _edge(edge_id="output_projection_to_attention_output", source="output_projection", target="attention_output", kind="runtime", scope=scope, evidence=_evidence("derived", [o_proj_call, forward]), display=_edge_display()),
    ]
    return {"id": "attention_detail", "title": "HYV3Attention Detail", "page_type": "attention_detail", "nodes": nodes, "edges": edges}


def _region_node(node_id: str, label: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    return _node(node_id=node_id, label=label, kind="container", evidence=evidence, display=_display(label, show_badges=False))


def _adapt_node(
    data: dict[str, Any],
    *,
    node_id: str,
    label: str,
    parent_id: str,
    kind: str = "note",
    records: list[dict[str, Any] | None] | None = None,
    badges: list[str] | None = None,
    phase: str = "runtime",
    scope: str | None = None,
    display_label: str | None = None,
) -> dict[str, Any]:
    return _node(
        node_id=node_id,
        label=label,
        kind=kind,
        phase=phase,
        scope=scope,
        parent_id=parent_id,
        badges=badges,
        evidence=_evidence("direct", records or [_import_record(data, label)], note=None),
        display=_display(display_label or label),
    )


def _build_adaptation_map_page(data: dict[str, Any], core: dict[str, Any]) -> dict[str, Any]:
    top_level_class = core["top_level_class"]
    model_class = core["model_class"]
    packed_attr = _class_attribute(data, top_level_class, "packed_modules_mapping")
    stacked_mapping = _weight_mapping_kind(data, "stacked_parameter")
    packed_mapping = _weight_mapping_kind(data, "packed_module")
    expert_hint = next(
        (
            item
            for item in data.get("weight_loading_hints", [])
            if isinstance(item, dict) and "expert" in str(item.get("value"))
        ),
        None,
    )
    top_class = _class_record(data, top_level_class)
    model_class_record = _class_record(data, model_class)
    model_decorator = next(
        (
            item
            for item in (model_class_record or {}).get("decorators", [])
            if isinstance(item, dict) and item.get("name") == "support_torch_compile"
        ),
        None,
    )
    common_evidence = _evidence("derived", [top_class, model_class_record])
    nodes = [
        _region_node("region_hf_inputs", "Hugging Face Inputs", _evidence("derived", [_class_record(data, "HYV3Config"), packed_attr])),
        _region_node("region_vllm_config", "vLLM Configuration", _evidence("derived", [_import_record(data, "VllmConfig"), _import_record(data, "CacheConfig")])),
        _region_node("region_adapter_interfaces", "Adapter Interfaces", common_evidence),
        _region_node("region_execution_components", "vLLM Execution Components", _evidence("derived", [_parallel_hint(data, "QKVParallelLinear"), _parallel_hint(data, "FusedMoE")])),
        _region_node("region_weight_parallel", "Weight and Parallel Adaptation", _evidence("derived", [packed_attr, stacked_mapping, _parallel_hint(data, "get_pp_group"), _parallel_hint(data, "get_ep_group")])),
        _adapt_node(data, node_id="hyv3_config", label="HYV3Config", parent_id="region_hf_inputs", kind="note", records=[_class_record(data, "HYV3Config")]),
        _adapt_node(data, node_id="hf_checkpoint", label="Hugging Face Checkpoint", parent_id="region_hf_inputs", kind="note", records=[packed_attr, stacked_mapping, expert_hint]),
        _adapt_node(data, node_id="vllm_config", label="VllmConfig", parent_id="region_vllm_config", records=[_import_record(data, "VllmConfig")]),
        _adapt_node(data, node_id="cache_config", label="CacheConfig", parent_id="region_vllm_config", records=[_import_record(data, "CacheConfig"), _assignment_by_attribute(data, "HYV3Attention", "attn")]),
        _adapt_node(data, node_id="quantization_config", label="QuantizationConfig", parent_id="region_vllm_config", records=[_import_record(data, "QuantizationConfig")]),
        _adapt_node(data, node_id="parallel_config_eplb_config", label="ParallelConfig / EPLBConfig", parent_id="region_vllm_config", records=[_parallel_hint(data, "get_ep_group"), _parallel_hint(data, "get_pp_group")]),
        _adapt_node(data, node_id="adapter_hyv3_for_causal_lm", label=str(top_level_class), parent_id="region_adapter_interfaces", kind="container", records=[top_class]),
        _adapt_node(data, node_id="adapter_hyv3_model", label=str(model_class), parent_id="region_adapter_interfaces", kind="container", records=[model_class_record]),
        _adapt_node(data, node_id="supports_pp", label="SupportsPP", parent_id="region_adapter_interfaces", records=[_import_record(data, "SupportsPP"), top_class]),
        _adapt_node(data, node_id="supports_lora", label="SupportsLoRA", parent_id="region_adapter_interfaces", records=[_import_record(data, "SupportsLoRA"), top_class]),
        _adapt_node(data, node_id="mixture_of_experts", label="MixtureOfExperts", parent_id="region_adapter_interfaces", records=[_import_record(data, "MixtureOfExperts"), model_class_record]),
        _adapt_node(data, node_id="support_torch_compile", label="support_torch_compile", parent_id="region_adapter_interfaces", records=[_import_record(data, "support_torch_compile"), model_decorator]),
        _adapt_node(data, node_id="adapt_vocab_parallel_embedding", label="VocabParallelEmbedding", parent_id="region_execution_components", kind="embedding", records=[_assignment_by_attribute(data, model_class, "embed_tokens")], badges=["TP"]),
        _adapt_node(data, node_id="adapt_qkv_parallel_linear", label="QKVParallelLinear", parent_id="region_execution_components", kind="attention", records=[_assignment_by_attribute(data, "HYV3Attention", "qkv_proj")], badges=["TP"]),
        _adapt_node(data, node_id="adapt_row_parallel_linear", label="RowParallelLinear", parent_id="region_execution_components", kind="head", records=[_assignment_by_attribute(data, "HYV3Attention", "o_proj")], badges=["TP"]),
        _adapt_node(data, node_id="adapt_vllm_attention", label="vLLM Attention", parent_id="region_execution_components", kind="attention", records=[_assignment_by_attribute(data, "HYV3Attention", "attn")]),
        _adapt_node(data, node_id="adapt_fused_moe", label="FusedMoE", parent_id="region_execution_components", kind="moe", records=[_assignment_by_attribute(data, "HYV3MoEFused", "experts")], badges=["EP"]),
        _adapt_node(data, node_id="adapt_parallel_lm_head", label="ParallelLMHead", parent_id="region_execution_components", kind="head", records=[_assignment_by_attribute(data, top_level_class, "lm_head")], badges=["TP"]),
        _adapt_node(data, node_id="adapt_logits_processor", label="LogitsProcessor", parent_id="region_execution_components", kind="logits_processor", records=[_assignment_by_attribute(data, top_level_class, "logits_processor")]),
        _adapt_node(data, node_id="packed_modules_mapping", label="packed_modules_mapping", parent_id="region_weight_parallel", kind="note", phase="checkpoint_loading", records=[packed_attr, packed_mapping]),
        _adapt_node(data, node_id="stacked_params_mapping", label="stacked_params_mapping", parent_id="region_weight_parallel", kind="note", phase="checkpoint_loading", records=[stacked_mapping]),
        _adapt_node(data, node_id="expert_parameter_mapping", label="expert parameter mapping", parent_id="region_weight_parallel", kind="note", phase="checkpoint_loading", records=[expert_hint]),
        _adapt_node(data, node_id="auto_weights_loader", label="AutoWeightsLoader", parent_id="region_weight_parallel", kind="note", phase="checkpoint_loading", records=[_import_record(data, "AutoWeightsLoader"), _methods(top_class).get("load_weights") if top_class else None]),
        _adapt_node(data, node_id="tensor_parallel", label="Tensor Parallel", parent_id="region_weight_parallel", records=[_parallel_hint(data, "QKVParallelLinear"), _parallel_hint(data, "ParallelLMHead")], display_label="Tensor Parallel"),
        _adapt_node(data, node_id="pipeline_parallel", label="Pipeline Parallel", parent_id="region_weight_parallel", records=[_parallel_hint(data, "make_layers"), _parallel_hint(data, "get_pp_group")], display_label="Pipeline Parallel"),
        _adapt_node(data, node_id="expert_parallel", label="Expert Parallel", parent_id="region_weight_parallel", records=[_parallel_hint(data, "FusedMoE"), _parallel_hint(data, "get_ep_group")], display_label="Expert Parallel"),
    ]
    e = _evidence
    edges = [
        _edge(edge_id="hyv3_config_to_vllm_config", source="hyv3_config", target="vllm_config", kind="dependency", evidence=e("derived", [_import_record(data, "VllmConfig"), _class_record(data, "HYV3Config")]), display=_edge_display(label="model config", show_label=True)),
        _edge(edge_id="vllm_config_to_cache_config", source="vllm_config", target="cache_config", kind="dependency", evidence=e("direct", [_import_record(data, "CacheConfig")]), display=_edge_display()),
        _edge(edge_id="vllm_config_to_quant_config", source="vllm_config", target="quantization_config", kind="dependency", evidence=e("direct", [_import_record(data, "QuantizationConfig")]), display=_edge_display()),
        _edge(edge_id="vllm_config_to_parallel_config", source="vllm_config", target="parallel_config_eplb_config", kind="dependency", evidence=e("derived", [_parallel_hint(data, "get_pp_group"), _parallel_hint(data, "get_ep_group")]), display=_edge_display()),
        _edge(edge_id="adapter_to_supports_pp", source="adapter_hyv3_for_causal_lm", target="supports_pp", kind="adaptation", evidence=e("direct", [top_class]), display=_edge_display(label="interface", show_label=True)),
        _edge(edge_id="adapter_to_supports_lora", source="adapter_hyv3_for_causal_lm", target="supports_lora", kind="adaptation", evidence=e("direct", [top_class]), display=_edge_display()),
        _edge(edge_id="model_to_mixture_of_experts", source="adapter_hyv3_model", target="mixture_of_experts", kind="adaptation", evidence=e("direct", [model_class_record]), display=_edge_display()),
        _edge(edge_id="compile_decorator_to_model", source="support_torch_compile", target="adapter_hyv3_model", kind="adaptation", evidence=e("direct", [model_decorator]), display=_edge_display()),
        _edge(edge_id="adapter_to_model_body", source="adapter_hyv3_for_causal_lm", target="adapter_hyv3_model", kind="adaptation", evidence=e("direct", [core.get("model_assignment")]), display=_edge_display()),
        _edge(edge_id="model_to_embedding_component", source="adapter_hyv3_model", target="adapt_vocab_parallel_embedding", kind="adaptation", evidence=e("direct", [_assignment_by_attribute(data, model_class, "embed_tokens")]), display=_edge_display()),
        _edge(edge_id="model_to_attention_components", source="adapter_hyv3_model", target="adapt_qkv_parallel_linear", kind="adaptation", evidence=e("direct", [_assignment_by_attribute(data, "HYV3Attention", "qkv_proj")]), display=_edge_display()),
        _edge(edge_id="attention_to_attention_core_component", source="adapt_qkv_parallel_linear", target="adapt_vllm_attention", kind="adaptation", evidence=e("direct", [_assignment_by_attribute(data, "HYV3Attention", "attn")]), display=_edge_display()),
        _edge(edge_id="attention_to_output_linear_component", source="adapt_vllm_attention", target="adapt_row_parallel_linear", kind="adaptation", evidence=e("direct", [_assignment_by_attribute(data, "HYV3Attention", "o_proj")]), display=_edge_display()),
        _edge(edge_id="model_to_fused_moe_component", source="adapter_hyv3_model", target="adapt_fused_moe", kind="adaptation", evidence=e("direct", [_assignment_by_attribute(data, "HYV3MoEFused", "experts")]), display=_edge_display()),
        _edge(edge_id="adapter_to_lm_head_component", source="adapter_hyv3_for_causal_lm", target="adapt_parallel_lm_head", kind="adaptation", evidence=e("direct", [_assignment_by_attribute(data, top_level_class, "lm_head")]), display=_edge_display()),
        _edge(edge_id="adapter_to_logits_processor_component", source="adapter_hyv3_for_causal_lm", target="adapt_logits_processor", kind="adaptation", evidence=e("direct", [_assignment_by_attribute(data, top_level_class, "logits_processor")]), display=_edge_display()),
        _edge(edge_id="checkpoint_to_packed_mapping", source="hf_checkpoint", target="packed_modules_mapping", kind="weight_mapping", phase="checkpoint_loading", evidence=e("direct", [packed_attr, packed_mapping]), display=_edge_display(label="packed", show_label=True)),
        _edge(edge_id="checkpoint_to_stacked_mapping", source="hf_checkpoint", target="stacked_params_mapping", kind="weight_mapping", phase="checkpoint_loading", evidence=e("direct", [stacked_mapping]), display=_edge_display(label="stacked", show_label=True)),
        _edge(edge_id="checkpoint_to_expert_mapping", source="hf_checkpoint", target="expert_parameter_mapping", kind="weight_mapping", phase="checkpoint_loading", evidence=e("direct", [expert_hint]), display=_edge_display(label="expert", show_label=True)),
        _edge(edge_id="mappings_to_loader_packed", source="packed_modules_mapping", target="auto_weights_loader", kind="weight_mapping", phase="checkpoint_loading", evidence=e("direct", [packed_attr, _methods(top_class).get("load_weights") if top_class else None]), display=_edge_display()),
        _edge(edge_id="mappings_to_loader_stacked", source="stacked_params_mapping", target="auto_weights_loader", kind="weight_mapping", phase="checkpoint_loading", evidence=e("direct", [stacked_mapping, _methods(model_class_record).get("load_weights") if model_class_record else None]), display=_edge_display()),
        _edge(edge_id="tp_to_embedding", source="tensor_parallel", target="adapt_vocab_parallel_embedding", kind="parallel_partition", phase="parallel_partition", evidence=e("direct", [_parallel_hint(data, "VocabParallelEmbedding")]), display=_edge_display(label="TP", show_label=True)),
        _edge(edge_id="tp_to_linear_layers", source="tensor_parallel", target="adapt_qkv_parallel_linear", kind="parallel_partition", phase="parallel_partition", evidence=e("direct", [_parallel_hint(data, "QKVParallelLinear")]), display=_edge_display(label="TP", show_label=True)),
        _edge(edge_id="tp_to_lm_head", source="tensor_parallel", target="adapt_parallel_lm_head", kind="parallel_partition", phase="parallel_partition", evidence=e("direct", [_parallel_hint(data, "ParallelLMHead")]), display=_edge_display(label="TP", show_label=True)),
        _edge(edge_id="pp_to_model_body", source="pipeline_parallel", target="adapter_hyv3_model", kind="parallel_partition", phase="parallel_partition", evidence=e("direct", [_parallel_hint(data, "make_layers"), _parallel_hint(data, "get_pp_group")]), display=_edge_display(label="PP", show_label=True)),
        _edge(edge_id="ep_to_fused_moe", source="expert_parallel", target="adapt_fused_moe", kind="parallel_partition", phase="parallel_partition", evidence=e("direct", [_parallel_hint(data, "FusedMoE"), _parallel_hint(data, "get_ep_group")]), display=_edge_display(label="EP", show_label=True)),
    ]
    return {"id": "vllm_adaptation_map", "title": "HY V3 Model Adapter in vLLM", "page_type": "adaptation_map", "nodes": nodes, "edges": edges}


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
    """Build a four-page Architecture IR from source-analysis data."""
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
            _build_attention_detail_page(data, core),
            _build_adaptation_map_page(data, core),
        ],
        "unresolved": unresolved,
    }


def _port(port_id: str, label: str, direction: str, data_kind: str) -> dict[str, str]:
    return {"id": port_id, "label": label, "direction": direction, "data_kind": data_kind}


def _ports(node: dict[str, Any], ports: list[dict[str, str]]) -> dict[str, Any]:
    node["ports"] = ports
    return node


def _node_v08(
    node_id: str,
    label: str,
    kind: str,
    evidence: list[dict[str, Any]],
    *,
    display_label: str | None = None,
    display_subtitle: str | None = None,
    parent_id: str | None = None,
    phase: str = "runtime",
    scope: str | None = None,
    badges: list[str] | None = None,
    ports: list[dict[str, str]] | None = None,
    variants: list[dict[str, Any]] | None = None,
    repetition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    node = _node(
        node_id=node_id,
        label=label,
        kind=kind,
        evidence=evidence,
        phase=phase,
        scope=scope,
        parent_id=parent_id,
        badges=badges,
        display=_display(display_label or label, display_subtitle),
        variants=variants,
        repetition=repetition,
    )
    if ports is not None:
        node["ports"] = ports
    return node


def _edge_v08(
    edge_id: str,
    source: str,
    target: str,
    kind: str,
    evidence: list[dict[str, Any]],
    *,
    source_port: str,
    target_port: str,
    phase: str = "runtime",
    scope: str | None = None,
    label: str | None = None,
    visible: bool = True,
    show_label: bool = False,
    route: str = "direct",
    condition: str | None = None,
) -> dict[str, Any]:
    return _edge(
        edge_id=edge_id,
        source=source,
        target=target,
        kind=kind,
        evidence=evidence,
        phase=phase,
        scope=scope,
        label=label,
        condition=condition,
        source_port=source_port,
        target_port=target_port,
        display=_edge_display(visible=visible, label=label, show_label=show_label, route=route),
    )


TENSOR_IN = [_port("hidden_in", "hidden states", "input", "tensor")]
TENSOR_OUT = [_port("hidden_out", "hidden states", "output", "tensor")]
TENSOR_IO = [
    _port("hidden_in", "hidden states", "input", "tensor"),
    _port("hidden_out", "hidden states", "output", "tensor"),
]
WEIGHTS_OUT = [_port("weights_out", "weights/config", "output", "weights")]
CONFIG_OUT = [_port("config_out", "configuration", "output", "config")]
CAP_OUT = [_port("capability_out", "capability", "output", "capability")]
CAP_IN = [_port("capability_in", "capability", "input", "capability")]


def _build_overview_page_v08(data: dict[str, Any], core: dict[str, Any], variants: list[dict[str, Any]]) -> dict[str, Any]:
    classes = core["classes"]
    top_level_class = core["top_level_class"]
    model_class = core["model_class"]
    model_assignment = core["model_assignment"]
    embedding_assignment = core["embedding_assignment"]
    layer_factory = core["layer_factory"]
    decoder_class = core["decoder_class"]
    final_norm_assignment = core["final_norm_assignment"]
    lm_head_assignment = core["lm_head_assignment"]
    logits_assignment = core["logits_assignment"]
    final_add_record = _find_add_record(data, model_class)
    layer_call = _control_call_record(data, model_class, "layer", resolved_collection="self.layers")
    logits_method = _methods(classes.get(top_level_class)).get("compute_logits") if top_level_class else None
    model_scope = f"{model_class}.forward" if model_class else None
    logits_scope = f"{top_level_class}.compute_logits" if top_level_class else None
    decoder_badges = ["TP", "PP", "EP"]
    nodes = [
        _node_v08("hyv3_for_causal_lm", str(top_level_class), "container", _evidence("direct", [classes.get(top_level_class)]), display_subtitle="Causal LM wrapper", ports=[_port("model_out", "model output", "output", "tensor")]),
        _node_v08("hyv3_model", str(model_class), "container", _evidence("direct", [classes.get(model_class), model_assignment]), parent_id="hyv3_for_causal_lm", display_subtitle="Transformer body", ports=TENSOR_IO),
        _node_v08("input", "Input IDs / Embeds", "input", _evidence("derived", [_forward_control_flow(data, model_class)]), parent_id="hyv3_model", display_label="Input", ports=[_port("tokens_out", "tokens", "output", "tensor")]),
        _node_v08("vocab_parallel_embedding", "VocabParallelEmbedding", "embedding", _evidence("direct", [embedding_assignment]), parent_id="hyv3_model", display_label="Token Embedding", display_subtitle="Vocab-parallel", badges=["TP"], ports=[_port("tokens_in", "tokens", "input", "tensor"), _port("hidden_out", "hidden states", "output", "tensor")]),
        _node_v08("hyv3_decoder_layer", str(decoder_class), "repeated_block", _evidence("derived", [classes.get(decoder_class), layer_factory, layer_call]), parent_id="hyv3_model", display_label="N x Decoder Layers", display_subtitle="Pipeline-local stack", badges=decoder_badges, ports=TENSOR_IO, repetition={"count_expression": layer_factory.get("repeat_expression") if layer_factory else "config.num_hidden_layers", "local_start": "self.start_layer", "local_end": "self.end_layer"}, variants=variants),
        _node_v08("final_residual_add", "Final Residual Add", "add", _evidence("direct", [final_add_record]), parent_id="hyv3_model", display_label="Final Residual Add", ports=[_port("hidden_in", "hidden states", "input", "tensor"), _port("residual_in", "residual", "input", "tensor"), _port("hidden_out", "hidden states", "output", "tensor")]),
        _node_v08("norm", "RMSNorm", "normalization", _evidence("direct", [final_norm_assignment, _control_call_record(data, model_class, "self.norm")]), parent_id="hyv3_model", display_label="Final RMSNorm", ports=TENSOR_IO),
        _node_v08("hidden_states_output", "Hidden States", "output", _evidence("derived", [final_norm_assignment]), display_label="Hidden States", ports=[_port("hidden_in", "hidden states", "input", "tensor"), _port("hidden_out", "hidden states", "output", "tensor")]),
        _node_v08("lm_head", "ParallelLMHead", "head", _evidence("direct", [lm_head_assignment]), display_label="LM Head", display_subtitle="TP output weights", badges=["TP"], ports=[_port("lm_head_out", "lm_head", "output", "weights")]),
        _node_v08("logits_processor", "LogitsProcessor", "logits_processor", _evidence("direct", [logits_assignment, logits_method]), display_label="Logits Processor", ports=[_port("hidden_states", "hidden states", "input", "tensor"), _port("lm_head", "lm_head", "input", "weights"), _port("logits", "logits", "output", "tensor")]),
        _node_v08("logits_output", "Logits", "output", _evidence("derived", [logits_method]), display_label="Logits", ports=[_port("logits_in", "logits", "input", "tensor")]),
    ]
    edges = [
        _edge_v08("top_invokes_model", "hyv3_for_causal_lm", "hyv3_model", "invocation", _evidence("direct", [model_assignment], lines=[line for line in [_call_line(data, top_level_class, "self.model")] if line]), source_port="model_out", target_port="hidden_in", scope=f"{top_level_class}.forward" if top_level_class else None, visible=False, route="hidden_semantic"),
        _edge_v08("input_to_embedding", "input", "vocab_parallel_embedding", "runtime", _evidence("direct", [embedding_assignment, _control_call_record(data, model_class, "self.embed_input_ids")]), source_port="tokens_out", target_port="tokens_in", scope=model_scope, route="horizontal_lane"),
        _edge_v08("embedding_to_decoder_layers", "vocab_parallel_embedding", "hyv3_decoder_layer", "runtime", _evidence("derived", [embedding_assignment, layer_factory, layer_call]), source_port="hidden_out", target_port="hidden_in", scope=model_scope, route="horizontal_lane"),
        _edge_v08("decoder_to_final_add", "hyv3_decoder_layer", "final_residual_add", "runtime", _evidence("derived", [layer_factory, final_add_record]), source_port="hidden_out", target_port="hidden_in", scope=model_scope, route="horizontal_lane"),
        _edge_v08("decoder_residual_to_final_add", "hyv3_decoder_layer", "final_residual_add", "residual", _evidence("direct", [final_add_record]), source_port="hidden_out", target_port="residual_in", scope=model_scope, label="residual", show_label=True, route="horizontal_lane"),
        _edge_v08("final_add_to_norm", "final_residual_add", "norm", "runtime", _evidence("direct", [final_add_record, _control_call_record(data, model_class, "self.norm")]), source_port="hidden_out", target_port="hidden_in", scope=model_scope, route="horizontal_lane"),
        _edge_v08("norm_to_hidden_states_output", "norm", "hidden_states_output", "runtime", _evidence("derived", [final_norm_assignment]), source_port="hidden_out", target_port="hidden_in", route="horizontal_lane"),
        _edge_v08("hidden_states_to_logits_processor", "hidden_states_output", "logits_processor", "runtime", _evidence("derived", [final_norm_assignment, logits_method]), source_port="hidden_out", target_port="hidden_states", scope=logits_scope, route="horizontal_lane"),
        _edge_v08("lm_head_to_logits_processor", "lm_head", "logits_processor", "dependency", _evidence("direct", [lm_head_assignment, logits_method]), source_port="lm_head_out", target_port="lm_head", scope=logits_scope, label="lm_head", show_label=True, route="vertical_branch"),
        _edge_v08("logits_processor_to_logits", "logits_processor", "logits_output", "runtime", _evidence("derived", [logits_assignment, logits_method]), source_port="logits", target_port="logits_in", scope=logits_scope, route="horizontal_lane"),
    ]
    return {"id": "overview", "title": "Model Overview", "page_type": "overview", "nodes": nodes, "edges": edges}


def _build_decoder_detail_page_v08(data: dict[str, Any], core: dict[str, Any], variants: list[dict[str, Any]]) -> dict[str, Any]:
    decoder_class = core["decoder_class"]
    attention_assignment = core["attention_assignment"]
    scope = f"{decoder_class}.forward" if decoder_class else None
    input_norm = _assignment_by_attribute(data, decoder_class, "input_layernorm")
    post_norm = _assignment_by_attribute(data, decoder_class, "post_attention_layernorm")
    mlp_call = _control_call_record(data, decoder_class, "self.mlp")
    self_attn_call = _control_call_record(data, decoder_class, "self.self_attn")
    input_norm_call = _control_call_record(data, decoder_class, "self.input_layernorm")
    post_norm_call = _control_call_record(data, decoder_class, "self.post_attention_layernorm")
    decoder_forward = _forward_control_flow(data, decoder_class)
    variant_records = [item for item in data.get("conditions", []) if isinstance(item, dict) and item.get("owner_class") == decoder_class and "first_k_dense_replace" in str(item.get("condition"))]
    norm_ports = [
        _port("hidden_states", "hidden states", "input", "tensor"),
        _port("residual", "residual", "input", "tensor"),
        _port("normalized_hidden", "normalized hidden", "output", "tensor"),
        _port("updated_residual", "updated residual", "output", "tensor"),
    ]
    nodes = [
        _node_v08("decoder_input", "Hidden States + Residual", "input", _evidence("derived", [decoder_forward]), display_label="Decoder Input", ports=[_port("hidden_states", "hidden states", "output", "tensor"), _port("residual", "residual", "output", "tensor")]),
        _node_v08("input_rmsnorm", "RMSNorm", "normalization", _evidence("direct", [input_norm, input_norm_call]), display_label="Input RMSNorm", display_subtitle="fused residual", ports=norm_ports),
        _node_v08("self_attention", "HYV3Attention", "attention", _evidence("direct", [_class_record(data, str(attention_assignment.get("constructor"))) if attention_assignment else None, attention_assignment, self_attn_call]), display_label="Self Attention", badges=["TP"], ports=[_port("hidden_states", "hidden states", "input", "tensor"), _port("attention_output", "attention output", "output", "tensor")]),
        _node_v08("post_attention_rmsnorm", "RMSNorm", "normalization", _evidence("direct", [post_norm, post_norm_call]), display_label="Post-Attention RMSNorm", display_subtitle="fused residual", ports=[_port("attention_output", "attention output", "input", "tensor"), _port("residual", "residual", "input", "tensor"), _port("normalized_hidden", "normalized hidden", "output", "tensor"), _port("updated_residual", "updated residual", "output", "tensor")]),
        _node_v08("ffn_stage", "Feed-Forward Stage", "container", _evidence("derived", variant_records + [mlp_call]), display_label="Feed-Forward Stage", ports=[_port("hidden_states", "hidden states", "input", "tensor"), _port("ffn_output", "ffn output", "output", "tensor")], variants=variants),
        _node_v08("dense_ffn", "HYV3FeedForward", "ffn", _evidence("direct", variant_records + _assignments(data, owner_class=decoder_class, attribute="mlp", constructor="HYV3FeedForward")), parent_id="ffn_stage", phase="construction", display_label="Dense FFN", ports=TENSOR_IO),
        _node_v08("moe_ffn", "HYV3MoEFused", "moe", _evidence("direct", variant_records + _assignments(data, owner_class=decoder_class, attribute="mlp", constructor="HYV3MoEFused")), parent_id="ffn_stage", phase="construction", display_label="MoE", badges=["EP"], ports=TENSOR_IO),
        _node_v08("decoder_output", "Hidden States + Residual", "output", _evidence("derived", [decoder_forward, mlp_call]), display_label="Decoder Output", ports=[_port("hidden_states", "hidden states", "input", "tensor"), _port("residual", "residual", "input", "tensor")]),
    ]
    edges = [
        _edge_v08("decoder_hidden_to_input_norm", "decoder_input", "input_rmsnorm", "runtime", _evidence("direct", [input_norm_call]), source_port="hidden_states", target_port="hidden_states", scope=scope, route="horizontal_lane"),
        _edge_v08("input_norm_to_attention", "input_rmsnorm", "self_attention", "runtime", _evidence("direct", [input_norm_call, self_attn_call]), source_port="normalized_hidden", target_port="hidden_states", scope=scope, route="horizontal_lane"),
        _edge_v08("attention_to_post_norm", "self_attention", "post_attention_rmsnorm", "runtime", _evidence("derived", [self_attn_call, post_norm_call]), source_port="attention_output", target_port="attention_output", scope=scope, route="horizontal_lane"),
        _edge_v08("post_norm_to_ffn", "post_attention_rmsnorm", "ffn_stage", "runtime", _evidence("direct", [post_norm_call, mlp_call]), source_port="normalized_hidden", target_port="hidden_states", scope=scope, route="horizontal_lane"),
        _edge_v08("ffn_to_decoder_output", "ffn_stage", "decoder_output", "runtime", _evidence("direct", [mlp_call]), source_port="ffn_output", target_port="hidden_states", scope=scope, route="horizontal_lane"),
        _edge_v08("decoder_residual_to_input_norm", "decoder_input", "input_rmsnorm", "residual", _evidence("derived", [input_norm_call]), source_port="residual", target_port="residual", scope=scope, label="residual", show_label=True, route="horizontal_lane"),
        _edge_v08("input_norm_residual_to_post_norm", "input_rmsnorm", "post_attention_rmsnorm", "residual", _evidence("derived", [input_norm_call, post_norm_call]), source_port="updated_residual", target_port="residual", scope=scope, label="residual", show_label=True, route="horizontal_lane"),
        _edge_v08("post_norm_residual_to_decoder_output", "post_attention_rmsnorm", "decoder_output", "residual", _evidence("derived", [post_norm_call, mlp_call]), source_port="updated_residual", target_port="residual", scope=scope, label="residual", show_label=True, route="horizontal_lane"),
    ]
    return {"id": "decoder_layer_detail", "title": "HYV3DecoderLayer Detail", "page_type": "decoder_detail", "nodes": nodes, "edges": edges}


def _build_attention_detail_page_v08(data: dict[str, Any], core: dict[str, Any]) -> dict[str, Any]:
    attention_class = "HYV3Attention"
    scope = f"{attention_class}.forward"
    condition = next((item for item in data.get("conditions", []) if isinstance(item, dict) and item.get("owner_class") == attention_class and "hpc_rope_norm" in str(item.get("condition"))), None)
    qkv_assignment = _assignment_by_attribute(data, attention_class, "qkv_proj")
    o_proj_assignment = _assignment_by_attribute(data, attention_class, "o_proj")
    attn_assignment = _assignment_by_attribute(data, attention_class, "attn")
    hpc_assignment = _assignment_by_attribute(data, attention_class, "hpc_rope_norm")
    rotary_assignment = _assignment_by_attribute(data, attention_class, "rotary_emb")
    q_norm_assignment = _assignment_by_attribute(data, attention_class, "q_norm")
    k_norm_assignment = _assignment_by_attribute(data, attention_class, "k_norm")
    qkv_call = _control_call_record(data, attention_class, "self.qkv_proj")
    hpc_call = _control_call_record(data, attention_class, "self.hpc_rope_norm")
    q_norm_call = _control_call_record(data, attention_class, "self.q_norm")
    k_norm_call = _control_call_record(data, attention_class, "self.k_norm")
    rotary_call = _control_call_record(data, attention_class, "self.rotary_emb")
    attn_call = _control_call_record(data, attention_class, "self.attn")
    o_proj_call = _control_call_record(data, attention_class, "self.o_proj")
    forward = _forward_control_flow(data, attention_class)
    nodes = [
        _node_v08("attention_input", "Hidden States", "input", _evidence("derived", [forward]), display_label="Hidden States", ports=[_port("hidden_out", "hidden states", "output", "tensor")]),
        _node_v08("qkv_projection", "QKVParallelLinear", "attention", _evidence("direct", [qkv_assignment, qkv_call]), display_label="QKV Projection", display_subtitle="QKVParallelLinear", badges=["TP"], ports=[_port("hidden_in", "hidden states", "input", "tensor"), _port("qkv_out", "qkv", "output", "tensor")]),
        _node_v08("qkv_split", "Split Q/K/V", "note", _evidence("derived", [qkv_call]), display_label="Split Q / K / V", ports=[_port("qkv_in", "qkv", "input", "tensor"), _port("q", "Q", "output", "tensor"), _port("k", "K", "output", "tensor"), _port("v", "V", "output", "tensor")]),
        _node_v08("q_stream", "Q Stream", "note", _evidence("derived", [qkv_call]), display_label="Q", ports=[_port("q_in", "Q", "input", "tensor"), _port("q_out", "Q", "output", "tensor")]),
        _node_v08("k_stream", "K Stream", "note", _evidence("derived", [qkv_call]), display_label="K", ports=[_port("k_in", "K", "input", "tensor"), _port("k_out", "K", "output", "tensor")]),
        _node_v08("v_stream", "V Stream", "note", _evidence("derived", [qkv_call]), display_label="V", ports=[_port("v_in", "V", "input", "tensor"), _port("v_out", "V", "output", "tensor")]),
        _node_v08("hpc_fused_processing", "HpcRopeNorm", "normalization", _evidence("direct", [hpc_assignment, hpc_call]), display_label="HPC Fused Processing", display_subtitle="QK norm + RoPE + cache write", ports=[_port("qkv_in", "QKV", "input", "tensor"), _port("processed_q", "processed Q", "output", "tensor"), _port("kv_write", "K/V write", "output", "tensor")]),
        _node_v08("fallback_q_norm", "Q RMSNorm", "normalization", _evidence("direct", [q_norm_assignment, q_norm_call]), display_label="Q RMSNorm", ports=[_port("q_in", "Q", "input", "tensor"), _port("q_out", "Q", "output", "tensor")]),
        _node_v08("fallback_k_norm", "K RMSNorm", "normalization", _evidence("direct", [k_norm_assignment, k_norm_call]), display_label="K RMSNorm", ports=[_port("k_in", "K", "input", "tensor"), _port("k_out", "K", "output", "tensor")]),
        _node_v08("rotary_embedding", "Rotary Embedding", "embedding", _evidence("direct", [rotary_assignment, rotary_call]), display_label="Rotary Embedding", ports=[_port("q_in", "Q", "input", "tensor"), _port("k_in", "K", "input", "tensor"), _port("q_out", "Q", "output", "tensor"), _port("k_out", "K", "output", "tensor")]),
        _node_v08("kv_cache", "KV Cache", "note", _evidence("derived", [hpc_call, attn_assignment]), display_label="KV Cache", ports=[_port("write", "write", "input", "cache"), _port("read", "read", "output", "cache")]),
        _node_v08("attention_core", "Attention", "attention", _evidence("direct", [attn_assignment, attn_call]), display_label="vLLM Attention", display_subtitle="Paged KV-cache attention", ports=[_port("q", "Q", "input", "tensor"), _port("k", "K", "input", "tensor"), _port("v", "V", "input", "tensor"), _port("kv_cache", "KV cache", "input", "cache"), _port("output", "attention output", "output", "tensor")]),
        _node_v08("output_projection", "RowParallelLinear", "head", _evidence("direct", [o_proj_assignment, o_proj_call]), display_label="Output Projection", display_subtitle="RowParallelLinear", badges=["TP"], ports=[_port("attention_in", "attention output", "input", "tensor"), _port("hidden_out", "hidden states", "output", "tensor")]),
        _node_v08("attention_output", "Attention Output", "output", _evidence("derived", [o_proj_call, forward]), display_label="Attention Output", ports=[_port("hidden_in", "hidden states", "input", "tensor")]),
    ]
    hpc_condition = str(condition.get("condition")) if isinstance(condition, dict) else "self.hpc_rope_norm is not None"
    edges = [
        _edge_v08("attention_input_to_qkv_projection", "attention_input", "qkv_projection", "runtime", _evidence("direct", [qkv_call]), source_port="hidden_out", target_port="hidden_in", scope=scope, route="horizontal_lane"),
        _edge_v08("qkv_projection_to_qkv_split", "qkv_projection", "qkv_split", "runtime", _evidence("derived", [qkv_call]), source_port="qkv_out", target_port="qkv_in", scope=scope, route="horizontal_lane"),
        _edge_v08("qkv_split_to_hpc_fused", "qkv_split", "hpc_fused_processing", "conditional_true", _evidence("direct", [condition, hpc_call]), source_port="q", target_port="qkv_in", scope=scope, condition=hpc_condition, label="HPC fused", show_label=True, route="local_branch"),
        _edge_v08("hpc_processed_q_to_attention", "hpc_fused_processing", "attention_core", "runtime", _evidence("direct", [hpc_call, attn_call]), source_port="processed_q", target_port="q", scope=scope, route="local_branch"),
        _edge_v08("hpc_kv_write_to_cache", "hpc_fused_processing", "kv_cache", "runtime", _evidence("direct", [hpc_call]), source_port="kv_write", target_port="write", scope=scope, label="cache write", show_label=False, route="cache_write"),
        _edge_v08("kv_cache_read_to_attention", "kv_cache", "attention_core", "runtime", _evidence("derived", [hpc_call, attn_call]), source_port="read", target_port="kv_cache", scope=scope, label="cache read", show_label=False, route="cache_read"),
        _edge_v08("qkv_split_to_q_stream", "qkv_split", "q_stream", "conditional_false", _evidence("derived", [condition, q_norm_call]), source_port="q", target_port="q_in", scope=scope, condition=f"not ({hpc_condition})", label="fallback", show_label=True, route="local_branch"),
        _edge_v08("qkv_split_to_k_stream", "qkv_split", "k_stream", "conditional_false", _evidence("derived", [condition, k_norm_call]), source_port="k", target_port="k_in", scope=scope, condition=f"not ({hpc_condition})", route="local_branch"),
        _edge_v08("qkv_split_to_v_stream", "qkv_split", "v_stream", "conditional_false", _evidence("derived", [condition, attn_call]), source_port="v", target_port="v_in", scope=scope, condition=f"not ({hpc_condition})", route="local_branch"),
        _edge_v08("q_stream_to_q_norm", "q_stream", "fallback_q_norm", "runtime", _evidence("direct", [q_norm_call]), source_port="q_out", target_port="q_in", scope=scope, route="horizontal_lane"),
        _edge_v08("k_stream_to_k_norm", "k_stream", "fallback_k_norm", "runtime", _evidence("direct", [k_norm_call]), source_port="k_out", target_port="k_in", scope=scope, route="horizontal_lane"),
        _edge_v08("q_norm_to_rotary", "fallback_q_norm", "rotary_embedding", "runtime", _evidence("direct", [q_norm_call, rotary_call]), source_port="q_out", target_port="q_in", scope=scope, route="horizontal_lane"),
        _edge_v08("k_norm_to_rotary", "fallback_k_norm", "rotary_embedding", "runtime", _evidence("direct", [k_norm_call, rotary_call]), source_port="k_out", target_port="k_in", scope=scope, route="horizontal_lane"),
        _edge_v08("rotary_q_to_attention", "rotary_embedding", "attention_core", "runtime", _evidence("direct", [rotary_call, attn_call]), source_port="q_out", target_port="q", scope=scope, route="local_branch"),
        _edge_v08("rotary_k_to_attention", "rotary_embedding", "attention_core", "runtime", _evidence("direct", [rotary_call, attn_call]), source_port="k_out", target_port="k", scope=scope, route="local_branch"),
        _edge_v08("v_stream_to_attention", "v_stream", "attention_core", "runtime", _evidence("direct", [attn_call]), source_port="v_out", target_port="v", scope=scope, route="local_branch"),
        _edge_v08("attention_core_to_output_projection", "attention_core", "output_projection", "runtime", _evidence("direct", [attn_call, o_proj_call]), source_port="output", target_port="attention_in", scope=scope, route="horizontal_lane"),
        _edge_v08("output_projection_to_attention_output", "output_projection", "attention_output", "runtime", _evidence("derived", [o_proj_call, forward]), source_port="hidden_out", target_port="hidden_in", scope=scope, route="horizontal_lane"),
    ]
    return {"id": "attention_detail", "title": "HYV3Attention Detail", "page_type": "attention_detail", "nodes": nodes, "edges": edges}


def _build_moe_detail_page_v08(data: dict[str, Any]) -> dict[str, Any]:
    moe_class = "HYV3MoEFused"
    scope = f"{moe_class}.forward"
    gate_assignment = _assignment_by_attribute(data, moe_class, "gate")
    experts_assignment = _assignment_by_attribute(data, moe_class, "experts")
    shared_assignment = _assignment_by_attribute(data, moe_class, "shared_mlp")
    expert_bias_assignment = _assignment_by_attribute(data, moe_class, "expert_bias")
    gate_call = _control_call_record(data, moe_class, "self.gate")
    experts_call = _control_call_record(data, moe_class, "self.experts")
    forward = _forward_control_flow(data, moe_class)
    ep_hint = _parallel_hint(data, "get_ep_group")
    nodes = [
        _node_v08("moe_input", "Hidden States", "input", _evidence("derived", [forward]), display_label="Hidden States", ports=[_port("hidden_out", "hidden states", "output", "tensor")]),
        _node_v08("reshape_tokens", "Flatten Tokens", "note", _evidence("derived", [forward]), display_label="Flatten Tokens", ports=TENSOR_IO),
        _node_v08("gate_linear", "GateLinear", "ffn", _evidence("direct", [gate_assignment, gate_call]), display_label="GateLinear", ports=[_port("hidden_in", "hidden states", "input", "tensor"), _port("router_logits", "router logits", "output", "tensor")]),
        _node_v08("router_logits", "Router Logits", "note", _evidence("direct", [gate_call]), display_label="Router Logits", display_subtitle="Top-K routing", ports=[_port("router_in", "router logits", "input", "tensor"), _port("router_out", "router logits", "output", "tensor")]),
        _node_v08("fused_moe", "FusedMoE", "container", _evidence("direct", [experts_assignment, experts_call, ep_hint]), display_label="FusedMoE", display_subtitle="Expert parallel", badges=["EP"], ports=[_port("hidden_in", "hidden states", "input", "tensor"), _port("router_logits", "router logits", "input", "tensor"), _port("expert_bias", "expert bias", "input", "tensor"), _port("eplb_metadata", "EPLB metadata", "input", "control"), _port("moe_out", "moe output", "output", "tensor")]),
        _node_v08("routed_experts", "Routed Experts", "moe", _evidence("direct", [experts_assignment]), parent_id="fused_moe", display_label="Routed experts", ports=TENSOR_IO),
        _node_v08("shared_experts", "Optional Shared Experts", "ffn", _evidence("direct", [shared_assignment]), parent_id="fused_moe", display_label="Optional shared experts", ports=TENSOR_IO),
        _node_v08("expert_bias", "Expert Bias", "note", _evidence("direct", [expert_bias_assignment]), display_label="Expert Bias", ports=[_port("bias_out", "expert bias", "output", "tensor")]),
        _node_v08("eplb_metadata", "EPLB Metadata", "note", _evidence("derived", [ep_hint], note="EPLB and physical/local expert metadata are recorded from HYV3MoEFused initialization."), display_label="EPLB metadata", ports=[_port("metadata_out", "EPLB metadata", "output", "control")]),
        _node_v08("moe_output", "Output", "output", _evidence("derived", [experts_call, forward]), display_label="Output", ports=[_port("hidden_in", "hidden states", "input", "tensor")]),
    ]
    edges = [
        _edge_v08("moe_input_to_reshape", "moe_input", "reshape_tokens", "runtime", _evidence("derived", [forward]), source_port="hidden_out", target_port="hidden_in", scope=scope, route="horizontal_lane"),
        _edge_v08("reshape_to_gate", "reshape_tokens", "gate_linear", "runtime", _evidence("direct", [gate_call]), source_port="hidden_out", target_port="hidden_in", scope=scope, route="horizontal_lane"),
        _edge_v08("gate_to_router_logits", "gate_linear", "router_logits", "runtime", _evidence("direct", [gate_call]), source_port="router_logits", target_port="router_in", scope=scope, route="horizontal_lane"),
        _edge_v08("router_logits_to_fused_moe", "router_logits", "fused_moe", "runtime", _evidence("direct", [experts_call]), source_port="router_out", target_port="router_logits", scope=scope, label="Top-K routing", show_label=False, route="horizontal_lane"),
        _edge_v08("reshape_to_fused_moe", "reshape_tokens", "fused_moe", "runtime", _evidence("direct", [experts_call]), source_port="hidden_out", target_port="hidden_in", scope=scope, route="horizontal_lane"),
        _edge_v08("expert_bias_to_fused_moe", "expert_bias", "fused_moe", "runtime", _evidence("direct", [expert_bias_assignment, experts_assignment]), source_port="bias_out", target_port="expert_bias", label="bias", show_label=False, route="vertical_branch"),
        _edge_v08("eplb_metadata_to_fused_moe", "eplb_metadata", "fused_moe", "parallel_partition", _evidence("derived", [ep_hint]), source_port="metadata_out", target_port="eplb_metadata", phase="parallel_partition", label="EPLB", show_label=True, route="vertical_branch"),
        _edge_v08("fused_moe_to_output", "fused_moe", "moe_output", "runtime", _evidence("direct", [experts_call]), source_port="moe_out", target_port="hidden_in", scope=scope, route="horizontal_lane"),
    ]
    return {"id": "moe_detail", "title": "HYV3MoEFused Detail", "page_type": "moe_detail", "nodes": nodes, "edges": edges}


def _build_adapter_integration_page_v08(data: dict[str, Any], core: dict[str, Any]) -> dict[str, Any]:
    top = core["top_level_class"]
    model = core["model_class"]
    top_class = _class_record(data, top)
    model_class = _class_record(data, model)
    model_decorator = next((item for item in (model_class or {}).get("decorators", []) if isinstance(item, dict) and item.get("name") == "support_torch_compile"), None)
    nodes = [
        _node_v08("external_hf_config", "HYV3Config", "note", _evidence("direct", [_class_record(data, "HYV3Config")]), display_label="HYV3Config", ports=[_port("config_out", "HF config", "output", "config")]),
        _node_v08("vllm_config", "VllmConfig", "note", _evidence("direct", [_import_record(data, "VllmConfig")]), display_label="VllmConfig", ports=[_port("config_in", "config", "input", "config"), _port("config_out", "config", "output", "config")]),
        _node_v08("cache_config", "CacheConfig", "note", _evidence("direct", [_import_record(data, "CacheConfig")]), display_label="CacheConfig", ports=[_port("config_in", "config", "input", "config")]),
        _node_v08("quantization_config", "QuantizationConfig", "note", _evidence("direct", [_import_record(data, "QuantizationConfig")]), display_label="QuantizationConfig", ports=[_port("config_in", "config", "input", "config")]),
        _node_v08("parallel_config_eplb", "ParallelConfig / EPLBConfig", "note", _evidence("derived", [_parallel_hint(data, "get_pp_group"), _parallel_hint(data, "get_ep_group")]), display_label="ParallelConfig / EPLBConfig", ports=[_port("config_in", "config", "input", "config")]),
        _node_v08("adapter_for_causal_lm", str(top), "container", _evidence("direct", [top_class]), display_label=str(top), badges=["PP"], ports=[_port("capability_out", "interfaces", "output", "capability"), _port("component_out", "components", "output", "control")]),
        _node_v08("adapter_model", str(model), "container", _evidence("direct", [model_class]), display_label=str(model), ports=[_port("capability_out", "interfaces", "output", "capability"), _port("component_out", "components", "output", "control")]),
        _node_v08("supports_pp", "SupportsPP", "note", _evidence("direct", [_import_record(data, "SupportsPP"), top_class]), display_label="SupportsPP", ports=CAP_IN),
        _node_v08("supports_lora", "SupportsLoRA", "note", _evidence("direct", [_import_record(data, "SupportsLoRA"), top_class]), display_label="SupportsLoRA", ports=CAP_IN),
        _node_v08("mixture_of_experts", "MixtureOfExperts", "note", _evidence("direct", [_import_record(data, "MixtureOfExperts"), model_class]), display_label="MixtureOfExperts", ports=CAP_IN),
        _node_v08("support_torch_compile", "support_torch_compile", "note", _evidence("direct", [_import_record(data, "support_torch_compile"), model_decorator]), display_label="support_torch_compile", ports=CAP_IN),
        _node_v08("exec_input_group", "Input Components", "container", _evidence("direct", [_assignment_by_attribute(data, model, "embed_tokens")]), display_label="Input", ports=[_port("control_in", "component", "input", "control")]),
        _node_v08("exec_attention_group", "Attention Components", "container", _evidence("direct", [_assignment_by_attribute(data, "HYV3Attention", "qkv_proj"), _assignment_by_attribute(data, "HYV3Attention", "attn")]), display_label="Attention", ports=[_port("control_in", "component", "input", "control")]),
        _node_v08("exec_ffn_moe_group", "FFN/MoE Components", "container", _evidence("direct", [_assignment_by_attribute(data, "HYV3FeedForward", "gate_up_proj"), _assignment_by_attribute(data, "HYV3MoEFused", "experts")]), display_label="FFN / MoE", ports=[_port("control_in", "component", "input", "control")]),
        _node_v08("exec_output_group", "Output Components", "container", _evidence("direct", [_assignment_by_attribute(data, top, "lm_head"), _assignment_by_attribute(data, top, "logits_processor")]), display_label="Output", ports=[_port("control_in", "component", "input", "control")]),
    ]
    edges = [
        _edge_v08("hf_config_to_vllm_config", "external_hf_config", "vllm_config", "dependency", _evidence("derived", [_class_record(data, "HYV3Config"), _import_record(data, "VllmConfig")]), source_port="config_out", target_port="config_in", label="hf_config", show_label=True, route="local_branch"),
        _edge_v08("vllm_to_cache_config", "vllm_config", "cache_config", "dependency", _evidence("direct", [_import_record(data, "CacheConfig")]), source_port="config_out", target_port="config_in", route="local_branch"),
        _edge_v08("vllm_to_quant_config", "vllm_config", "quantization_config", "dependency", _evidence("direct", [_import_record(data, "QuantizationConfig")]), source_port="config_out", target_port="config_in", route="local_branch"),
        _edge_v08("vllm_to_parallel_config", "vllm_config", "parallel_config_eplb", "dependency", _evidence("derived", [_parallel_hint(data, "get_pp_group"), _parallel_hint(data, "get_ep_group")]), source_port="config_out", target_port="config_in", route="local_branch"),
        _edge_v08("causal_lm_to_supports_pp", "adapter_for_causal_lm", "supports_pp", "adaptation", _evidence("direct", [top_class]), source_port="capability_out", target_port="capability_in", route="local_branch"),
        _edge_v08("causal_lm_to_supports_lora", "adapter_for_causal_lm", "supports_lora", "adaptation", _evidence("direct", [top_class]), source_port="capability_out", target_port="capability_in", route="local_branch"),
        _edge_v08("model_to_moe_interface", "adapter_model", "mixture_of_experts", "adaptation", _evidence("direct", [model_class]), source_port="capability_out", target_port="capability_in", route="local_branch"),
        _edge_v08("model_to_compile_decorator", "adapter_model", "support_torch_compile", "adaptation", _evidence("direct", [model_decorator]), source_port="capability_out", target_port="capability_in", route="local_branch"),
        _edge_v08("model_to_input_components", "adapter_model", "exec_input_group", "adaptation", _evidence("direct", [_assignment_by_attribute(data, model, "embed_tokens")]), source_port="component_out", target_port="control_in", route="local_branch"),
        _edge_v08("model_to_attention_components", "adapter_model", "exec_attention_group", "adaptation", _evidence("direct", [_assignment_by_attribute(data, "HYV3Attention", "qkv_proj")]), source_port="component_out", target_port="control_in", route="local_branch"),
        _edge_v08("model_to_ffn_moe_components", "adapter_model", "exec_ffn_moe_group", "adaptation", _evidence("direct", [_assignment_by_attribute(data, "HYV3MoEFused", "experts")]), source_port="component_out", target_port="control_in", route="local_branch"),
        _edge_v08("causal_lm_to_output_components", "adapter_for_causal_lm", "exec_output_group", "adaptation", _evidence("direct", [_assignment_by_attribute(data, top, "lm_head")]), source_port="component_out", target_port="control_in", route="local_branch"),
    ]
    return {"id": "adapter_integration", "title": "vLLM Adapter Integration", "page_type": "adapter_integration", "nodes": nodes, "edges": edges}


def _build_parallelism_weight_loading_page_v08(data: dict[str, Any], core: dict[str, Any]) -> dict[str, Any]:
    top = core["top_level_class"]
    packed_attr = _class_attribute(data, top, "packed_modules_mapping")
    stacked_mapping = _weight_mapping_kind(data, "stacked_parameter")
    packed_mapping = _weight_mapping_kind(data, "packed_module")
    expert_hint = next((item for item in data.get("weight_loading_hints", []) if isinstance(item, dict) and "expert" in str(item.get("value"))), None)
    nodes = [
        _node_v08("tensor_parallel_lane", "Tensor Parallel", "container", _evidence("direct", [_parallel_hint(data, "QKVParallelLinear")]), display_label="Tensor Parallel", ports=CAP_OUT),
        _node_v08("pipeline_parallel_lane", "Pipeline Parallel", "container", _evidence("direct", [_parallel_hint(data, "make_layers"), _parallel_hint(data, "get_pp_group")]), display_label="Pipeline Parallel", ports=CAP_OUT),
        _node_v08("expert_parallel_lane", "Expert Parallel", "container", _evidence("direct", [_parallel_hint(data, "FusedMoE"), _parallel_hint(data, "get_ep_group")]), display_label="Expert Parallel", ports=CAP_OUT),
        _node_v08("tp_components", "Embedding + Linear Layers", "note", _evidence("direct", [_parallel_hint(data, "VocabParallelEmbedding"), _parallel_hint(data, "QKVParallelLinear"), _parallel_hint(data, "RowParallelLinear"), _parallel_hint(data, "ParallelLMHead")]), display_label="TP layers", ports=CAP_IN),
        _node_v08("pp_components", "make_layers + IntermediateTensors", "note", _evidence("direct", [_parallel_hint(data, "make_layers"), _import_record(data, "IntermediateTensors"), _parallel_hint(data, "PPMissingLayer")]), display_label="PP layer range", ports=CAP_IN),
        _node_v08("ep_components", "FusedMoE + local experts", "note", _evidence("direct", [_parallel_hint(data, "FusedMoE"), _parallel_hint(data, "get_ep_group")]), display_label="EP experts", ports=CAP_IN),
        _node_v08("hf_checkpoint", "HF Checkpoint", "input", _evidence("derived", [packed_attr, stacked_mapping, expert_hint]), display_label="HF Checkpoint", ports=[_port("weights_out", "weights", "output", "weights")]),
        _node_v08("qkv_checkpoint_weights", "q/k/v checkpoint weights", "note", _evidence("direct", [stacked_mapping]), display_label="q_proj / k_proj / v_proj", ports=[_port("weights_in", "weights", "input", "weights"), _port("weights_out", "weights", "output", "weights")]),
        _node_v08("qkv_proj_mapping", "qkv_proj", "note", _evidence("direct", [packed_attr, stacked_mapping]), display_label="qkv_proj", ports=[_port("weights_in", "weights", "input", "weights"), _port("weights_out", "weights", "output", "weights")]),
        _node_v08("gate_up_checkpoint_weights", "gate/up checkpoint weights", "note", _evidence("direct", [stacked_mapping]), display_label="gate_proj / up_proj", ports=[_port("weights_in", "weights", "input", "weights"), _port("weights_out", "weights", "output", "weights")]),
        _node_v08("gate_up_proj_mapping", "gate_up_proj", "note", _evidence("direct", [stacked_mapping]), display_label="gate_up_proj", ports=[_port("weights_in", "weights", "input", "weights"), _port("weights_out", "weights", "output", "weights")]),
        _node_v08("expert_checkpoint_weights", "expert checkpoint weights", "note", _evidence("direct", [expert_hint]), display_label="expert checkpoint weights", ports=[_port("weights_in", "weights", "input", "weights"), _port("weights_out", "weights", "output", "weights")]),
        _node_v08("fused_moe_parameter_mapping", "FusedMoE expert parameters", "moe", _evidence("direct", [expert_hint]), display_label="FusedMoE expert params", ports=[_port("weights_in", "weights", "input", "weights"), _port("weights_out", "weights", "output", "weights")]),
        _node_v08("remaining_weights", "remaining weights", "note", _evidence("derived", [_methods(_class_record(data, top)).get("load_weights") if top else None]), display_label="remaining weights", ports=[_port("weights_in", "weights", "input", "weights"), _port("weights_out", "weights", "output", "weights")]),
        _node_v08("auto_weights_loader", "AutoWeightsLoader", "note", _evidence("direct", [_import_record(data, "AutoWeightsLoader"), _methods(_class_record(data, top)).get("load_weights") if top else None]), display_label="AutoWeightsLoader", ports=[_port("weights_in", "weights", "input", "weights")]),
        _node_v08("pp_missing_filter", "PP missing parameter filtering", "note", _evidence("direct", [_parallel_hint(data, "PPMissingLayer")]), display_label="PP missing filter", ports=[_port("weights_in", "weights", "input", "weights"), _port("weights_out", "weights", "output", "weights")]),
        _node_v08("fp8_kv_scale_remap", "FP8 KV scale remapping", "note", _evidence("direct", [_import_record(data, "maybe_remap_kv_scale_name")]), display_label="FP8 KV scale remap", ports=[_port("weights_in", "weights", "input", "weights"), _port("weights_out", "weights", "output", "weights")]),
    ]
    edges = [
        _edge_v08("tp_to_components", "tensor_parallel_lane", "tp_components", "parallel_partition", _evidence("direct", [_parallel_hint(data, "QKVParallelLinear")]), source_port="capability_out", target_port="capability_in", phase="parallel_partition", label="TP", show_label=True, route="horizontal_lane"),
        _edge_v08("pp_to_components", "pipeline_parallel_lane", "pp_components", "parallel_partition", _evidence("direct", [_parallel_hint(data, "make_layers")]), source_port="capability_out", target_port="capability_in", phase="parallel_partition", label="PP", show_label=True, route="horizontal_lane"),
        _edge_v08("ep_to_components", "expert_parallel_lane", "ep_components", "parallel_partition", _evidence("direct", [_parallel_hint(data, "FusedMoE")]), source_port="capability_out", target_port="capability_in", phase="parallel_partition", label="EP", show_label=True, route="horizontal_lane"),
        _edge_v08("checkpoint_to_qkv_weights", "hf_checkpoint", "qkv_checkpoint_weights", "weight_mapping", _evidence("direct", [stacked_mapping]), source_port="weights_out", target_port="weights_in", phase="checkpoint_loading", label="Q/K/V", show_label=True, route="weight_mapping"),
        _edge_v08("qkv_weights_to_qkv_proj", "qkv_checkpoint_weights", "qkv_proj_mapping", "weight_mapping", _evidence("direct", [packed_attr, stacked_mapping]), source_port="weights_out", target_port="weights_in", phase="checkpoint_loading", route="weight_mapping"),
        _edge_v08("checkpoint_to_gate_up_weights", "hf_checkpoint", "gate_up_checkpoint_weights", "weight_mapping", _evidence("direct", [stacked_mapping]), source_port="weights_out", target_port="weights_in", phase="checkpoint_loading", label="gate/up", show_label=True, route="weight_mapping"),
        _edge_v08("gate_up_weights_to_gate_up_proj", "gate_up_checkpoint_weights", "gate_up_proj_mapping", "weight_mapping", _evidence("direct", [stacked_mapping]), source_port="weights_out", target_port="weights_in", phase="checkpoint_loading", route="weight_mapping"),
        _edge_v08("checkpoint_to_expert_weights", "hf_checkpoint", "expert_checkpoint_weights", "weight_mapping", _evidence("direct", [expert_hint]), source_port="weights_out", target_port="weights_in", phase="checkpoint_loading", label="experts", show_label=True, route="weight_mapping"),
        _edge_v08("expert_weights_to_fused_moe_params", "expert_checkpoint_weights", "fused_moe_parameter_mapping", "weight_mapping", _evidence("direct", [expert_hint]), source_port="weights_out", target_port="weights_in", phase="checkpoint_loading", route="weight_mapping"),
        _edge_v08("checkpoint_to_remaining_weights", "hf_checkpoint", "remaining_weights", "weight_mapping", _evidence("derived", [packed_mapping]), source_port="weights_out", target_port="weights_in", phase="checkpoint_loading", route="weight_mapping"),
        _edge_v08("remaining_weights_to_loader", "remaining_weights", "auto_weights_loader", "weight_mapping", _evidence("direct", [_methods(_class_record(data, top)).get("load_weights") if top else None]), source_port="weights_out", target_port="weights_in", phase="checkpoint_loading", route="weight_mapping"),
        _edge_v08("qkv_proj_to_pp_filter", "qkv_proj_mapping", "pp_missing_filter", "weight_mapping", _evidence("direct", [_parallel_hint(data, "PPMissingLayer")]), source_port="weights_out", target_port="weights_in", phase="checkpoint_loading", route="weight_mapping"),
        _edge_v08("qkv_proj_to_fp8_remap", "qkv_proj_mapping", "fp8_kv_scale_remap", "weight_mapping", _evidence("direct", [_import_record(data, "maybe_remap_kv_scale_name")]), source_port="weights_out", target_port="weights_in", phase="checkpoint_loading", route="weight_mapping"),
    ]
    return {"id": "parallelism_weight_loading", "title": "Parallelism & Weight Loading", "page_type": "parallelism_weight_loading", "nodes": nodes, "edges": edges}


def build_architecture_ir(data: dict[str, Any]) -> dict[str, Any]:
    """Build a six-page Architecture IR 0.5 with semantic ports."""
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
        "detail_level": "full",
        "pages": [
            _build_overview_page_v08(data, core, variants),
            _build_decoder_detail_page_v08(data, core, variants),
            _build_attention_detail_page_v08(data, core),
            _build_moe_detail_page_v08(data),
            _build_adapter_integration_page_v08(data, core),
            _build_parallelism_weight_loading_page_v08(data, core),
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
