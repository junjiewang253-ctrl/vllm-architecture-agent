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

ARCHITECTURE_IR_VERSION = "0.4"
SOURCE_ANALYSIS_VERSION = "0.2"

TP_SYMBOLS = {"QKVParallelLinear", "RowParallelLinear", "VocabParallelEmbedding", "ParallelLMHead"}
PP_SYMBOLS = {"get_pp_group", "make_layers", "PPMissingLayer"}
EP_SYMBOLS = {"get_ep_group", "FusedMoE"}
CONFIG_EXPR_PATTERN = re.compile(r"\bconfig\.[A-Za-z_][A-Za-z0-9_]*\b")
DISPLAY_ROUTES = {"direct", "top_lane", "bottom_lane"}


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
