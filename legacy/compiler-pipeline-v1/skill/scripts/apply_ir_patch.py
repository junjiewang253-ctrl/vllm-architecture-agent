#!/usr/bin/env python3
"""Apply a validated Architecture IR patch with deterministic audit metadata."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

IR_VERSION = "0.6"
PATCH_VERSION = "0.1"


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read {label}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} root must be an object")
    return data


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stable_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _hash_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _page(ir: dict[str, Any], page_id: str) -> dict[str, Any]:
    for page in ir.get("pages", []):
        if isinstance(page, dict) and page.get("id") == page_id:
            return page
    raise ValueError(f"page not found: {page_id}")


def _collection_item(page: dict[str, Any], collection: str, item_id: str) -> dict[str, Any]:
    for item in page.get(collection, []):
        if isinstance(item, dict) and item.get("id") == item_id:
            return item
    raise ValueError(f"{collection[:-1]} not found: {item_id}")


def _remove_item(page: dict[str, Any], collection: str, item_id: str) -> None:
    items = page.get(collection)
    if not isinstance(items, list):
        raise ValueError(f"page collection is not a list: {collection}")
    new_items = [item for item in items if not (isinstance(item, dict) and item.get("id") == item_id)]
    if len(new_items) == len(items):
        raise ValueError(f"{collection[:-1]} not found: {item_id}")
    page[collection] = new_items


def _item_by_id(ir: dict[str, Any], item_type: str, item_id: str, page_id: str | None) -> dict[str, Any]:
    collection = "nodes" if item_type == "node" else "edges"
    pages = [_page(ir, page_id)] if page_id else [page for page in ir.get("pages", []) if isinstance(page, dict)]
    for page in pages:
        for item in page.get(collection, []):
            if isinstance(item, dict) and item.get("id") == item_id:
                return item
    raise ValueError(f"{item_type} not found: {item_id}")


def _guard_evidence_type(item: dict[str, Any], new_type: str) -> None:
    if new_type == "direct":
        for evidence in item.get("evidence", []):
            if isinstance(evidence, dict) and evidence.get("type") == "external":
                raise ValueError("external evidence cannot be promoted to direct")


def _guard_edge_update(edge: dict[str, Any], updates: dict[str, Any]) -> None:
    if edge.get("kind") == "weight_mapping" and updates.get("kind") == "runtime":
        raise ValueError("weight_mapping edge cannot be changed to runtime by patch")
    if edge.get("phase") == "construction" and updates.get("phase") == "runtime":
        evidence_ids = [
            fact_id
            for evidence in edge.get("evidence", [])
            if isinstance(evidence, dict)
            for fact_id in evidence.get("fact_ids", [])
            if isinstance(fact_id, str)
        ]
        if not any(":call:" in fact_id or ":branch:" in fact_id for fact_id in evidence_ids):
            raise ValueError("construction edge cannot become runtime without call or branch evidence")


def _apply_operation(ir: dict[str, Any], operation: dict[str, Any]) -> None:
    op = operation.get("op")
    page_id = operation.get("page_id")
    if op == "add_page":
        page = operation.get("page")
        if not isinstance(page, dict) or not isinstance(page.get("id"), str):
            raise ValueError("add_page requires page object with id")
        if any(isinstance(existing, dict) and existing.get("id") == page["id"] for existing in ir.get("pages", [])):
            raise ValueError(f"page already exists: {page['id']}")
        ir.setdefault("pages", []).append(copy.deepcopy(page))
        return
    if op == "remove_page":
        target = operation.get("page_id")
        pages = ir.get("pages")
        if not isinstance(target, str) or not isinstance(pages, list):
            raise ValueError("remove_page requires page_id")
        new_pages = [page for page in pages if not (isinstance(page, dict) and page.get("id") == target)]
        if len(new_pages) == len(pages):
            raise ValueError(f"page not found: {target}")
        ir["pages"] = new_pages
        return
    if op == "rename_page":
        _page(ir, str(page_id))["title"] = str(operation.get("title") or operation.get("new_title"))
        return
    if op == "set_unresolved":
        item = operation.get("item")
        if not isinstance(item, dict):
            item = {"item": operation.get("item_id", "reviewed-unresolved"), "reason": operation.get("reason", "")}
        ir.setdefault("unresolved", []).append(copy.deepcopy(item))
        return
    if op == "resolve_unresolved":
        token = operation.get("item") or operation.get("item_id")
        unresolved = ir.get("unresolved")
        if not isinstance(unresolved, list):
            return
        ir["unresolved"] = [
            item for item in unresolved
            if not (
                isinstance(item, dict)
                and token in {item.get("item"), item.get("id"), item.get("kind"), item.get("name")}
            )
        ]
        return

    if not isinstance(page_id, str):
        raise ValueError(f"{op} requires page_id")
    page = _page(ir, page_id)
    if op == "add_node":
        node = operation.get("node")
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            raise ValueError("add_node requires node object with id")
        page.setdefault("nodes", []).append(copy.deepcopy(node))
    elif op == "remove_node":
        node_id = str(operation.get("node_id"))
        for edge in page.get("edges", []):
            if isinstance(edge, dict) and (edge.get("source") == node_id or edge.get("target") == node_id):
                raise ValueError(f"cannot remove node still referenced by edge: {node_id}")
        _remove_item(page, "nodes", node_id)
    elif op == "update_node":
        node = _collection_item(page, "nodes", str(operation.get("node_id")))
        updates = operation.get("updates")
        if not isinstance(updates, dict):
            raise ValueError("update_node requires updates object")
        node.update(copy.deepcopy(updates))
    elif op == "add_edge":
        edge = operation.get("edge")
        if not isinstance(edge, dict) or not isinstance(edge.get("id"), str):
            raise ValueError("add_edge requires edge object with id")
        page.setdefault("edges", []).append(copy.deepcopy(edge))
    elif op == "remove_edge":
        _remove_item(page, "edges", str(operation.get("edge_id")))
    elif op == "update_edge":
        edge = _collection_item(page, "edges", str(operation.get("edge_id")))
        updates = operation.get("updates")
        if not isinstance(updates, dict):
            raise ValueError("update_edge requires updates object")
        _guard_edge_update(edge, updates)
        edge.update(copy.deepcopy(updates))
    elif op in {"add_evidence", "replace_evidence", "set_evidence_type"}:
        item = _item_by_id(ir, str(operation.get("item_type", "node")), str(operation.get("item_id")), page_id)
        if op == "set_evidence_type":
            evidence_type = str(operation.get("evidence_type"))
            _guard_evidence_type(item, evidence_type)
            for evidence in item.setdefault("evidence", []):
                if isinstance(evidence, dict):
                    evidence["type"] = evidence_type
        else:
            evidence = operation.get("evidence")
            if not isinstance(evidence, list):
                evidence = [evidence] if isinstance(evidence, dict) else []
            if op == "replace_evidence":
                item["evidence"] = copy.deepcopy(evidence)
            else:
                item.setdefault("evidence", []).extend(copy.deepcopy(evidence))
    elif op in {"add_port", "update_port", "remove_port"}:
        node = _collection_item(page, "nodes", str(operation.get("node_id")))
        ports = node.setdefault("ports", [])
        port_id = str(operation.get("port_id") or operation.get("port", {}).get("id"))
        if op == "add_port":
            port = operation.get("port")
            if not isinstance(port, dict) or not isinstance(port.get("id"), str):
                raise ValueError("add_port requires port object with id")
            ports.append(copy.deepcopy(port))
        elif op == "remove_port":
            node["ports"] = [port for port in ports if not (isinstance(port, dict) and port.get("id") == port_id)]
        else:
            updates = operation.get("updates")
            if not isinstance(updates, dict):
                raise ValueError("update_port requires updates object")
            for port in ports:
                if isinstance(port, dict) and port.get("id") == port_id:
                    port.update(copy.deepcopy(updates))
                    break
            else:
                raise ValueError(f"port not found: {port_id}")
    elif op in {"add_annotation", "update_annotation"}:
        annotations = page.setdefault("annotations", [])
        annotation = operation.get("annotation")
        if not isinstance(annotation, dict) or not isinstance(annotation.get("id"), str):
            raise ValueError(f"{op} requires annotation object with id")
        if op == "add_annotation":
            annotations.append(copy.deepcopy(annotation))
        else:
            for existing in annotations:
                if isinstance(existing, dict) and existing.get("id") == annotation["id"]:
                    existing.update(copy.deepcopy(annotation))
                    break
            else:
                raise ValueError(f"annotation not found: {annotation['id']}")
    elif op == "move_item_to_page":
        raise ValueError("move_item_to_page is allowed by schema but not implemented for automatic v0.9 patches")
    elif op in {"merge_nodes", "split_node"}:
        raise ValueError(f"{op} requires manual semantic review and is not auto-applied")
    else:
        raise ValueError(f"unsupported patch operation: {op}")


def apply_ir_patch(base_ir_path: Path, patch_path: Path) -> dict[str, Any]:
    base_ir = _load_json(base_ir_path, "baseline Architecture IR")
    patch = _load_json(patch_path, "Architecture IR patch")
    if base_ir.get("schema_version") != IR_VERSION:
        raise ValueError(f"baseline IR schema_version must be {IR_VERSION!r}")
    if patch.get("schema_version") != PATCH_VERSION:
        raise ValueError(f"patch schema_version must be {PATCH_VERSION!r}")
    base_hash = _hash_file(base_ir_path)
    if patch.get("base_ir_sha256") != base_hash:
        raise ValueError("patch base_ir_sha256 does not match baseline IR")
    reviewed = copy.deepcopy(base_ir)
    applied: list[str] = []
    deferred: list[dict[str, Any]] = list(patch.get("deferred_operations", [])) if isinstance(patch.get("deferred_operations"), list) else []
    for operation in patch.get("operations", []):
        if not isinstance(operation, dict):
            raise ValueError("patch operation must be an object")
        if float(operation.get("confidence", 0)) < 0.70:
            deferred.append(copy.deepcopy(operation))
            continue
        _apply_operation(reviewed, operation)
        applied.append(str(operation.get("op_id")))
    reviewed["review"] = {
        "mode": "agent-guided",
        "base_ir_sha256": base_hash,
        "patch_sha256": _hash_file(patch_path),
        "applied_operations": applied,
        "deferred_operations": deferred,
    }
    return reviewed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Architecture IR patch.")
    parser.add_argument("baseline_ir", type=Path)
    parser.add_argument("architecture_ir_patch", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        reviewed = apply_ir_patch(args.baseline_ir, args.architecture_ir_patch)
        rendered = _stable_json(reviewed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote reviewed Architecture IR to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
