"""Pydantic to OpenRouter strict JSON Schema conversion."""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel


def pydantic_model_to_strict_json_schema(model: Type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model to a strict OpenRouter JSON Schema payload.

    Parameters
    ----------
    model : Type[BaseModel]
        Pydantic model class used for structured output.

    Returns
    -------
    dict[str, Any]
        JSON Schema object with ``additionalProperties: false`` on objects.
    """
    schema = model.model_json_schema(mode="serialization")
    defs = schema.get("$defs", {})
    resolved = _resolve_schema_refs(schema, defs)
    return _enforce_strict_object_schema(resolved)


def _resolve_schema_refs(node: Any, defs: dict[str, Any]) -> Any:
    """Inline ``$ref`` entries from a Pydantic JSON Schema document."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if ref:
            ref_name = ref.rsplit("/", maxsplit=1)[-1]
            return _resolve_schema_refs(defs[ref_name], defs)
        return {key: _resolve_schema_refs(value, defs) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_schema_refs(item, defs) for item in node]
    return node


def _enforce_strict_object_schema(node: Any) -> Any:
    """Recursively enforce strict object constraints on a JSON Schema tree."""
    if not isinstance(node, dict):
        return node

    result = dict(node)
    node_type = result.get("type")

    if node_type == "object" or "properties" in result:
        result["additionalProperties"] = False
        properties = result.get("properties")
        if isinstance(properties, dict):
            result["properties"] = {
                key: _enforce_strict_object_schema(value) for key, value in properties.items()
            }
        additional = result.get("additionalProperties")
        if isinstance(additional, dict):
            result["additionalProperties"] = _enforce_strict_object_schema(additional)

    if node_type == "array":
        items = result.get("items")
        if isinstance(items, dict):
            result["items"] = _enforce_strict_object_schema(items)
        elif isinstance(items, list):
            result["items"] = [_enforce_strict_object_schema(item) for item in items]

    for keyword in ("allOf", "anyOf", "oneOf"):
        variants = result.get(keyword)
        if isinstance(variants, list):
            result[keyword] = [_enforce_strict_object_schema(item) for item in variants]

    defs = result.get("$defs") or result.get("definitions")
    if isinstance(defs, dict):
        key = "$defs" if "$defs" in result else "definitions"
        result[key] = {
            name: _enforce_strict_object_schema(value) for name, value in defs.items()
        }

    return result
