#!/usr/bin/env python
"""Render docs/config-reference.md from the pydantic section models (ledger DOC-002).

    PYTHONPATH=src python tools/docs/render_config_reference.py            # rewrite the reference
    PYTHONPATH=src python tools/docs/render_config_reference.py --check    # exit 1 if it is stale

Every section of ``gmd_active_learning.config.SECTIONS`` is walked recursively: nested models
become dotted keys, each leaf gets its type, default, constraints and the field description.
PHAL's hand-written reference omitted whole sections (candidate_pool, hotspot, model_registry);
a generated one cannot.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
import typing
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Sequence

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "docs" / "config-reference.md"
SECTION_YAML_DIR = REPO / "configs" / "sections"


def _is_model(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _nested_model(annotation: Any) -> type[BaseModel] | None:
    """The model of ``Model`` or ``Model | None``; None for anything else."""

    if _is_model(annotation):
        return annotation
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        models = [arg for arg in typing.get_args(annotation) if _is_model(arg)]
        others = [arg for arg in typing.get_args(annotation) if not _is_model(arg) and arg is not type(None)]
        if len(models) == 1 and not others:
            return models[0]
    return None


def type_name(annotation: Any) -> str:
    if annotation is None or annotation is type(None):
        return "null"
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Literal:
        return " \\| ".join(json.dumps(arg.value if isinstance(arg, Enum) else arg) for arg in args)
    if origin in (typing.Union, types.UnionType):
        return " \\| ".join(type_name(arg) for arg in args)
    if origin is typing.Annotated:
        return type_name(args[0])
    if origin is not None:
        name = getattr(origin, "__name__", str(origin))
        return f"{name}[{', '.join(type_name(arg) for arg in args)}]" if args else name
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return " \\| ".join(json.dumps(member.value) for member in annotation)
    return getattr(annotation, "__name__", str(annotation).replace("typing.", ""))


def _default(field: FieldInfo) -> Any:
    if field.default is not PydanticUndefined:
        return field.default
    if field.default_factory is not None:
        try:
            return field.default_factory()  # type: ignore[call-arg]
        except TypeError:
            return "<computed>"
    return PydanticUndefined


def _json(value: Any) -> str:
    if value is PydanticUndefined:
        return "**required**"
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, Enum):
        value = value.value
    try:
        text = json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        text = str(value)
    return f"`{text}`" if len(text) <= 80 else f"`{text[:77]}...`"


def _constraints(field: FieldInfo) -> str:
    parts = []
    for item in field.metadata:
        for attribute, symbol in (("gt", ">"), ("ge", ">="), ("lt", "<"), ("le", "<="), ("min_length", "len>="), ("max_length", "len<=")):
            value = getattr(item, attribute, None)
            if value is not None:
                parts.append(f"{symbol} {value}")
    return ", ".join(parts)


def _clean(text: str | None) -> str:
    return " ".join((text or "").split()).replace("|", "\\|")


def leaves(model: type[BaseModel], prefix: str) -> Iterator[tuple[str, FieldInfo, type[BaseModel] | None]]:
    """(dotted key, field, nested model or None) for every field, depth first."""

    for name, field in model.model_fields.items():
        key = f"{prefix}.{name}"
        nested = _nested_model(field.annotation)
        yield key, field, nested
        if nested is not None:
            yield from leaves(nested, key)


def yaml_comments(directory: Path = SECTION_YAML_DIR) -> dict[str, str]:
    """Inline ``# comments`` of configs/sections/*.yaml keyed by dotted path (fallback descriptions)."""

    comments: dict[str, str] = {}
    for path in sorted(directory.glob("*.yaml")):
        stack: list[tuple[int, str]] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            stripped = raw.lstrip()
            if not stripped or stripped.startswith("#") or stripped.startswith("- "):
                continue
            body, _, comment = raw.partition(" #")
            if ":" not in body:
                continue
            indent = len(raw) - len(stripped)
            key = body.strip().split(":", 1)[0].strip().strip("'\"")
            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack.append((indent, key))
            if comment.strip():
                comments[".".join(name for _, name in stack)] = comment.strip().lstrip("#").strip()
    return comments


def section_models() -> dict[str, type[BaseModel]]:
    from gmd_active_learning.config import SECTIONS, section_model

    return {name: section_model(name) for name in SECTIONS}


def render() -> str:
    from gmd_active_learning.config import REQUIRED_SECTIONS

    models = section_models()
    comments = yaml_comments()
    out = [
        "# Configuration reference",
        "",
        "<!-- GENERATED by tools/docs/render_config_reference.py from the pydantic section models; do not edit. -->",
        "",
        "One YAML file configures a run; each top-level key is a section validated by a strict pydantic model",
        "(`extra=\"forbid\"`: unknown sections and keys are errors). Relative paths in `md.executable`,",
        "`md.initial_structure` and `mlip.source_path` resolve against the config file's directory.",
        "Validate with `gmd-al validate-config --config run.yaml`; `configs/default.yaml` holds every key at its",
        "default and `configs/sections/*.yaml` document each section. Units: A, eV, eV/A, eV/A^3, fs.",
        "Descriptions come from the model's field descriptions, else from the inline comments of",
        "`configs/sections/<section>.yaml`; topic pages explain the semantics ([monitoring](monitoring.md),",
        "[extraction and datasets](extraction_and_datasets.md), [labeling and scheduling](labeling_and_scheduling.md),",
        "[SCF tooling](scf/README.md), [architecture](architecture.md)).",
        "",
        f"Required sections: {', '.join(f'`{name}`' for name in REQUIRED_SECTIONS)}. Every other section is optional and",
        "uses its defaults when absent.",
        "",
        "| Section | Model |",
        "|---|---|",
    ]
    out += [f"| [`{name}`](#{name}) | `{model.__module__}.{model.__qualname__}` |" for name, model in models.items()]
    for name, model in models.items():
        out += ["", f"## {name}", ""]
        doc = _clean((model.__doc__ or "").strip().split("\n\n")[0])
        if doc:
            out += [doc, ""]
        out += ["| Key | Type | Default | Constraints | Description |", "|---|---|---|---|---|"]
        for key, field, nested in leaves(model, name):
            if nested is not None:
                description = _clean(field.description) or _clean(comments.get(key)) or _clean((nested.__doc__ or "").strip().split("\n\n")[0])
                out.append(f"| `{key}` | section `{nested.__name__}` | | | {description} |")
                continue
            out.append(
                f"| `{key}` | {type_name(field.annotation)} | {_json(_default(field))} | {_constraints(field)} | {_clean(field.description) or _clean(comments.get(key))} |"
            )
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if current != text:
            print(f"{args.output} is out of date; run tools/docs/render_config_reference.py")
            return 1
        return 0
    args.output.write_text(text, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
