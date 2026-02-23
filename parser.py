"""
parser.py — AST-based Python code parser.

Extracts functions, classes, and module-level docstrings from .py files,
producing structured metadata records ready for embedding and storage.
"""
import ast
import os
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


@dataclass
class ParsedUnit:
    """A single parseable unit: function, class, or module."""
    id: str                          # sha256 of file_path + name
    kind: str                        # "function" | "class" | "method" | "module"
    name: str                        # qualified name, e.g. MyClass.my_method
    file_path: str                   # relative path from repo root
    module: str                      # dotted module name
    line_start: int
    line_end: int
    signature: str                   # def foo(x, y) -> str
    docstring: Optional[str]         # raw docstring or None
    source: str                      # full source of the unit
    parent: Optional[str] = None     # class name if this is a method
    decorators: list[str] = field(default_factory=list)

    def to_embed_text(self) -> str:
        """Produce the text that gets embedded for semantic search."""
        parts = [f"[{self.kind.upper()}] {self.module}.{self.name}"]
        parts.append(f"Signature: {self.signature}")
        if self.docstring:
            parts.append(f"Docstring:\n{self.docstring.strip()}")
        parts.append(f"Source:\n{self.source}")
        return "\n\n".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)


def _make_id(file_path: str, name: str) -> str:
    raw = f"{file_path}::{name}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _get_source_segment(source_lines: list[str], node: ast.AST) -> str:
    start = node.lineno - 1
    end = node.end_lineno
    return "".join(source_lines[start:end])


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef, source: str) -> str:
    """Reconstruct a clean def signature (without the body)."""
    lines = source.splitlines()
    sig_lines = []
    for line in lines:
        sig_lines.append(line)
        stripped = line.rstrip()
        if stripped.endswith(":"):
            break
    return "\n".join(sig_lines)


def _path_to_module(file_path: str, root: str) -> str:
    rel = os.path.relpath(file_path, root)
    return rel.replace(os.sep, ".").removesuffix(".py")


def parse_file(file_path: str, repo_root: str) -> list[ParsedUnit]:
    """Parse a single Python file and return all extracted units."""
    units: list[ParsedUnit] = []
    path = Path(file_path)

    try:
        source = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Could not read {file_path}: {e}")
        return units

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"[WARN] Syntax error in {file_path}: {e}")
        return units

    source_lines = source.splitlines(keepends=True)
    module_name = _path_to_module(file_path, repo_root)
    rel_path = os.path.relpath(file_path, repo_root)

    # ── Module-level docstring ──────────────────────────────────────────────
    module_doc = ast.get_docstring(tree)
    if module_doc:
        units.append(ParsedUnit(
            id=_make_id(rel_path, "__module__"),
            kind="module",
            name="__module__",
            file_path=rel_path,
            module=module_name,
            line_start=1,
            line_end=len(source_lines),
            signature=f"module {module_name}",
            docstring=module_doc,
            source=source[:500],  # first 500 chars as context
        ))

    # ── Walk top-level and class-level nodes ───────────────────────────────
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_doc = ast.get_docstring(node)
            class_src = _get_source_segment(source_lines, node)
            decorators = [ast.unparse(d) for d in node.decorator_list]
            units.append(ParsedUnit(
                id=_make_id(rel_path, node.name),
                kind="class",
                name=node.name,
                file_path=rel_path,
                module=module_name,
                line_start=node.lineno,
                line_end=node.end_lineno,
                signature=f"class {node.name}({', '.join(ast.unparse(b) for b in node.bases)})",
                docstring=class_doc,
                source=class_src,
                decorators=decorators,
            ))
            # Methods inside this class
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _add_function(
                        units, item, source_lines, rel_path, module_name,
                        parent=node.name
                    )

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Only top-level functions (parent handled above)
            if not any(
                isinstance(n, ast.ClassDef) and any(
                    isinstance(child, type(node)) and child is node
                    for child in ast.walk(n)
                )
                for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef)
            ):
                _add_function(units, node, source_lines, rel_path, module_name)

    return units


def _add_function(
    units: list[ParsedUnit],
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    source_lines: list[str],
    rel_path: str,
    module_name: str,
    parent: Optional[str] = None,
):
    fn_src = _get_source_segment(source_lines, node)
    qualified = f"{parent}.{node.name}" if parent else node.name
    decorators = [ast.unparse(d) for d in node.decorator_list]
    kind = "method" if parent else "function"
    units.append(ParsedUnit(
        id=_make_id(rel_path, qualified),
        kind=kind,
        name=qualified,
        file_path=rel_path,
        module=module_name,
        line_start=node.lineno,
        line_end=node.end_lineno,
        signature=_signature(node, fn_src),
        docstring=ast.get_docstring(node),
        source=fn_src,
        parent=parent,
        decorators=decorators,
    ))


def parse_repository(repo_root: str, exclude_dirs: set[str] | None = None) -> list[ParsedUnit]:
    """
    Recursively parse all .py files in a repository.

    :param repo_root: (str) Absolute or relative path to the repository root.
    :param exclude_dirs: (set) Directory names to skip (e.g. {'tests', '.venv'}).
    :return: (list[ParsedUnit]) All extracted code units.
    """
    exclude_dirs = exclude_dirs or {"__pycache__", ".venv", "venv", "node_modules", ".git", "dist", "build"}
    all_units: list[ParsedUnit] = []

    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for fname in filenames:
            if fname.endswith(".py"):
                full_path = os.path.join(dirpath, fname)
                units = parse_file(full_path, repo_root)
                all_units.extend(units)
                print(f"  Parsed {os.path.relpath(full_path, repo_root)}: {len(units)} units")

    return all_units
