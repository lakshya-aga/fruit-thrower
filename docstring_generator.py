"""
docstring_generator.py — Generate missing docstrings using Claude.

For units that have no docstring (or a stub), this module calls the
Anthropic API to produce a well-formatted NumPy/Google-style docstring
and patches the source file in place.
"""
import logging
import re
import ast
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Load .env from repo root if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

try:
    import openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

from parser import ParsedUnit


_SYSTEM_PROMPT_GENERATE = """\
You are an expert Python developer. Your job is to write clear, accurate docstrings.
Follow NumPy docstring style exactly:

\"\"\"
One-line summary of what the function/class does.

Extended description if needed (2-4 sentences). Explain the algorithm or key behaviour.

Parameters
----------
param_name : type
    Description.

Returns
-------
type
    Description of the return value.

Raises
------
ExceptionType
    When this is raised.

Examples
--------
>>> result = my_function(x=1)
\"\"\"

Return ONLY the docstring content between the triple quotes. No extra text.
"""

_SYSTEM_PROMPT_CONVERT = """\
You are an expert Python developer. Convert an existing docstring to NumPy style.

Output format:

\"\"\"
One-line summary (keep from original, improve wording only if unclear).

Extended description if present in the original (optional, omit if not needed).

Parameters
----------
param_name : type
    Description. One line per param. Infer types from the signature and source.

Returns
-------
type
    Description.

Raises
------
ExceptionType
    When this is raised. Include only if the code raises explicitly.

Examples
--------
>>> # Realistic minimal usage example inferred from the function signature and body.
>>> result = function_name(typical_args)
\"\"\"

Rules:
- Preserve the meaning of the original docstring exactly.
- Infer parameter types from the signature (type hints) and source code.
- The Examples section must be a runnable-looking snippet — not pseudocode.
- Return ONLY the docstring content between the triple quotes. No extra text.
"""


def _build_generate_prompt(unit: ParsedUnit) -> str:
    return f"""Write a docstring for this Python {unit.kind}:

File: {unit.file_path}
Module: {unit.module}

```python
{unit.source}
```

Return only the docstring text (without the surrounding triple quotes).
"""


def _build_convert_prompt(unit: ParsedUnit) -> str:
    return f"""Convert this existing docstring to NumPy style.

File: {unit.file_path}
Module: {unit.module}
Kind: {unit.kind}

Signature:
```python
{unit.signature}
```

Existing docstring:
{unit.docstring}

Source (for context — infer types and behaviour):
```python
{unit.source}
```

Return only the converted docstring text (without the surrounding triple quotes).
"""


def _strip_triple_quotes(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^"""', "", text).strip()
    text = re.sub(r'"""$', "", text).strip()
    return text.strip()


_CODEX_BIN = "/Applications/Codex.app/Contents/Resources/codex"


def _codex_available() -> bool:
    return Path(_CODEX_BIN).exists()


def _make_client(api_key: Optional[str] = None, openai_api_key: Optional[str] = None,
                 agent: Optional[str] = None):
    """
    Return (client, backend) where backend is 'codex', 'anthropic', or 'openai'.

    Priority:
      1. If agent='codex' (or FRUIT_CODE_AGENT=codex) and Codex binary exists → codex
      2. Anthropic if installed + key available
      3. OpenAI if installed + key available
    """
    import os
    effective_agent = agent or os.environ.get("FRUIT_CODE_AGENT", "")
    if effective_agent.lower() == "codex" or (not effective_agent and _codex_available()):
        if _codex_available():
            return None, "codex"

    ant_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    oai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

    if _ANTHROPIC_AVAILABLE and ant_key:
        import anthropic as _ant
        return _ant.Anthropic(api_key=ant_key), "anthropic"
    if _OPENAI_AVAILABLE and oai_key:
        import openai as _oai
        return _oai.OpenAI(api_key=oai_key), "openai"

    # Fall back to codex if binary exists and no API keys found
    if _codex_available():
        return None, "codex"

    if not _ANTHROPIC_AVAILABLE and not _OPENAI_AVAILABLE:
        raise ImportError("pip install anthropic  # or pip install openai")
    if _ANTHROPIC_AVAILABLE:
        raise ValueError("No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")
    raise ValueError("No API key found. Set OPENAI_API_KEY (or install anthropic + set ANTHROPIC_API_KEY).")


def _call_llm(client, backend: str, system: str, user: str) -> str:
    """Send a prompt to whichever backend client was returned by _make_client."""
    if backend == "codex":
        import subprocess
        prompt = f"{system}\n\n{user}"
        proc = subprocess.run(
            [_CODEX_BIN, "exec", prompt],
            capture_output=True, text=True, timeout=120,
        )
        output = proc.stdout.strip()
        if not output and proc.stderr:
            raise RuntimeError(f"Codex error: {proc.stderr.strip()[:300]}")
        return _strip_triple_quotes(output)
    elif backend == "anthropic":
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return _strip_triple_quotes(resp.content[0].text)
    else:  # openai
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return _strip_triple_quotes(resp.choices[0].message.content)


def generate_docstring(unit: ParsedUnit, client, backend: str = "anthropic") -> Optional[str]:
    """
    Generate a NumPy-style docstring for a ParsedUnit that lacks one.

    Parameters
    ----------
    unit : ParsedUnit
        The code unit needing a docstring.
    client : anthropic.Anthropic or openai.OpenAI
        Authenticated API client.
    backend : str
        Either 'anthropic' or 'openai'.

    Returns
    -------
    str or None
        Generated docstring text, or None on failure.

    Examples
    --------
    >>> client, backend = _make_client()
    >>> doc = generate_docstring(unit, client, backend)
    """
    return _call_llm(client, backend, _SYSTEM_PROMPT_GENERATE, _build_generate_prompt(unit))


def convert_docstring(unit: ParsedUnit, client, backend: str = "anthropic") -> Optional[str]:
    """
    Convert an existing docstring to NumPy style.

    Parameters
    ----------
    unit : ParsedUnit
        The code unit whose docstring should be converted.
    client : anthropic.Anthropic or openai.OpenAI
        Authenticated API client.
    backend : str
        Either 'anthropic' or 'openai'.

    Returns
    -------
    str or None
        Converted NumPy-style docstring text, or None on failure.

    Examples
    --------
    >>> client, backend = _make_client()
    >>> doc = convert_docstring(unit, client, backend)
    """
    return _call_llm(client, backend, _SYSTEM_PROMPT_CONVERT, _build_convert_prompt(unit))


def _find_ast_node(tree: ast.Module, unit: "ParsedUnit") -> ast.AST | None:
    """Locate the AST node for the given unit using context-aware traversal."""
    if unit.kind == "module":
        return tree

    if unit.kind == "class":
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == unit.name:
                return node
        return None

    # function or method
    fn_name = unit.name.split(".")[-1]  # "M2N.fit" → "fit", "my_func" → "my_func"

    if unit.parent:
        # Find the parent class, then locate the method in its direct body
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == unit.parent:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == fn_name:
                            return item
        return None
    else:
        # Top-level function: walk all nodes but exclude functions that are
        # direct children of a class (those are methods, handled above).
        # Use line_start as a tiebreaker when the same name appears multiple times.
        candidates = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == fn_name:
                    candidates.append(node)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        # Multiple definitions (e.g. in if/else blocks) — pick closest line
        return min(candidates, key=lambda n: abs(n.lineno - unit.line_start))


def patch_file_with_docstring(unit: ParsedUnit, docstring: str, repo_root: str) -> bool:
    """
    Insert or replace the docstring in the source file for the given unit.

    :param unit: (ParsedUnit) The unit to patch.
    :param docstring: (str) New docstring content (without triple quotes).
    :param repo_root: (str) Repository root to resolve file paths.
    :return: (bool) True if file was modified.
    """
    file_path = Path(repo_root) / unit.file_path
    try:
        source = file_path.read_text(encoding="utf-8")
    except OSError:
        logger.error("Failed to read %s for docstring patching", file_path, exc_info=True)
        return False
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    target = _find_ast_node(tree, unit)

    if target is None:
        return False

    # Find insertion point: line after def/class line (after colon)
    body = target.body if hasattr(target, "body") else []
    if not body:
        return False

    first_stmt = body[0]
    # If there's already a docstring, replace it
    if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
        # Replace existing docstring
        start = first_stmt.lineno - 1
        end = first_stmt.end_lineno
        indent = " " * (first_stmt.col_offset)
        new_doc_lines = f'{indent}"""\n'
        for line in docstring.splitlines():
            new_doc_lines += f"{indent}{line}\n" if line.strip() else "\n"
        new_doc_lines += f'{indent}"""\n'
        lines[start:end] = [new_doc_lines]
    else:
        # Insert before first statement
        insert_at = first_stmt.lineno - 1
        indent = " " * (first_stmt.col_offset)
        new_doc_lines = f'{indent}"""\n'
        for line in docstring.splitlines():
            new_doc_lines += f"{indent}{line}\n" if line.strip() else "\n"
        new_doc_lines += f'{indent}"""\n'
        lines.insert(insert_at, new_doc_lines)

    try:
        file_path.write_text("".join(lines), encoding="utf-8")
    except OSError:
        logger.error("Failed to write patched docstring to %s", file_path, exc_info=True)
        return False
    return True


def generate_missing_docstrings(
    units: list[ParsedUnit],
    repo_root: str,
    api_key: Optional[str] = None,
    dry_run: bool = False,
    agent: Optional[str] = None,
) -> dict[str, str]:
    """
    Generate docstrings for all units that are missing one, and patch source files.

    Parameters
    ----------
    units : list[ParsedUnit]
        All parsed units.
    repo_root : str
        Path to repository root.
    api_key : str or None
        Anthropic API key (or set ANTHROPIC_API_KEY env var).
    dry_run : bool
        If True, return generated docstrings without writing files.

    Returns
    -------
    dict
        Mapping of unit.id → generated docstring.

    Examples
    --------
    >>> results = generate_missing_docstrings(units, repo_root="/path/to/repo")
    """
    client, backend = _make_client(api_key=api_key, agent=agent)
    print(f"Using backend: {backend}")

    results = {}
    missing = [u for u in units if not u.docstring and u.kind != "module"]
    print(f"Generating docstrings for {len(missing)} units...")

    for unit in missing:
        print(f"  → {unit.module}.{unit.name}")
        try:
            doc = generate_docstring(unit, client, backend)
            results[unit.id] = doc
            if not dry_run:
                patched = patch_file_with_docstring(unit, doc, repo_root)
                status = "patched" if patched else "failed"
                print(f"    {status}")
        except Exception as e:
            print(f"    ERROR: {e}")

    return results


def _is_sphinx_style(docstring: str) -> bool:
    """Return True if docstring uses Sphinx :param:/:return: RST style."""
    return bool(re.search(r":param\s+\w+:", docstring) or re.search(r":return[s]?:", docstring))


def convert_all_docstrings(
    units: list[ParsedUnit],
    repo_root: str,
    api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    dry_run: bool = False,
    module_filter: Optional[str] = None,
    force: bool = False,
    agent: Optional[str] = None,
) -> dict[str, str]:
    """
    Convert existing Sphinx-style docstrings to NumPy style across a repo.

    Parameters
    ----------
    units : list[ParsedUnit]
        All parsed units.
    repo_root : str
        Path to repository root.
    api_key : str or None
        Anthropic API key (or set ANTHROPIC_API_KEY env var).
    dry_run : bool
        If True, print what would change without writing files.
    module_filter : str or None
        Only convert units whose module contains this string.
    force : bool
        If True, convert all docstrings, not just Sphinx-style ones.

    Returns
    -------
    dict
        Mapping of unit.id → converted docstring.

    Examples
    --------
    >>> results = convert_all_docstrings(units, repo_root="./fin-kit", dry_run=True)
    """
    candidates = [
        u for u in units
        if u.docstring
        and u.kind != "module"
        and (not module_filter or module_filter in u.module)
        and (force or _is_sphinx_style(u.docstring))
    ]

    print(f"Converting {len(candidates)} docstrings to NumPy style...")
    if dry_run:
        for u in candidates:
            print(f"  [dry-run] {u.kind:8s} {u.module}.{u.name} (line {u.line_start})")
        return {}

    client, backend = _make_client(api_key=api_key, openai_api_key=openai_api_key, agent=agent)
    print(f"Using backend: {backend}")

    # Generate all docstrings first, then patch each file bottom-to-top so
    # earlier patches don't shift the line numbers of later ones in the same file.
    generated: dict[str, tuple[ParsedUnit, str]] = {}  # unit.id → (unit, doc)
    for unit in candidates:
        print(f"  → {unit.module}.{unit.name}")
        try:
            doc = convert_docstring(unit, client, backend)
            generated[unit.id] = (unit, doc)
        except Exception as e:
            print(f"    ERROR generating: {e}")

    # Group by file, patch bottom-to-top within each file
    from collections import defaultdict
    by_file: dict[str, list[tuple[ParsedUnit, str]]] = defaultdict(list)
    for uid, (unit, doc) in generated.items():
        by_file[unit.file_path].append((unit, doc))

    results = {}
    for file_path, file_units in by_file.items():
        # Reverse order so patches at lower lines don't shift patches at higher lines
        for unit, doc in sorted(file_units, key=lambda x: x[0].line_start, reverse=True):
            try:
                patched = patch_file_with_docstring(unit, doc, repo_root)
                status = "patched" if patched else "failed"
                print(f"  {status}: {unit.module}.{unit.name}")
                if patched:
                    results[unit.id] = doc
            except Exception as e:
                print(f"  ERROR patching {unit.module}.{unit.name}: {e}")

    return results
