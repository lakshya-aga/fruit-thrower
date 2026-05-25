"""
code_agent.py — AI coding agent integration for generating new functions.

Backends
--------
codex  (default)
    Runs the Codex CLI binary non-interactively via `codex exec "<prompt>"`.
    Binary path: FRUIT_CODEX_BIN env var, else /Applications/Codex.app/Contents/Resources/codex.
    Model override: pass --model via FRUIT_CODEX_EXTRA_ARGS or the `model` param.

claude
    Calls the Anthropic Messages API directly.
    Auth (priority order): ANTHROPIC_OAUTH_TOKEN → ANTHROPIC_API_KEY.
    Model: FRUIT_CLAUDE_MODEL (default: claude-sonnet-4-6).

openai
    Calls the OpenAI Chat Completions API directly.
    Auth (priority order): OPENAI_OAUTH_TOKEN → OPENAI_API_KEY.
    Model: FRUIT_OPENAI_MODEL (default: gpt-4o).

mock
    Returns a deterministic stub — no API call. For offline testing.

Global selector: FRUIT_CODE_AGENT env var (default: codex).
"""

import logging
import os
import re
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert Python developer specialising in quantitative finance libraries.
Generate a well-documented Python function based on the user's request.

Rules:
- Production-quality Python with full type hints.
- Google-style docstring (Args / Returns / Raises / Example sections).
- Include all necessary imports at the top of the code block.
- The function must be self-contained and reusable as a library function.
- Match the style and conventions of any existing functions shown as context.

Output format: return ONLY the Python code wrapped in a single ```python ... ``` fence.
No prose, no explanations outside the fence.
"""


def _extract_code(text: str) -> str:
    """Pull the first ```python … ``` or ``` … ``` block from model output."""
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _build_user_message(request: str, context: str) -> str:
    if context:
        return (
            "## Related existing functions (for style / context)\n\n"
            f"{context}\n\n"
            "---\n\n"
            f"## Request\n\n{request}"
        )
    return request


def _anthropic_client(oauth_token: Optional[str], api_key: Optional[str]):
    """Return an anthropic.Anthropic client, preferring OAuth over API key."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    if oauth_token:
        # auth_token sends Authorization: Bearer <token> instead of x-api-key
        return anthropic.Anthropic(auth_token=oauth_token)
    if api_key:
        return anthropic.Anthropic(api_key=api_key)
    raise ValueError(
        "No Claude credentials found. Set ANTHROPIC_OAUTH_TOKEN (OAuth) "
        "or ANTHROPIC_API_KEY (API key)."
    )


def _openai_client(oauth_token: Optional[str], api_key: Optional[str]):
    """Return an openai.OpenAI client, preferring OAuth over API key."""
    try:
        import openai
    except ImportError:
        raise ImportError("pip install openai")

    if oauth_token:
        # OpenAI SDK accepts any bearer token via api_key; set auth header explicitly
        import httpx

        return openai.OpenAI(
            api_key="oauth",
            http_client=httpx.Client(
                headers={"Authorization": f"Bearer {oauth_token}"},
            ),
        )
    if api_key:
        return openai.OpenAI(api_key=api_key)
    raise ValueError(
        "No OpenAI credentials found. Set OPENAI_OAUTH_TOKEN (OAuth) "
        "or OPENAI_API_KEY (API key)."
    )


def _generate_claude(
    request: str,
    context: str,
    model: str,
    oauth_token: Optional[str],
    api_key: Optional[str],
) -> str:
    client = _anthropic_client(oauth_token, api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_message(request, context)}],
    )
    return _extract_code(msg.content[0].text)


def _generate_openai(
    request: str,
    context: str,
    model: str,
    oauth_token: Optional[str],
    api_key: Optional[str],
) -> str:
    client = _openai_client(oauth_token, api_key)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(request, context)},
        ],
    )
    return _extract_code(resp.choices[0].message.content)


_DEFAULT_CODEX_BIN = "/Applications/Codex.app/Contents/Resources/codex"


def _find_codex_bin() -> str:
    """Resolve the Codex CLI binary path."""
    from_env = os.environ.get("FRUIT_CODEX_BIN", "").strip()
    if from_env:
        return from_env
    # Try PATH first (e.g. if user symlinked it)
    in_path = shutil.which("codex")
    if in_path:
        return in_path
    return _DEFAULT_CODEX_BIN


def _generate_codex(
    request: str,
    context: str,
    repo_root: str,
    module_path: Optional[str],
    model_override: Optional[str],
) -> dict:
    """
    Run `codex exec "<prompt>"` non-interactively inside repo_root.

    Codex is agentic — it writes files itself. We don't write anything;
    we just re-index the repo afterwards.

    Returns dict with keys: output (str), wrote_files (bool)
    """
    bin_path = _find_codex_bin()
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(
            f"Codex binary not found at '{bin_path}'. "
            "Set FRUIT_CODEX_BIN to the correct path."
        )

    # Validate repo_root is a real directory
    if not os.path.isdir(repo_root):
        raise FileNotFoundError(f"repo_root '{repo_root}' is not a valid directory.")

    # Build prompt — include module_path as a placement hint
    placement = (
        f"\nWrite the function into `{module_path}` "
        f"(create the file if it doesn't exist).\n"
        if module_path
        else ""
    )
    prompt = _build_user_message(request + placement, context)

    cmd = [bin_path, "exec", "--sandbox", "workspace-write"]
    if model_override:
        cmd += ["--model", model_override]
    extra = os.environ.get("FRUIT_CODEX_EXTRA_ARGS", "").strip()
    if extra:
        cmd += extra.split()
    cmd.append(prompt)

    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = (proc.stdout or "").strip()
    if proc.returncode != 0:
        stderr = (proc.stderr or "")[-1000:]
        logger.error("codex exec failed (code %d): %s", proc.returncode, stderr)
        raise RuntimeError(
            f"codex exec exited with code {proc.returncode}.\nstderr: {stderr}"
        )

    # Detect whether Codex actually wrote files using git status.
    # -uall expands untracked directories into individual files so new
    # directories (e.g. mlfinlab/portfolio/) are listed as .py paths.
    git_proc = subprocess.run(
        ["git", "status", "--porcelain", "-uall"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    changed_files = [
        line[3:].strip()
        for line in (git_proc.stdout or "").splitlines()
        if line.strip() and line[3:].strip().endswith(".py")
    ]
    wrote_files = bool(changed_files)

    # Always try to extract a code block — fallback if Codex couldn't write
    extracted = _extract_code(output)

    return {
        "output": output,
        "extracted_code": extracted,
        "wrote_files": wrote_files,
        "changed_files": changed_files,
    }


def generate_function(
    request: str,
    context: str = "",
    agent: Optional[str] = None,
    model_override: Optional[str] = None,
    repo_root: str = ".",
    **kwargs,
) -> dict:
    """
    Call an AI coding agent and return a generated Python function.

    Args:
        request:        Natural language description of the desired function.
        context:        Related existing functions to pass as context.
        agent:          "codex", "claude", or "openai". Falls back to
                        FRUIT_CODE_AGENT env var, then defaults to "codex".
        model_override: Model ID override.
        repo_root:      Repo directory — Codex runs with this as cwd.

    Returns:
        dict with keys: code, agent_used, model_used, auth_method

    Raises:
        ValueError / FileNotFoundError / RuntimeError on failure.
    """
    backend = (agent or os.environ.get("FRUIT_CODE_AGENT", "codex")).lower()

    if backend == "codex":
        result = _generate_codex(
            request, context, repo_root, kwargs.get("module_path"), model_override
        )
        model = model_override or os.environ.get("FRUIT_CODEX_MODEL", "default")
        return {
            "code": result["extracted_code"] or result["output"],
            "raw_output": result["output"],
            "wrote_files": result["wrote_files"],
            "changed_files": result["changed_files"],
            "agent_used": "codex",
            "model_used": model,
            "auth_method": "codex_auth",
        }

    if backend == "mock":
        # Offline test mode — returns a deterministic stub without any API call.
        stub = (
            "import pandas as pd\n\n\n"
            "def mock_generated_function(series: pd.Series, window: int = 252) -> pd.Series:\n"
            '    """Mock function generated by fruit-thrower test mode.\n\n'
            "    Args:\n"
            "        series: Input time series.\n"
            "        window: Rolling window size.\n\n"
            "    Returns:\n"
            "        Rolling mean of the input series.\n"
            '    """\n'
            "    return series.rolling(window).mean()\n"
        )
        return {
            "code": stub,
            "agent_used": "mock",
            "model_used": "mock",
            "auth_method": "none",
        }

    if backend == "claude":
        oauth_token = os.environ.get("ANTHROPIC_OAUTH_TOKEN", "").strip() or None
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip() or None
        model = model_override or os.environ.get(
            "FRUIT_CLAUDE_MODEL", "claude-sonnet-4-6"
        )
        code = _generate_claude(request, context, model, oauth_token, api_key)
        auth_method = "oauth" if oauth_token else "api_key"
        return {
            "code": code,
            "agent_used": "claude",
            "model_used": model,
            "auth_method": auth_method,
        }

    if backend in ("openai", "codex", "gpt"):
        oauth_token = os.environ.get("OPENAI_OAUTH_TOKEN", "").strip() or None
        api_key = os.environ.get("OPENAI_API_KEY", "").strip() or None
        model = model_override or os.environ.get("FRUIT_OPENAI_MODEL", "gpt-4o")
        code = _generate_openai(request, context, model, oauth_token, api_key)
        auth_method = "oauth" if oauth_token else "api_key"
        return {
            "code": code,
            "agent_used": "openai",
            "model_used": model,
            "auth_method": auth_method,
        }

    raise ValueError(
        f"Unknown agent '{backend}'. Set FRUIT_CODE_AGENT to 'claude' or 'openai'."
    )
