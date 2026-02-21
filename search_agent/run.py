"""
Unified entry-point for BrowseComp-Plus search agent.

Dispatches to the correct conversation / persist / format functions based
on the ``--model`` name (or the explicit ``--backend`` flag):

  - Models matching *liquid* or *lfm* (case-insensitive) use the LFM
    text-based tool-call protocol.
  - Everything else uses the standard vLLM Chat Completions tool-calling
    protocol.

Usage:
    python search_agent/run.py --model Qwen/Qwen3-32B --searcher-type bm25 ...
    python search_agent/run.py --model LFM-40B --searcher-type bm25 ...
    python search_agent/run.py --model my-model --backend lfm --searcher-type bm25 ...
"""

from __future__ import annotations

import argparse
import atexit
import json as _json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import openai
from rich import print as rprint

from prompts import format_messages
from vllm_client import (
    SearchToolHandler,
    _build_extra_kwargs,
    process_tsv_dataset,
    run_conversation_with_tools,
    persist_response,
)
from lfm_client import (
    run_lfm_conversation_with_tools,
    persist_lfm_response,
)
from searcher.searchers import SearcherType


def is_local_lfm_checkpoint(model_path: str) -> bool:
    """
    Check if a local model checkpoint is an LFM by looking for 'lfm' in JSON config files.

    Args:
        model_path: Path to local model checkpoint directory

    Returns:
        True if 'lfm' is found (case-insensitive) in any JSON file, False otherwise
    """
    import glob

    if not os.path.isdir(model_path):
        return False

    json_files = glob.glob(os.path.join(model_path, "*.json"))
    if not json_files:
        return False

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                content = f.read()
                if re.search(r"lfm", content, re.IGNORECASE):
                    return True
        except Exception:
            continue

    return False


def is_hf_lfm_model(model_path: str) -> bool:
    """
    Check if a HuggingFace model is an LFM by inspecting its config.json.

    Checks if 'lfm' appears in either:
    - model_type field
    - any entry in architectures list

    Args:
        model_path: HuggingFace model ID (e.g., 'LiquidAI/jimmy_8B_moe_4A32E_midtrain')

    Returns:
        True if the model is detected as an LFM, False otherwise
    """
    try:
        import json

        from huggingface_hub import hf_hub_download

        hf_token = os.environ.get("HF_TOKEN", None)

        # If HF_TOKEN not set, try reading from default cache location
        if not hf_token:
            token_path = os.path.expanduser("~/.cache/huggingface/token")
            if os.path.exists(token_path):
                with open(token_path) as f:
                    hf_token = f.read().strip() or None

        config_path = hf_hub_download(
            repo_id=model_path, filename="config.json", token=hf_token
        )
        with open(config_path) as f:
            config = json.load(f)

        model_type = config.get("model_type", "")
        architectures = config.get("architectures", [])

        # Check if model_type contains 'lfm' (case-insensitive)
        if re.search(r"lfm", model_type, re.IGNORECASE):
            return True

        # Check if any architecture contains 'lfm' (case-insensitive)
        for arch in architectures:
            if re.search(r"lfm", arch, re.IGNORECASE):
                return True

        return False
    except Exception:
        return False


def is_lfm_model(model_path: str) -> bool:
    """
    Check if a model is an LFM, supporting both local checkpoints and HuggingFace models.

    Detection order:
    1. If model_path is a local directory, check JSON config files for 'lfm'
    2. If model_path looks like a HuggingFace ID (contains '/'), download and check config.json

    Args:
        model_path: Local path or HuggingFace model ID

    Returns:
        True if the model is detected as an LFM, False otherwise
    """
    # Check if it's a local directory first
    if os.path.isdir(model_path):
        return is_local_lfm_checkpoint(model_path)

    # Check if it looks like a HuggingFace model ID (org/model format)
    if "/" in model_path:
        return is_hf_lfm_model(model_path)

    return False

# Serving Name
def get_served_name(model_path: str) -> str:
    """
    Get the served name for a model.
    """
    if is_lfm_model(model_path):
        return "Liquid-API"
    return model_path.split("/")[-1]


# ── Argument parsing ─────────────────────────────────────────


def build_arg_parser(
    description: str = "Unified BrowseComp-Plus search agent (auto-detects LFM vs vLLM).",
    default_model: str | None = None,
    default_output_dir: str = "runs",
) -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--query",
        default="topics-qrels/queries.tsv",
        help="User query *or* TSV file path",
    )


    # required for all models (LFM or vLLM)
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the model",
    )

    # serve model name
    parser.add_argument(
        "--model",
        default="Liquid-API",
        help="Model name served by vLLM, if it's LFM, it would be Liquid-API",
    )

    parser.add_argument(
        "--tool-call-parser",
        default="pythonic",
        help="Tool call parser to use - must be specified for Non-LFM models",
    )


    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10000,
        help="max_tokens for Chat Completions API",
    )
    parser.add_argument(
        "--output-dir",
        default=default_output_dir,
        help="Directory to store run JSON files",
    )
    parser.add_argument(
        "--query-template",
        choices=[
            "QUERY_TEMPLATE",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION",
        ],
        default="QUERY_TEMPLATE_NO_GET_DOCUMENT",
        help="Specify the query template to use",
    )
    parser.add_argument(
        "--num-threads", type=int, default=1, help="Parallel threads for dataset mode"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Max conversation rounds with function calls",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--model-url",
        default="http://localhost:8000/v1",
        help="vLLM server URL",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=None, help="Top-p nucleus sampling"
    )

    # Searcher selection and shared tool options
    parser.add_argument(
        "--searcher-type",
        choices=SearcherType.get_choices(),
        default="faiss",
        help=f"Type of searcher to use: {', '.join(SearcherType.get_choices())}",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face token for accessing private datasets/models",
    )
    parser.add_argument(
        "--hf-home", type=str, help="Hugging Face home directory for caching"
    )
    parser.add_argument(
        "--snippet-max-tokens",
        type=int,
        default=512,
        help="Tokens per document snippet",
    )
    parser.add_argument(
        "--k", type=int, default=5, help="Top-k search results to return for each query"
    )
    parser.add_argument(
        "--get-document",
        action="store_true",
        help="Also register the get_document tool",
    )

    # ── vLLM serving options ──────────────────────────────────
    parser.add_argument(
        "--port", type=int, default=0,
        help="Port for vLLM server (0 = auto-detect an available port)",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=None,
        help="Number of GPUs for vLLM tensor parallelism (default: all GPUs minus --searcher-gpu)",
    )
    parser.add_argument(
        "--searcher-gpu", type=int, default=1,
        help="GPU index reserved for the FAISS embedding model (ignored for bm25)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=32768,
        help="Maximum model context length for vLLM",
    )
    parser.add_argument(
        "--max-num-seqs", type=int, default=512,
        help="Maximum number of concurrent sequences for vLLM",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.8,
        help="GPU memory utilization fraction for vLLM",
    )
    parser.add_argument(
        "--dtype", default="bfloat16",
        help="Data type for vLLM model weights",
    )

    return parser


def parse_args():
    """Build the CLI parser, register searcher-specific flags, and parse."""
    parser = build_arg_parser()

    temp_args, _ = parser.parse_known_args()
    searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
    searcher_class.parse_args(parser)

    parsed_args = parser.parse_args()
    parsed_args.model = get_served_name(parsed_args.model_path)
    return parsed_args


# ── vLLM serving ─────────────────────────────────────────────


def _find_available_port(base: int = 8600, search_range: int = 300) -> int:
    """Find an available TCP port starting from *base*."""
    for offset in range(search_range):
        port = base + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"No available port found in range {base}–{base + search_range - 1}"
    )


def _detect_num_gpus() -> int:
    """Return the number of visible CUDA devices, falling back to 1."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        return len([d for d in visible.split(",") if d.strip()])
    try:
        import torch
        return torch.cuda.device_count() or 1
    except Exception:
        return 1


def _detect_expert_parallel(model_path: str) -> bool:
    """Return True if the model config contains any key with 'expert'."""
    try:
        from transformers import AutoConfig

        hf_token = os.environ.get("HF_TOKEN") or None
        cfg = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True, token=hf_token,
        )
        return any(
            re.search(r"expert", k, re.IGNORECASE) for k in cfg.to_dict()
        )
    except Exception:
        return False


def _wait_for_server(port: int, proc: subprocess.Popen, timeout: int = 900) -> bool:
    """Poll until the vLLM server responds on *port*.

    Returns True if the server became ready, False on timeout or process death.
    """
    import urllib.request
    import urllib.error

    url = f"http://127.0.0.1:{port}/v1/models"
    elapsed = 0
    interval = 2
    print(f"Waiting for vLLM server on port {port} (timeout {timeout}s)...")
    while elapsed < timeout:
        if proc.poll() is not None:
            print(
                f"ERROR: vLLM process (PID {proc.pid}) died during startup "
                f"(exit code {proc.returncode})"
            )
            return False
        try:
            urllib.request.urlopen(url, timeout=2)
            print(f"vLLM ready after ~{elapsed}s")
            return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(interval)
        elapsed += interval

    print(f"ERROR: vLLM did not become ready within {timeout}s")
    return False


def _smoke_test(port: int, model_name: str) -> None:
    """Send a trivial chat completion to verify the server works."""
    import urllib.request

    print("Smoke test...")
    payload = _json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 10,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = _json.loads(resp.read())
            print(_json.dumps(body, indent=2)[:500])
    except Exception as exc:
        print(f"Smoke test failed (non-fatal): {exc}")
    print()


def vllm_serving(args) -> bool:
    """Launch a vLLM server for ``args.model_path`` and wait until it is ready.

    Sets ``args.model_url`` to the running server's URL when successful.
    Registers an atexit handler to shut the server down on exit.

    Returns True if the server is up and ready, False on error or timeout.
    """
    # ── Port ──────────────────────────────────────────────────
    try:
        port = args.port or _find_available_port()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return False
    print(f"Using port: {port}")

    # ── GPUs / tensor parallel ────────────────────────────────
    total_gpus = _detect_num_gpus()

    # Reserve one GPU for the FAISS embedding model when using faiss searcher
    if args.searcher_type == "faiss":
        searcher_gpu = args.searcher_gpu
        vllm_gpu_indices = [i for i in range(total_gpus) if i != searcher_gpu]
        print(f"GPU allocation: GPU {searcher_gpu} → searcher, GPUs {vllm_gpu_indices} → vLLM")
    else:
        vllm_gpu_indices = list(range(total_gpus))

    num_gpus = args.num_gpus or len(vllm_gpu_indices)

    # ── Expert parallel ───────────────────────────────────────
    use_expert_parallel = _detect_expert_parallel(args.model_path)
    print(f"Expert parallel: {'enabled' if use_expert_parallel else 'disabled'}")

    # ── LFM doesn't need vLLM-side tool-call parsing ─────────
    use_lfm = is_lfm_model(args.model_path)

    # ── Build command ─────────────────────────────────────────
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model_path,
        "--served-model-name", args.model,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dtype", args.dtype,
        "--trust-remote-code",
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]

    cmd += ["--tensor-parallel-size", str(2)]

    if use_expert_parallel:
        cmd += ["--enable-expert-parallel"]


    cmd += [
        "--tool-call-parser", args.tool_call_parser,
        "--enable-auto-tool-choice",
    ]

    vllm_env = os.environ.copy()
    vllm_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in vllm_gpu_indices)

    print(f"Launching vLLM: CUDA_VISIBLE_DEVICES={vllm_env['CUDA_VISIBLE_DEVICES']} {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(cmd, env=vllm_env)
    except Exception as exc:
        print(f"ERROR: failed to launch vLLM process: {exc}")
        return False

    # ── Cleanup on exit ───────────────────────────────────────
    def _shutdown():
        if proc.poll() is None:
            print(f"Stopping vLLM server (PID {proc.pid})...")
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            print("Server stopped.")

    atexit.register(_shutdown)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    # ── Wait (15 min timeout) ─────────────────────────────────
    if not _wait_for_server(port, proc, timeout=1800):
        _shutdown()
        return False, port

    # ── Smoke test ────────────────────────────────────────────
    _smoke_test(port, args.model)

    # Update args so run() connects to the right URL
    args.model_url = f"http://localhost:{port}/v1"

    return True, port


# ── Run ──────────────────────────────────────────────────────


def run(args):
    """Run the search agent with the given arguments.

    Automatically selects the LFM or vLLM backend based on ``args.model_path``.
    """
    use_lfm = is_lfm_model(args.model_path)

    if use_lfm:
        conversation_fn = run_lfm_conversation_with_tools
        persist_fn = persist_lfm_response
        format_fn = partial(format_messages, if_lfm=True)
    else:
        conversation_fn = run_conversation_with_tools
        persist_fn = persist_response
        format_fn = partial(format_messages, if_lfm=False)

    backend_label = "LFM (text-based tool calls)" if use_lfm else "vLLM (native tool calls)"
    print(f"Backend: {backend_label}")

    # ── Environment ───────────────────────────────────────────
    if getattr(args, "hf_token", None):
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
    if getattr(args, "hf_home", None):
        os.environ["HF_HOME"] = args.hf_home

    # ── Client & tools ────────────────────────────────────────
    client = openai.OpenAI(base_url=args.model_url, api_key="EMPTY")

    searcher_class = SearcherType.get_searcher_class(args.searcher_type)
    searcher = searcher_class(args)

    tool_handler = SearchToolHandler(
        searcher=searcher,
        snippet_max_tokens=args.snippet_max_tokens,
        k=args.k,
        include_get_document=args.get_document,
    )

    tools_registered = ["search"]
    if args.get_document:
        tools_registered.append("get_document")

    print(
        f"Search agent started with {searcher.search_type} search "
        f"(snippet_max_tokens={args.snippet_max_tokens}, k={args.k})"
    )
    print(f"Registered tools: {', '.join(tools_registered)}")

    # ── TSV dataset mode ──────────────────────────────────────
    if isinstance(args.query, str):
        qstr = args.query.strip()
        if qstr.lower().endswith(".tsv"):
            potential_path = Path(qstr)
            try:
                if potential_path.is_file():
                    print("Processing TSV dataset", potential_path)
                    process_tsv_dataset(
                        str(potential_path),
                        client,
                        args,
                        tool_handler,
                        conversation_fn=conversation_fn,
                        persist_fn=persist_fn,
                        format_messages_fn=format_fn,
                    )
                    return
            except OSError:
                pass

    # ── Single query mode ─────────────────────────────────────
    print("Processing single query", args.query)
    messages = format_fn(args.query, args.query_template)

    extra_kwargs = _build_extra_kwargs(args)

    messages, tool_usage, status = conversation_fn(
        client,
        messages,
        args.model,
        tool_handler.get_tool_definitions(),
        tool_handler,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        **extra_kwargs,
    )

    persist_fn(
        args.output_dir, args.model, messages, tool_usage, status,
        query_id=None, model_path=args.model_path,
    )

    rprint(messages)


if __name__ == "__main__":
    arguments = parse_args()    
    success, port = vllm_serving(arguments)
    if not success:
        sys.exit(1)
    arguments.model_url = f"http://localhost:{port}/v1"
    run(arguments)
