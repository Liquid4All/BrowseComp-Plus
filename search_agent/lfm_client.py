"""
LFM (Liquid Foundation Model) building blocks for BrowseComp-Plus.

LFM-specific handling:
  - Text-based tool calls: <|tool_call_start|>[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]<|tool_call_end|>
  - Think blocks: <think>...</think>

The text-based parsing is adapted from search_evals/agents/llms/liquid_api.py.
The unified CLI entry-point lives in ``run.py``.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from datetime import datetime

import openai

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import extract_retrieved_docids_from_result

from vllm_client import SearchToolHandler


# ── LFM text-format parsing (from liquid_api.py) ────────────


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks."""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    test = text.split("</think>")[-1]
    test = test.strip()
    return test


def strip_tool_call_markers(text: str) -> str:
    """Remove <|tool_call_start|>...<|tool_call_end|> markers."""
    return re.sub(
        r"<\|tool_call_start\|>.*?<\|tool_call_end\|>", "", text, flags=re.DOTALL
    ).strip()


def clean_lfm_output(text: str) -> str:
    """Strip both think blocks and tool-call markers from LFM output."""
    text = strip_think_blocks(text)
    text = strip_tool_call_markers(text)
    return text.strip()


# ── AST helpers for pythonic tool-call parsing ───────────────


def _eval_ast_node(node: ast.AST):
    """Safely evaluate a limited subset of AST literal nodes."""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.List):
        return [_eval_ast_node(elt) for elt in node.elts]
    elif isinstance(node, ast.Tuple):
        return tuple(_eval_ast_node(elt) for elt in node.elts)
    elif isinstance(node, ast.Dict):
        return {
            _eval_ast_node(k): _eval_ast_node(v)
            for k, v in zip(node.keys, node.values)
        }
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_ast_node(node.operand)
    else:
        raise ValueError(f"Unsupported AST node: {ast.dump(node)}")


def _get_function_name(node: ast.expr) -> str:
    """Extract function name, handling dotted names like ``module.func``."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return _get_function_name(node.value) + "." + node.attr
    else:
        raise ValueError(f"Invalid function name node: {node}")


def extract_tool_calls_from_text(text: str) -> list[dict]:
    """
    Parse LFM text-format tool calls.

    Format::

        <|tool_call_start|>[func_name(arg1="val", arg2=123)]<|tool_call_end|>

    Returns a list of ``{"name": str, "arguments": dict}``.
    """
    if not text:
        return []

    calls: list[dict] = []
    pattern = r"<\|tool_call_start\|>\s*\[(.*?)\]\s*<\|tool_call_end\|>"
    matches = re.findall(pattern, text, flags=re.DOTALL)

    for match in matches:
        try:
            parsed = ast.parse(f"x = [{match}]").body[0].value.elts  # type: ignore[attr-defined]
            for call in parsed:
                if not isinstance(call, ast.Call):
                    continue
                func_name = _get_function_name(call.func)
                args = {kw.arg: _eval_ast_node(kw.value) for kw in call.keywords}
                calls.append({"name": func_name, "arguments": args})
        except Exception as e:
            print(f"Warning: failed to parse tool-call block: {e}")
            continue

    return calls


# ── LFM conversation loop ───────────────────────────────────


def run_lfm_conversation_with_tools(
    client: openai.OpenAI,
    messages: list,
    model: str,
    tools: list,
    tool_handler: SearchToolHandler,
    max_tokens: int = 10000,
    max_iterations: int = 100,
    verbose: bool = True,
    **kwargs,
):
    """
    Multi-turn conversation loop for LFM text-based tool calls.

    LFM emits tool calls as plain text in the form:
        <|tool_call_start|>[func(arg="val")]<|tool_call_end|>

    Returns ``(messages, tool_usage, status)`` — same contract as
    ``vllm_client.run_conversation_with_tools``.
    """

    tool_usage: dict[str, int] = {}

    for iteration in range(max_iterations):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as e:
            if verbose:
                print(f"\nError on iteration {iteration}: {e}\n")
            continue

        choice = response.choices[0]
        msg = choice.message
        raw_content = msg.content or ""
        raw_content = strip_think_blocks(raw_content)

        # ── Text-based tool calls ─────────────────────────────
        text_calls = extract_tool_calls_from_text(raw_content)

        if text_calls:
            # Keep the raw assistant content (model expects to see its own output)
            messages.append({"role": "assistant", "content": raw_content})

            # Execute tools and build LFM-format result list
            results_for_model: list[dict] = []
            for tc in text_calls:
                try:
                    result = tool_handler.execute_tool(tc["name"], tc["arguments"])
                    tool_usage[tc["name"]] = tool_usage.get(tc["name"], 0) + 1
                    try:
                        parsed_result = json.loads(result)
                    except (json.JSONDecodeError, TypeError):
                        parsed_result = result
                    results_for_model.append(
                        {"name": tc["name"], "result": parsed_result}
                    )
                except Exception as e:
                    results_for_model.append(
                        {"name": tc["name"], "result": f"Error: {e}"}
                    )

            # Tool results in LFM native format: [{"name": ..., "result": ...}]
            messages.append(
                {"role": "tool", "content": json.dumps(results_for_model)}
            )
            continue

        # ── No tool calls → final answer ──────────────────────
        clean_content = clean_lfm_output(raw_content)
        messages.append({"role": "assistant", "content": clean_content})

        status = "completed" if choice.finish_reason == "stop" else "incomplete"
        return messages, tool_usage, status

    return messages, tool_usage, "incomplete"


# ── LFM persistence ─────────────────────────────────────────


def persist_lfm_response(
    out_dir: str,
    model: str,
    messages: list,
    tool_usage: dict,
    status: str,
    *,
    query_id: str | None = None,
    model_path: str | None = None,
):
    """
    Persist results for the LFM client.

    Processes text-based LFM tool-call messages and cleans
    ``<think>``/``<|tool_call_*|>`` artefacts from the final output.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Tool messages have no tool_call_id; content is a JSON list of results.
    text_tool_result_queue: list[list[dict]] = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "tool":
            try:
                parsed = json.loads(m["content"])
                if isinstance(parsed, list):
                    text_tool_result_queue.append(parsed)
                else:
                    text_tool_result_queue.append([])
            except Exception:
                text_tool_result_queue.append([])

    text_result_idx = 0
    normalized_results: list[dict] = []

    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue

        content = m.get("content", "")

        # ── Text-based tool calls ─────────────────────────────
        text_calls = extract_tool_calls_from_text(content)
        if text_calls:
            results_list = (
                text_tool_result_queue[text_result_idx]
                if text_result_idx < len(text_tool_result_queue)
                else []
            )
            text_result_idx += 1

            for i, tc in enumerate(text_calls):
                output = None
                if i < len(results_list) and isinstance(results_list[i], dict):
                    output = json.dumps(results_list[i].get("result"))
                normalized_results.append(
                    {
                        "type": "tool_call",
                        "tool_name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                        "output": output,
                    }
                )
            continue

        # ── Final output text ─────────────────────────────────
        clean = clean_lfm_output(content)
        if clean:
            normalized_results.append(
                {
                    "type": "output_text",
                    "tool_name": None,
                    "arguments": None,
                    "output": clean,
                }
            )

    # ── Normalize tool counts ────────────────────────────────
    normalized_tool_counts: dict[str, int] = {}
    for tool_name, count in (tool_usage or {}).items():
        nn = (
            "search"
            if "retrieval" in tool_name.lower() or "search" in tool_name.lower()
            else tool_name
        )
        normalized_tool_counts[nn] = normalized_tool_counts.get(nn, 0) + count

    metadata = {"model": model, "output_dir": str(out_dir)}
    if model_path:
        metadata["model_path"] = model_path

    record = {
        "metadata": metadata,
        "query_id": query_id,
        "tool_call_counts": normalized_tool_counts,
        "status": status,
        "retrieved_docids": extract_retrieved_docids_from_result(normalized_results),
        "result": normalized_results,
    }

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    filename = os.path.join(str(out_dir), f"run_{ts}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)

    print("Saved response to", filename, "| tool call counts:", normalized_tool_counts)

