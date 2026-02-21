"""
vLLM Chat Completions building blocks for BrowseComp-Plus.

Provides the core conversation loop, persistence, tool handler, and TSV
dataset processing.  The unified CLI entry-point lives in ``run.py``.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import openai
from prompts import format_messages
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import extract_retrieved_docids_from_result


# ── Tool handler ─────────────────────────────────────────────


class SearchToolHandler:
    """Bridges between the Chat Completions tool-calling protocol and local searchers."""

    def __init__(
        self,
        searcher,
        snippet_max_tokens: int | None = None,
        k: int = 5,
        include_get_document: bool = True,
    ):
        self.searcher = searcher
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.include_get_document = include_get_document

        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    def execute_tool(self, tool_name: str, arguments: dict):
        if tool_name == "search":
            return self._search(arguments["query"])
        elif tool_name == "get_document":
            return self._get_document(arguments["docid"])
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def get_tool_definitions(self):
        """Return tools in Chat Completions format."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": self.searcher.search_description(self.k),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Query to search the local knowledge base for relevant information",
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

        if self.include_get_document:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "get_document",
                        "description": self.searcher.get_document_description(),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "docid": {
                                    "type": "string",
                                    "description": "Document ID to retrieve",
                                }
                            },
                            "required": ["docid"],
                        },
                    },
                }
            )

        return tools

    def _search(self, query: str):
        candidates = self.searcher.search(query, self.k)

        if self.snippet_max_tokens and self.snippet_max_tokens > 0 and self.tokenizer:
            for cand in candidates:
                text = cand["text"]
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > self.snippet_max_tokens:
                    cand["snippet"] = self.tokenizer.decode(
                        tokens[: self.snippet_max_tokens], skip_special_tokens=True
                    )
                else:
                    cand["snippet"] = text
        else:
            for cand in candidates:
                cand["snippet"] = cand["text"]

        results = []
        for cand in candidates:
            entry = {"docid": cand["docid"], "snippet": cand["snippet"]}
            if cand.get("score") is not None:
                entry["score"] = cand["score"]
            results.append(entry)

        return json.dumps(results, indent=2)

    def _get_document(self, docid: str):
        result = self.searcher.get_document(docid)
        if result is None:
            return json.dumps({"error": f"Document with docid '{docid}' not found"})
        return json.dumps(result, indent=2)


# ── Core conversation loop ───────────────────────────────────


def run_conversation_with_tools(
    client: openai.OpenAI,
    messages: list,
    model: str,
    tools: list,
    tool_handler: SearchToolHandler,
    max_tokens: int = 10000,
    max_iterations: int = 100,
    verbose: bool = False,
    **kwargs,
):
    """Run a multi-turn conversation with tool calling using Chat Completions API."""

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
                print(f"Error on iteration {iteration}: {e}")
            continue

        choice = response.choices[0]
        assistant_msg = choice.message

        # Build a serializable assistant message dict
        assistant_dict: dict = {"role": "assistant", "content": assistant_msg.content or ""}
        if assistant_msg.tool_calls:
            assistant_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in assistant_msg.tool_calls
            ]
        messages.append(assistant_dict)

        # If no tool calls, the model is done
        if not assistant_msg.tool_calls:
            status = "completed" if choice.finish_reason == "stop" else "incomplete"
            return messages, tool_usage, status

        # Execute each tool call and append results
        for tool_call in assistant_msg.tool_calls:
            try:
                arguments = json.loads(tool_call.function.arguments)
                result = tool_handler.execute_tool(tool_call.function.name, arguments)
                tool_usage[tool_call.function.name] = (
                    tool_usage.get(tool_call.function.name, 0) + 1
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
            except Exception as e:
                error_msg = f"Error executing {tool_call.function.name}: {str(e)}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": error_msg,
                    }
                )

    return messages, tool_usage, "incomplete"


# ── Persistence ──────────────────────────────────────────────


def persist_response(
    out_dir: str,
    model: str,
    messages: list,
    tool_usage: dict,
    status: str,
    *,
    query_id: str | None = None,
    model_path: str | None = None,
):
    """Persist results in the format expected by evaluate_run.py."""
    os.makedirs(out_dir, exist_ok=True)

    # Build tool output lookup: tool_call_id -> content
    tool_output_by_id: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            tool_output_by_id[msg.get("tool_call_id")] = msg.get("content")

    normalized_results: list[dict] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")

        if role == "assistant":
            # Handle tool calls
            for tc in msg.get("tool_calls", []):
                tc_id = tc["id"]
                tc_name = tc["function"]["name"]
                tc_args = tc["function"]["arguments"]
                normalized_results.append(
                    {
                        "type": "tool_call",
                        "tool_name": tc_name,
                        "arguments": tc_args,
                        "output": tool_output_by_id.get(tc_id),
                    }
                )

            # Handle final text output (only if no tool calls in this message)
            content = msg.get("content")
            if content and not msg.get("tool_calls"):
                normalized_results.append(
                    {
                        "type": "output_text",
                        "tool_name": None,
                        "arguments": None,
                        "output": content,
                    }
                )

    # Normalize tool counts
    normalized_tool_counts: dict[str, int] = {}
    for tool_name, count in (tool_usage or {}).items():
        normalized_name = (
            "search"
            if "retrieval" in tool_name.lower() or "search" in tool_name.lower()
            else tool_name
        )
        normalized_tool_counts[normalized_name] = (
            normalized_tool_counts.get(normalized_name, 0) + count
        )

    metadata = {
        "model": model,
        "output_dir": str(out_dir),
    }
    if model_path:
        metadata["model_path"] = model_path

    normalized_record = {
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
        json.dump(normalized_record, f, indent=2, default=str)

    print("Saved response to", filename, "| tool call counts:", normalized_tool_counts)


# ── Dataset processing ───────────────────────────────────────


def process_tsv_dataset(
    tsv_path: str,
    client: openai.OpenAI,
    args,
    tool_handler: SearchToolHandler,
    conversation_fn=None,
    persist_fn=None,
    format_messages_fn=None,
):
    """Process a TSV file of (id \\t query) pairs and persist responses.

    Parameters
    ----------
    conversation_fn : callable, optional
        Replacement for ``run_conversation_with_tools``.  Must accept the
        same positional/keyword arguments and return ``(messages, tool_usage, status)``.
    persist_fn : callable, optional
        Replacement for ``persist_response``.  Must accept the same
        positional/keyword arguments.
    format_messages_fn : callable, optional
        Replacement for ``format_messages``.  Must accept ``(query, query_template)``
        and return a message list.
    """
    if conversation_fn is None:
        conversation_fn = run_conversation_with_tools
    if persist_fn is None:
        persist_fn = persist_response
    if format_messages_fn is None:
        format_messages_fn = format_messages

    dataset_path = Path(tsv_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    out_dir = Path(args.output_dir).expanduser().resolve()

    queries = []
    with dataset_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            queries.append((row[0].strip(), row[1].strip()))

    processed_ids = set()
    if out_dir.exists():
        for json_path in out_dir.glob("run_*.json"):
            try:
                with json_path.open("r", encoding="utf-8") as jf:
                    meta = json.load(jf)
                    qid_saved = meta.get("query_id")
                    if qid_saved:
                        processed_ids.add(str(qid_saved))
            except Exception:
                continue

    remaining = [(qid, qtext) for qid, qtext in queries if qid not in processed_ids]

    print(
        f"Processing {len(remaining)} remaining queries "
        f"(skipping {len(processed_ids)}) from {dataset_path} ..."
    )

    completed_lock = threading.Lock()
    completed_count = [0]

    def _handle_single_query(qid: str, qtext: str, pbar=None):
        messages = format_messages_fn(qtext, args.query_template)

        extra_kwargs = _build_extra_kwargs(args)

        try:
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

            if status == "completed":
                with completed_lock:
                    completed_count[0] += 1
                    if pbar:
                        pbar.set_postfix(completed=completed_count[0])

            persist_fn(
                out_dir, args.model, messages, tool_usage, status,
                query_id=qid, model_path=getattr(args, "model_path", None),
            )

        except Exception as exc:
            print(f"[Error] Query id={qid} failed: {exc}")

    if args.num_threads <= 1:
        with tqdm(remaining, desc="Queries", unit="query") as pbar:
            for qid, qtext in pbar:
                _handle_single_query(qid, qtext, pbar)
    else:
        with (
            ThreadPoolExecutor(max_workers=args.num_threads) as executor,
            tqdm(total=len(remaining), desc="Queries", unit="query") as pbar,
        ):
            futures = [
                executor.submit(_handle_single_query, qid, qtext, pbar)
                for qid, qtext in remaining
            ]
            for _ in as_completed(futures):
                pbar.update(1)


# ── Helpers ──────────────────────────────────────────────────


def _build_extra_kwargs(args) -> dict:
    """Build extra kwargs for the Chat Completions API from parsed args."""
    extra: dict = {"temperature": 0.7, "top_p": 0.9}
    if getattr(args, "temperature", None) is not None:
        extra["temperature"] = args.temperature
    if getattr(args, "top_p", None) is not None:
        extra["top_p"] = args.top_p
    return extra
