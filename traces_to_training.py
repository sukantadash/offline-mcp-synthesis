"""Turn recorded agent traces into tool-calling training conversations.

Reads MLflow traces from a tracking store or from exported trace JSON files,
rebuilds each conversation, checks it, and writes JSONL records with
``messages`` (assistant ``tool_calls`` and ``tool`` results) and ``tools``.

A conversation is rebuilt from the last LLM span in the trace when there is one:
that span's input messages plus its response are exactly what the model saw and
returned. Traces without LLM spans are rebuilt from the root span (system
prompt, query, answer) and the TOOL spans in order. Either way, every tool
result in the conversation must match a recorded TOOL span, so the training
data only carries outputs that the tools actually returned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from jsonschema import ValidationError

from agent_traces import normalize_messages, tool_result_text
from offline_mcp_synthesis import DEFAULT_SCHEMA, ROOT, _write_jsonl, load_tools, validate_trajectory

DEFAULT_OUTPUT = ROOT / "output" / "traces_training.jsonl"
DEFAULT_TRACKING_URI = f"sqlite:///{ROOT / 'output' / 'mlflow.db'}"
DEFAULT_EXPERIMENT = "order-ops-stateful-env"
LLM_SPAN_TYPES = ("LLM", "CHAT_MODEL")


def load_traces(*, tracking_uri: str | None = None, experiment: str | None = None, trace_files: list[str] | None = None) -> list:
    """Load traces from exported JSON files, or from an MLflow experiment."""
    from mlflow.entities import Trace

    if trace_files:
        traces = []
        for path in trace_files:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            for item in data if isinstance(data, list) else [data]:
                traces.append(Trace.from_json(json.dumps(item)))
        return traces

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    found = mlflow.get_experiment_by_name(experiment)
    if found is None:
        raise ValueError(f"No MLflow experiment named {experiment!r} at {tracking_uri}")
    return mlflow.search_traces(experiment_ids=[found.experiment_id], return_type="list", flush=True)


def _spans(trace, *span_types: str) -> list:
    return sorted((span for span in trace.data.spans if span.span_type in span_types), key=lambda span: span.start_time_ns)


def _root(trace):
    roots = [span for span in trace.data.spans if span.parent_id is None]
    if len(roots) != 1:
        raise ValueError(f"expected one root span, found {len(roots)}")
    return roots[0]


def rebuild_from_llm_spans(trace) -> tuple[list[dict], list[dict] | None]:
    """Rebuild the conversation from the last model call: its input messages plus its response."""
    llm_spans = _spans(trace, *LLM_SPAN_TYPES)
    if not llm_spans:
        raise ValueError("trace has no LLM spans")
    last = llm_spans[-1]
    inputs, outputs = last.inputs or {}, last.outputs or {}
    raw = list(inputs["messages"]) + [outputs["choices"][0]["message"]]
    return normalize_messages(raw), inputs.get("tools")


def rebuild_from_tool_spans(trace) -> tuple[list[dict], None]:
    """Rebuild the conversation from the root span and the TOOL spans, one call per assistant turn."""
    root = _root(trace)
    inputs, outputs = root.inputs or {}, root.outputs or {}
    for key, value in (("system_prompt", inputs.get("system_prompt")), ("query", inputs.get("query")), ("answer", outputs.get("answer"))):
        if not value:
            raise ValueError(f"root span is missing {key}")
    messages = [{"role": "system", "content": inputs["system_prompt"]}, {"role": "user", "content": inputs["query"]}]
    for index, span in enumerate(_spans(trace, "TOOL"), start=1):
        call_id = span.get_attribute("tool_call_id") or f"call_{index}"
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": span.name, "arguments": json.dumps(span.inputs or {})}}],
        })
        messages.append({"role": "tool", "tool_call_id": call_id, "name": span.name, "content": tool_result_text(span.outputs)})
    messages.append({"role": "assistant", "content": outputs["answer"]})
    return messages, None


def tool_span_mismatches(messages: list[dict], trace) -> list[str]:
    """Compare every tool call and result in a conversation with the trace's TOOL spans."""
    spans = _spans(trace, "TOOL")
    if not spans:
        return ["trace has no TOOL spans to check tool results against"]
    calls = [call for message in messages if message["role"] == "assistant" for call in message.get("tool_calls", [])]
    results = {message["tool_call_id"]: message["content"] for message in messages if message["role"] == "tool"}
    problems = []
    if len(calls) != len(spans):
        problems.append(f"{len(calls)} tool calls in the conversation but {len(spans)} TOOL spans")
    for call, span in zip(calls, spans):
        name = call["function"]["name"]
        span_call_id = span.get_attribute("tool_call_id")
        if span_call_id and span_call_id != call["id"]:
            problems.append(f"{name}: call id {call['id']} does not match TOOL span id {span_call_id}")
        if span.name != name or (span.inputs or {}) != json.loads(call["function"]["arguments"]):
            problems.append(f"{name}: arguments differ from the TOOL span")
        if tool_result_text(span.outputs) != results.get(call["id"]):
            problems.append(f"{name}: result differs from the TOOL span output")
    return problems


def convert(traces: list, *, tools_fallback: list[dict], source: str = "auto") -> tuple[list[dict], list[dict]]:
    """Rebuild, check and deduplicate conversations; return accepted and rejected records."""
    accepted: list[dict] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    for trace in traces:
        trace_id = trace.info.trace_id
        use_llm = source == "llm-spans" or (source == "auto" and bool(_spans(trace, *LLM_SPAN_TYPES)))
        try:
            messages, tools = rebuild_from_llm_spans(trace) if use_llm else rebuild_from_tool_spans(trace)
            tools = tools or tools_fallback
            if not messages or messages[0]["role"] != "system":
                raise ValueError("conversation must start with a system message")
            record = validate_trajectory({"messages": messages}, tools, messages[0]["content"])
        except (ValueError, TypeError, KeyError, IndexError, ValidationError, json.JSONDecodeError) as error:
            rejected.append({"trace_id": trace_id, "stage": "structure", "error": str(error)})
            continue
        problems = tool_span_mismatches(record["messages"], trace)
        if problems:
            rejected.append({"trace_id": trace_id, "stage": "tool_spans", "error": "; ".join(problems), "messages": record["messages"]})
            continue
        key = hashlib.sha256(json.dumps(record["messages"], sort_keys=True).encode()).hexdigest()
        if key in seen:
            rejected.append({"trace_id": trace_id, "stage": "duplicate", "error": "same conversation as an earlier trace"})
            continue
        seen.add(key)
        accepted.append({
            "messages": record["messages"],
            "tools": tools,
            "source_question": record["source_question"],
            "trace_id": trace_id,
            "rebuilt_from": "llm_spans" if use_llm else "tool_spans",
            "mode": "recorded_traces",
        })
    return accepted, rejected


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert recorded agent traces into tool-calling training conversations.")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--trace-files", nargs="+", help="Exported trace JSON files to read instead of the tracking store")
    parser.add_argument("--export-dir", help="Also write each loaded trace to this directory as JSON")
    parser.add_argument("--source", choices=["auto", "llm-spans", "tool-spans"], default="auto",
                        help="Rebuild from LLM spans, from TOOL spans, or pick automatically")
    parser.add_argument("--tool-schema", default=str(DEFAULT_SCHEMA), help="Tool definitions for traces that do not record them")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Accepted rows JSONL")
    args = parser.parse_args()

    traces = load_traces(tracking_uri=args.tracking_uri, experiment=args.experiment, trace_files=args.trace_files)
    if args.export_dir:
        export_dir = Path(args.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        for trace in traces:
            (export_dir / f"{trace.info.trace_id}.json").write_text(trace.to_json(), encoding="utf-8")
    accepted, rejected = convert(traces, tools_fallback=load_tools(Path(args.tool_schema)), source=args.source)
    output = Path(args.output)
    rejected_path = output.with_suffix(".rejected.jsonl")
    _write_jsonl(output, accepted)
    _write_jsonl(rejected_path, rejected)
    print(json.dumps({
        "traces": len(traces),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "output": str(output),
        "rejected_output": str(rejected_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
