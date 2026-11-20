"""Rebuild training conversations from MLflow traces written to a temporary store."""
import json
import tempfile
import unittest
from pathlib import Path

import mlflow
from mlflow.entities import SpanType

from offline_mcp_synthesis import DEFAULT_SCHEMA, load_tools
from traces_to_training import convert, load_traces

SYSTEM = "You are an order-operations assistant."
QUERY = "Look up order ORD-1001 and create a UPS shipment if possible."
ORDER = {"id": "ORD-1001", "customer": "Acme Retail", "sku": "SKU-LAP-14", "qty": 2, "status": "paid", "ship_to": "Austin, TX"}
SHIPMENT = {"shipment_id": "SHP-0001", "order_id": "ORD-1001", "carrier": "UPS", "sku": "SKU-LAP-14", "qty": 2, "warehouse": "DFW"}
ANSWER = "ORD-1001 shipped via UPS as SHP-0001."


def _wrapped(payload):
    return json.dumps([{"type": "text", "text": json.dumps(payload)}])


def _record_trace(tools, *, model_saw_wrong_result=False):
    """Write one trace shaped like stateful_env.py runs: an AGENT root, LLM spans and TOOL spans.

    With model_saw_wrong_result, the messages sent to the model carry a get_order
    result that differs from what the TOOL span recorded.
    """
    calls = [
        ("call_a", "get_order", {"order_id": "ORD-1001"}, ORDER),
        ("call_b", "create_shipment", {"order_id": "ORD-1001", "carrier": "UPS"}, SHIPMENT),
    ]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUERY}]
    with mlflow.start_span(name="order_ops_agent", span_type=SpanType.AGENT) as root:
        root.set_inputs({"system_prompt": SYSTEM, "query": QUERY})
        for call_id, name, arguments, result in calls:
            assistant = {
                "content": None,
                "role": "assistant",
                "tool_calls": [{"function": {"arguments": json.dumps(arguments), "name": name}, "id": call_id, "type": "function"}],
                "function_call": None,
            }
            with mlflow.start_span(name="litellm-acompletion", span_type=SpanType.LLM) as llm:
                llm.set_inputs({"messages": list(messages), "tools": tools})
                llm.set_outputs({"choices": [{"index": 0, "message": assistant}]})
            messages.append(assistant)
            with mlflow.start_span(name=name, span_type=SpanType.TOOL) as tool:
                tool.set_attribute("tool_call_id", call_id)
                tool.set_inputs(arguments)
                tool.set_outputs(_wrapped(result))
            seen = dict(result, status="cancelled") if model_saw_wrong_result and name == "get_order" else result
            messages.append({"role": "tool", "tool_call_id": call_id, "content": _wrapped(seen)})
        final = {"content": ANSWER, "role": "assistant", "tool_calls": None}
        with mlflow.start_span(name="litellm-acompletion", span_type=SpanType.LLM) as llm:
            llm.set_inputs({"messages": list(messages), "tools": tools})
            llm.set_outputs({"choices": [{"index": 0, "message": final}]})
        root.set_outputs({"answer": ANSWER})
        return root.trace_id


class TracesToTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.tracking_uri = f"sqlite:///{Path(cls.tmp.name) / 'mlflow.db'}"
        mlflow.set_tracking_uri(cls.tracking_uri)
        mlflow.set_experiment("traces-to-training-test")
        cls.tools = load_tools(DEFAULT_SCHEMA)
        cls.good = _record_trace(cls.tools)
        cls.duplicate = _record_trace(cls.tools)
        cls.mismatched = _record_trace(cls.tools, model_saw_wrong_result=True)
        cls.traces = load_traces(tracking_uri=cls.tracking_uri, experiment="traces-to-training-test")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_rebuilds_from_llm_spans_and_checks_against_tool_spans(self):
        accepted, rejected = convert(self.traces, tools_fallback=self.tools)
        self.assertEqual(len(self.traces), 3)
        self.assertEqual(len(accepted), 1)
        stages = {row["trace_id"]: row["stage"] for row in rejected}
        self.assertEqual(stages[self.mismatched], "tool_spans")
        self.assertEqual(sorted(stages.values()), ["duplicate", "tool_spans"])

        record = accepted[0]
        self.assertEqual(record["rebuilt_from"], "llm_spans")
        self.assertEqual([m["role"] for m in record["messages"]], ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"])
        self.assertEqual(record["messages"][3], {"role": "tool", "tool_call_id": "call_a", "content": json.dumps(ORDER), "name": "get_order"})
        self.assertEqual(record["messages"][-1]["content"], ANSWER)
        self.assertEqual(record["tools"], self.tools)

    def test_rebuilds_from_tool_spans_only(self):
        accepted, rejected = convert(self.traces, tools_fallback=self.tools, source="tool-spans")
        self.assertEqual([row["rebuilt_from"] for row in accepted], ["tool_spans"])
        self.assertEqual(sorted(row["stage"] for row in rejected), ["duplicate", "duplicate"])
        self.assertEqual(accepted[0]["messages"][5]["content"], json.dumps(SHIPMENT))

    def test_reads_exported_trace_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for trace in self.traces:
                path = Path(tmp) / f"{trace.info.trace_id}.json"
                path.write_text(trace.to_json())
                paths.append(str(path))
            exported = load_traces(trace_files=paths)
        accepted, _ = convert(exported, tools_fallback=self.tools)
        self.assertEqual(len(exported), 3)
        self.assertEqual(len(accepted), 1)


if __name__ == "__main__":
    unittest.main()
