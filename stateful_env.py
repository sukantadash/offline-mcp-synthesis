"""Run an agent against the Order Ops server with a fresh state per scenario.

For every (scenario state, seed query) pair this script:

1. writes the scenario's orders and inventory to a state file,
2. starts mcp_server.py with ORDER_OPS_STATE pointing at that file,
3. lets the teacher model act as the agent through sdg_hub's MCPAgentBlock,
4. reads the final state back and checks that every change is explained by a
   successful create_shipment result in the transcript,
5. records the conversation, the tool definitions, and the state before and after.

Tool results in accepted rows come from executing the tools, not from the model.
Runs are traced to MLflow (model calls plus a TOOL span per tool call) unless
--no-tracing is given; traces_to_training.py rebuilds training data from those traces.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator

from jsonschema import ValidationError

from agent_traces import normalize_messages
from offline_mcp_synthesis import (
    DEFAULT_PROMPT,
    DEFAULT_SCHEMA,
    DEFAULT_SEEDS,
    ROOT,
    _write_jsonl,
    load_seed_queries,
    load_system_prompt,
    load_tools,
    validate_trajectory,
)

DEFAULT_STATES = ROOT / "input" / "scenario_states.json"
DEFAULT_OUTPUT = ROOT / "output" / "stateful_env_dataset.jsonl"
DEFAULT_TRACKING_URI = f"sqlite:///{ROOT / 'output' / 'mlflow.db'}"
DEFAULT_EXPERIMENT = "order-ops-stateful-env"
SERVER = ROOT / "mcp_server.py"


def load_scenario_states(path: Path | str = DEFAULT_STATES) -> list[dict]:
    """Load named scenario states and check that every order's SKU is in the inventory."""
    states = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(states, list) or not states:
        raise ValueError(f"{path}: expected a nonempty list of scenario states")
    names: set[str] = set()
    for state in states:
        name = state.get("name")
        if not name or name in names:
            raise ValueError(f"{path}: scenario names must be unique and nonempty, got {name!r}")
        names.add(name)
        for order_id, order in state["orders"].items():
            if order["id"] != order_id or order["sku"] not in state["inventory"]:
                raise ValueError(f"{path}: scenario {name}: order {order_id} has a mismatched id or an unknown SKU")
    return states


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def run_server(state_file: Path, log_file: Path, timeout: float = 30.0) -> Iterator[str]:
    """Start mcp_server.py on a free port with the given state file and yield its MCP URL."""
    port = _free_port()
    env = dict(os.environ, PORT=str(port), HOST="127.0.0.1", ORDER_OPS_STATE=str(state_file))
    with open(log_file, "w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, str(SERVER)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"mcp_server.py exited with code {process.returncode}; see {log_file}")
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"mcp_server.py did not accept connections within {timeout}s; see {log_file}") from None
                    time.sleep(0.2)
            yield f"http://127.0.0.1:{port}/mcp"
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def expected_state(initial: dict, messages: list[dict]) -> dict:
    """Apply the successful create_shipment results in a transcript to the initial state."""
    state = copy.deepcopy(initial)
    state["shipments"] = []
    for message in messages:
        if message["role"] != "tool" or message.get("name") != "create_shipment":
            continue
        result = json.loads(message["content"])
        if "shipment_id" not in result:
            continue
        order = state["orders"][result["order_id"]]
        order["status"] = "shipped"
        state["inventory"][order["sku"]]["on_hand"] -= order["qty"]
        state["shipments"].append(result)
    return state


def state_changes(before: dict, after: dict) -> list[dict]:
    """List field-level changes to orders and inventory, followed by new shipments."""
    changes = []
    for section in ("orders", "inventory"):
        for key in sorted(set(before[section]) | set(after[section])):
            old, new = before[section].get(key, {}), after[section].get(key, {})
            for field in sorted(set(old) | set(new)):
                if old.get(field) != new.get(field):
                    changes.append({"path": f"{section}.{key}.{field}", "before": old.get(field), "after": new.get(field)})
    for shipment in after.get("shipments", [])[len(before.get("shipments", [])):]:
        changes.append({"path": "shipments", "before": None, "after": shipment})
    return changes


def _agent_block_class(tracing: bool):
    """Return MCPAgentBlock, or a subclass that records each MCP tool call as an MLflow TOOL span."""
    from sdg_hub.core.blocks import MCPAgentBlock

    if not tracing:
        return MCPAgentBlock

    import mlflow
    from mlflow.entities import SpanType

    class TracedMCPAgentBlock(MCPAgentBlock):
        # mlflow.litellm.autolog() traces the model calls. MCPAgentBlock's MCP tool
        # calls are not instrumented, so each one is wrapped in a TOOL span here.
        async def _execute_tool_calls(self, session, tool_calls, messages):
            for call in tool_calls:
                with mlflow.start_span(name=call.function.name, span_type=SpanType.TOOL) as span:
                    span.set_attribute("tool_call_id", call.id)
                    try:
                        span.set_inputs(json.loads(call.function.arguments or "{}"))
                    except json.JSONDecodeError:
                        span.set_inputs({"raw_arguments": call.function.arguments})
                    await super()._execute_tool_calls(session, [call], messages)
                    span.set_outputs(messages[-1]["content"])

    return TracedMCPAgentBlock


def run(
    states: list[dict],
    seed_queries: list[str],
    system_prompt: str,
    tools: list[dict],
    output: Path,
    *,
    model: str,
    api_key: str | None = None,
    api_base: str | None = None,
    max_iterations: int = 10,
    tracing: bool = True,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment: str = DEFAULT_EXPERIMENT,
) -> dict:
    """Run every scenario state against every seed query and write accepted and rejected rows."""
    import pandas as pd

    output = Path(output)
    rejected_path = output.with_suffix(".rejected.jsonl")
    log_dir = output.parent / "stateful_env_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if tracing:
        import mlflow
        from mlflow.entities import SpanType

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        mlflow.litellm.autolog()
    agent_class = _agent_block_class(tracing)

    accepted: list[dict] = []
    rejected: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        for state in states:
            for query_index, query in enumerate(seed_queries):
                run_name = f"{state['name']}__q{query_index}"
                initial = {"orders": state["orders"], "inventory": state["inventory"], "shipments": []}
                state_file = Path(tmp) / f"{run_name}.json"
                state_file.write_text(json.dumps(initial, indent=2), encoding="utf-8")
                trace_id = None
                with run_server(state_file, log_dir / f"{run_name}.log") as url:
                    block = agent_class(
                        block_name="order_ops_agent",
                        mcp_server_url=url,
                        model=model,
                        api_key=api_key,
                        api_base=api_base,
                        system_prompt=system_prompt,
                        input_cols=["query"],
                        output_cols=["agent_trace"],
                        max_iterations=max_iterations,
                    )
                    frame = pd.DataFrame({"query": [query]})
                    if tracing:
                        with mlflow.start_span(name="order_ops_agent", span_type=SpanType.AGENT) as root:
                            root.set_inputs({"system_prompt": system_prompt, "query": query, "scenario": state["name"]})
                            agent_trace = block(frame)["agent_trace"].iloc[0]
                            root.set_outputs({"answer": agent_trace["messages"][-1].get("content")})
                            trace_id = root.trace_id
                    else:
                        agent_trace = block(frame)["agent_trace"].iloc[0]
                final = json.loads(state_file.read_text(encoding="utf-8"))

                base = {"scenario": state["name"], "seed_query": query, "mlflow_trace_id": trace_id}
                try:
                    messages = normalize_messages(agent_trace["messages"])
                    record = validate_trajectory({"messages": messages}, tools, system_prompt)
                except (ValueError, TypeError, KeyError, ValidationError, json.JSONDecodeError) as error:
                    rejected.append({**base, "stage": "structure", "error": str(error), "raw_messages": agent_trace["messages"]})
                    continue
                expected = expected_state(initial, record["messages"])
                if expected != final:
                    rejected.append({
                        **base,
                        "stage": "state",
                        "error": "final state does not match the successful writes in the transcript",
                        "unexplained_changes": state_changes(expected, final),
                        "messages": record["messages"],
                    })
                    continue
                record.update(
                    base,
                    tools=tools,
                    scenario_description=state.get("description", ""),
                    initial_state=initial,
                    final_state=final,
                    state_changes=state_changes(initial, final),
                    agent_iterations=agent_trace["iterations"],
                    mode="stateful_mock_env",
                    agent_model=model,
                )
                accepted.append(record)

    _write_jsonl(output, accepted)
    _write_jsonl(rejected_path, rejected)
    return {
        "requested": len(states) * len(seed_queries),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "output": str(output),
        "rejected_output": str(rejected_path),
        "mlflow_tracking_uri": tracking_uri if tracing else None,
        "mlflow_experiment": experiment if tracing else None,
    }


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run the agent against a fresh Order Ops state per scenario.")
    parser.add_argument("--states", default=str(DEFAULT_STATES), help="Scenario states JSON")
    parser.add_argument("--seeds", default=str(DEFAULT_SEEDS), help="Seed queries JSON")
    parser.add_argument("--system-prompt", default=str(DEFAULT_PROMPT), help="System prompt file")
    parser.add_argument("--tool-schema", default=str(DEFAULT_SCHEMA), help="Tool schema used to validate calls")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Accepted rows JSONL")
    parser.add_argument("--max-iterations", type=int, default=10, help="Agent loop limit per run")
    parser.add_argument("--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    parser.add_argument("--mlflow-experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--no-tracing", action="store_true", help="Do not record MLflow traces")
    args = parser.parse_args()

    model = os.getenv("TEACHER_MODEL_NAME") or os.getenv("MODEL", "")
    if not model:
        sys.exit("Set TEACHER_MODEL_NAME in .env; the teacher model acts as the agent")
    report = run(
        load_scenario_states(args.states),
        load_seed_queries(Path(args.seeds)),
        load_system_prompt(Path(args.system_prompt)),
        load_tools(Path(args.tool_schema)),
        Path(args.output),
        model=model,
        api_key=os.getenv("TEACHER_API_KEY") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("TEACHER_MODEL_URL") or os.getenv("API_BASE"),
        max_iterations=args.max_iterations,
        tracing=not args.no_tracing,
        tracking_uri=args.mlflow_tracking_uri,
        experiment=args.mlflow_experiment,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
