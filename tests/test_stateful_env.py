"""Exercise the stateful environment without calling a model."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

import mcp_server
from agent_traces import normalize_messages, tool_result_text
from offline_mcp_synthesis import DEFAULT_PROMPT, DEFAULT_SCHEMA, load_system_prompt, load_tools, validate_trajectory
from stateful_env import expected_state, load_scenario_states, run_server, state_changes

STATES = {state["name"]: state for state in load_scenario_states()}


def _initial(name):
    state = STATES[name]
    return {"orders": state["orders"], "inventory": state["inventory"], "shipments": []}


def _wrapped(payload):
    """A tool result as MCPAgentBlock records it: a JSON list of MCP content items."""
    return json.dumps([{"type": "text", "text": json.dumps(payload)}])


def _agent_messages(system_prompt):
    """Messages in the shape MCPAgentBlock records them (LiteLLM message dumps)."""
    def call(call_id, name, arguments):
        return {
            "content": None,
            "role": "assistant",
            "tool_calls": [{"function": {"arguments": json.dumps(arguments), "name": name}, "id": call_id, "type": "function"}],
            "function_call": None,
            "provider_specific_fields": {"refusal": None},
            "annotations": [],
        }

    order = STATES["baseline"]["orders"]["ORD-1001"]
    shipment = {"shipment_id": "SHP-0001", "order_id": "ORD-1001", "carrier": "UPS", "sku": "SKU-LAP-14", "qty": 2, "warehouse": "DFW"}
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Look up order ORD-1001 and create a UPS shipment if possible."},
        call("call_1", "get_order", {"order_id": "ORD-1001"}),
        {"role": "tool", "tool_call_id": "call_1", "content": _wrapped(order)},
        call("call_2", "create_shipment", {"order_id": "ORD-1001", "carrier": "UPS"}),
        {"role": "tool", "tool_call_id": "call_2", "content": _wrapped(shipment)},
        {"content": "Shipped ORD-1001 via UPS as SHP-0001.", "role": "assistant", "tool_calls": None, "function_call": None},
    ]


class StatefulEnvTests(unittest.TestCase):
    def tearDown(self):
        mcp_server.STATE_FILE = None
        mcp_server.reset_catalog()

    def test_scenario_states_load_and_reject_unknown_skus(self):
        self.assertEqual(set(STATES), {"baseline", "restocked_monitors", "unpaid_laptop_order", "keyboard_order_paid_low_stock"})
        bad = json.loads(json.dumps([STATES["baseline"]]))
        bad[0]["orders"]["ORD-1001"]["sku"] = "SKU-NONE"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "states.json"
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                load_scenario_states(path)

    def test_state_file_is_loaded_and_updated_after_a_shipment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text(json.dumps(_initial("restocked_monitors")))
            mcp_server.load_state(path)
            mcp_server.STATE_FILE = str(path)
            shipment = mcp_server.create_shipment("ORD-1002", "FedEx")
            saved = json.loads(path.read_text())
        self.assertEqual(shipment["order_id"], "ORD-1002")
        self.assertEqual(saved["orders"]["ORD-1002"]["status"], "shipped")
        self.assertEqual(saved["inventory"]["SKU-MON-27"]["on_hand"], 3)
        self.assertEqual(saved["shipments"], [shipment])

    def test_server_process_persists_writes_and_later_reads_see_them(self):
        async def ship_and_read(url):
            async with streamablehttp_client(url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    shipped = await session.call_tool("create_shipment", {"order_id": "ORD-1002", "carrier": "FedEx"})
                    order = await session.call_tool("get_order", {"order_id": "ORD-1002"})
                    return shipped.content[0].text, order.content[0].text

        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            state_file.write_text(json.dumps(_initial("restocked_monitors")))
            with run_server(state_file, Path(tmp) / "server.log") as url:
                shipped, order = asyncio.run(ship_and_read(url))
            saved = json.loads(state_file.read_text())
        self.assertEqual(json.loads(shipped)["order_id"], "ORD-1002")
        self.assertEqual(json.loads(order)["status"], "shipped")
        self.assertEqual(saved["inventory"]["SKU-MON-27"]["on_hand"], 3)

    def test_agent_messages_normalize_validate_and_explain_the_state_change(self):
        system_prompt = load_system_prompt(DEFAULT_PROMPT)
        messages = normalize_messages(_agent_messages(system_prompt))
        record = validate_trajectory({"messages": messages}, load_tools(DEFAULT_SCHEMA), system_prompt)
        self.assertEqual(record["messages"][3]["content"], json.dumps(STATES["baseline"]["orders"]["ORD-1001"]))
        self.assertEqual({key for message in record["messages"] for key in message}, {"role", "content", "tool_calls", "tool_call_id", "name"})

        initial = _initial("baseline")
        expected = expected_state(initial, record["messages"])
        self.assertEqual(expected["orders"]["ORD-1001"]["status"], "shipped")
        self.assertEqual(expected["inventory"]["SKU-LAP-14"]["on_hand"], 10)
        paths = [change["path"] for change in state_changes(initial, expected)]
        self.assertEqual(paths, ["orders.ORD-1001.status", "inventory.SKU-LAP-14.on_hand", "shipments"])

        unexplained = json.loads(json.dumps(expected))
        unexplained["inventory"]["SKU-LAP-14"]["on_hand"] = 12
        self.assertNotEqual(expected, unexplained)

    def test_tool_result_text_unwraps_mcp_content(self):
        self.assertEqual(tool_result_text(_wrapped({"a": 1})), '{"a": 1}')
        two = json.dumps([{"type": "text", "text": '{"a": 1}'}, {"type": "text", "text": '{"b": 2}'}])
        self.assertEqual(json.loads(tool_result_text(two)), [{"a": 1}, {"b": 2}])
        self.assertEqual(tool_result_text('{"plain": true}'), '{"plain": true}')


if __name__ == "__main__":
    unittest.main()
